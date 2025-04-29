import streamlit as st
import io
import os
from datetime import datetime
import pandas as pd
import uuid 
from utils.helpers import get_file_extension, generate_uuid, get_current_timestamp
from core.ocr_processor import OCRProcessor
from core.vector_db import VectorDB
from core.llm_analyzer import LLMAnalyzer
from core.pdf_generator import PDFGenerator 
from core.user_manager import UserManager, USER_DB_PATH # Import UserManager and DB path


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Medical Report Analyzer (AI Assistant)", layout="wide", initial_sidebar_state="expanded")

st.title("🩺 Medical Report Analyzer (AI Assistant)")
st.markdown("""
This application uses AI to help you understand your medical reports (Blood work, Scans, etc.) by extracting text,
storing it in a personal profile, and providing a simplified interpretation, guidance, and diet suggestions.
You can also ask follow-up questions about your uploaded reports.

**Disclaimer:** This is an AI-generated analysis and is NOT a substitute for professional medical advice.
Always consult with a qualified healthcare professional.
""")

# --- Session State Initialization ---
# These keys track the application's state across reruns
if 'openai_key' not in st.session_state:
    st.session_state.openai_key = ''
if 'current_user' not in st.session_state:
    st.session_state.current_user = None # Stores the dictionary of the current user profile
if 'user_reports_metadata' not in st.session_state:
     # Stores a dict of report metadata {report_id: {metadata}} for the current user, loaded from DB
     st.session_state.user_reports_metadata = {}
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None # Extracted text of the active report
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None # AI analysis result of the active report
if 'qna_history' not in st.session_state:
    st.session_state.qna_history = [] # History of Q&A for the active report/context
if 'active_report_id' not in st.session_state:
     st.session_state.active_report_id = None # The unique ID of the report currently loaded/analyzed
if 'pdf_preview_images' not in st.session_state:
    st.session_state.pdf_preview_images = [] # Images of PDF pages for preview
if 'processing_upload' not in st.session_state:
    st.session_state.processing_upload = False # Flag to prevent duplicate upload processing on rerun

# --- Session state for the Q&A input box value ---
if 'current_qna_question_value' not in st.session_state:
    st.session_state.current_qna_question_value = ""




@st.cache_resource(hash_funcs={UserManager: id})
def init_user_manager():
     """Initializes and caches the UserManager."""
     return UserManager()


@st.cache_resource(hash_funcs={OCRProcessor: id})
def init_ocr_processor(api_key):
     """Initializes and caches the OCRProcessor."""
     if not api_key:
         return None
     try:
         return OCRProcessor(api_key)
     except Exception as e:
         st.sidebar.error(f"Failed to initialize OCRProcessor: {e}") # Display error in sidebar
         return None


@st.cache_resource(hash_funcs={VectorDB: id})
def init_vector_db(api_key):
    """Initializes and caches the VectorDB."""
    if not api_key:
         return None
    try:
        return VectorDB(api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize VectorDB: {e}") # Display error in sidebar
        return None


@st.cache_resource(hash_funcs={LLMAnalyzer: id, VectorDB: id})
def init_llm_analyzer(api_key, _vector_db_instance):
     """Initializes and caches the LLMAnalyzer."""
     if not api_key:
         return None
     try:
         return LLMAnalyzer(api_key, _vector_db_instance)
     except Exception as e:
         st.sidebar.error(f"Failed to initialize LLMAnalyzer: {e}") # Display error in sidebar
         return None

@st.cache_resource(hash_funcs={PDFGenerator: id})
def init_pdf_generator():
    """Initializes and caches the PDFGenerator."""
    try:
        return PDFGenerator()
    except Exception as e:
        st.sidebar.error(f"Failed to initialize PDFGenerator: {e}") # Display error in sidebar
        return None


# --- Sidebar: API Key Input (Always Displayed) ---
st.sidebar.header("1. OpenAI API Key")
# Text input for the API key, using session state to persist value. This is ALWAYS rendered.
st.session_state.openai_key = st.sidebar.text_input("Enter OpenAI API Key", type="password", value=st.session_state.openai_key, key="sidebar_api_key_input")


user_manager = init_user_manager() # User manager doesn't need API key
pdf_generator = init_pdf_generator() # PDF Generator doesn't need API key

ocr_processor = None
vector_db = None
llm_analyzer = None

if st.session_state.openai_key:
    # Attempt to initialize AI-dependent components if the key is present
    ocr_processor = init_ocr_processor(st.session_state.openai_key)
    vector_db = init_vector_db(st.session_state.openai_key)
    # Only initialize LLM Analyzer if VectorDB was successfully initialized
    if vector_db and vector_db.is_available():
        llm_analyzer = init_llm_analyzer(st.session_state.openai_key, vector_db)


# --- Check Component Readiness ---
# Check if ALL essential components (AI and PDF) are successfully initialized.
components_ready = ocr_processor is not None and vector_db is not None and llm_analyzer is not None \
                   and vector_db.is_available() and (llm_analyzer is None or llm_analyzer.analysis_chain is not None) \
                   and pdf_generator is not None

# Display overall component status message in sidebar
if not st.session_state.openai_key:
    st.sidebar.info("Please enter your OpenAI API key to enable AI features.")
elif not components_ready:
    st.sidebar.warning("Core components are initializing or failed to load. Check error messages above.")
else:
     st.sidebar.success("All components ready.")


# --- Sidebar: Profile Management (Always Displayed) ---
st.sidebar.header("2. Your Profile")

with st.sidebar.form("profile_form"):
    st.write("Create a new profile or load an existing one.")
    # Use keys to manage form state independently across reruns
    form_name = st.text_input("Name", key="profile_form_name")
    form_age = st.number_input("Age", min_value=0, max_value=120, step=1, key="profile_form_age")
    form_gender = st.selectbox("Gender", ["", "Male", "Female", "Other", "Prefer not to say"], key="profile_form_gender_select")
    submit_profile = st.form_submit_button("Create/Load Profile")

    if submit_profile:
        if not form_name or form_age <= 0 or not form_gender:
            st.error("Please enter Name, valid Age, and select Gender.")
        else:
            # Try to find existing user
            found_user = user_manager.find_user(form_name, form_age, form_gender)
            if found_user:
                st.session_state.current_user = found_user
                st.success(f"Loaded profile for {st.session_state.current_user['name']}.")
                # Load user's report history metadata from the vector DB ONLY if DB is available
                if vector_db and vector_db.is_available():
                     st.session_state.user_reports_metadata = vector_db.get_user_reports_metadata(st.session_state.current_user['id'])
                else:
                     st.session_state.user_reports_metadata = {} # Cannot load history if DB is down

                # Clear state related to previous interactions
                st.session_state.qna_history = []
                st.session_state.extracted_text = None
                st.session_state.analysis_result = None
                st.session_state.active_report_id = None
                st.session_state.pdf_preview_images = []
                st.session_state.processing_upload = False # Reset upload flag
                st.session_state.current_qna_question_value = "" # Clear Q&A input state

                st.rerun() # Rerun to update the main content based on the loaded user

            else:
                # Create new user
                new_user = user_manager.add_user(form_name, form_age, form_gender)
                st.session_state.current_user = new_user
                st.success(f"New profile created for {st.session_state.current_user['name']}.")
                st.session_state.user_reports_metadata = {} # New user has no reports yet

                # Clear state related to previous interactions
                st.session_state.qna_history = []
                st.session_state.extracted_text = None
                st.session_state.analysis_result = None
                st.session_state.active_report_id = None
                st.session_state.pdf_preview_images = []
                st.session_state.processing_upload = False # Reset upload flag
                st.session_state.current_qna_question_value = "" # Clear Q&A input state

                st.rerun() # Rerun to update the main content

# --- Display Current User and Delete Option (Always Displayed if User is Loaded) ---
if st.session_state.current_user:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Active Profile:")
    st.sidebar.write(f"**Name:** {st.session_state.current_user.get('name', 'N/A')}")
    st.sidebar.write(f"**Age:** {st.session_state.current_user.get('age', 'N/A')}")
    st.sidebar.write(f"**Gender:** {st.session_state.current_user.get('gender', 'N/A')}")

    # Delete Profile Button
    st.sidebar.markdown("---")
    # Add confirmation step outside the button for clearer UI flow
    confirm_delete = st.sidebar.checkbox("Confirm Profile Deletion", key="confirm_delete_checkbox", help="Check this box and click the button to delete.")
    if st.sidebar.button("Delete Active Profile (and all data)", help="This will permanently delete the current user profile and all associated reports from the database."):
         if confirm_delete:
              user_id_to_delete = st.session_state.current_user['id']
              st.sidebar.warning(f"Proceeding with deletion of profile for {st.session_state.current_user['name']} (ID: {user_id_to_delete}). This action is irreversible.")

              with st.spinner("Deleting user data..."):
                   # Delete from ChromaDB first ONLY if DB is available
                   chroma_deleted = False
                   if vector_db and vector_db.is_available():
                        chroma_deleted = vector_db.delete_user_data(user_id_to_delete)
                   else:
                        st.warning("ChromaDB not available. Could not delete data from VectorDB. Profile will be removed from list.")
                        chroma_deleted = True # Proceed with user deletion even if DB delete failed

                   # Delete from User Manager (JSON)
                   user_deleted = user_manager.delete_user(user_id_to_delete)

                   if user_deleted: # User must be deleted from JSON
                      if chroma_deleted:
                         st.sidebar.success(f"Profile and all associated data for {st.session_state.current_user['name']} successfully deleted.")
                      # else case handled above if DB not available

                      # Clear session state for the user
                      st.session_state.current_user = None
                      st.session_state.user_reports_metadata = {}
                      st.session_state.extracted_text = None
                      st.session_state.analysis_result = None
                      st.session_state.qna_history = []
                      st.session_state.active_report_id = None
                      st.session_state.pdf_preview_images = []
                      st.session_state.processing_upload = False # Reset upload flag
                      st.session_state.current_qna_question_value = "" # Clear Q&A input state
                      # st.session_state.confirm_delete_checkbox = False # Cannot uncheck checkbox state directly in rerun

                      st.rerun() # Rerun to clear the main page and sidebar profile display
                   else:
                      st.sidebar.error("Failed to delete user profile from the database.")
         else:
             st.sidebar.warning("Please confirm deletion by checking the box.")


# --- Sidebar: User History Dashboard (Always Displayed) ---
st.sidebar.header("3. User History")
all_users = user_manager.get_all_users()

if all_users:
    # Create a dictionary for the selectbox {Display Name: User ID}
    user_options = {f"{user.get('name', 'Unknown')} (Age: {user.get('age', 'N/A')})": user.get('id') for user in all_users}

    # Find the current user's display string to set as default in the selectbox
    current_user_display = ""
    if st.session_state.current_user:
         for display_str, user_id in user_options.items():
              if user_id == st.session_state.current_user['id']:
                   current_user_display = display_str
                   break # Found the current user in the options

    # Add a default "Select Profile" option
    select_box_options = ["Select or Create Profile"] + list(user_options.keys())
    # Set the default value based on whether a user is currently logged in
    # Need to find the index of the current user display string in the options list
    try:
         default_index = select_box_options.index(current_user_display) if current_user_display else 0
    except ValueError:
         # This might happen if the current user was deleted externally but still in session state
         default_index = 0
         # Clear current user state if they are not found in the loaded list of users
         if st.session_state.current_user:
              st.warning(f"Active user '{st.session_state.current_user.get('name', 'Unknown')}' not found in user list. Clearing session state.")
              st.session_state.current_user = None
              st.session_state.user_reports_metadata = {} # Clear history too


    selected_user_display = st.sidebar.selectbox(
        "Load Existing Profile",
        select_box_options,
        index=default_index, # Set default
        key="sidebar_profile_selector" # Add key
    )

    # Handle selection change only if a profile is selected and it's different from the current one
    if selected_user_display != "Select or Create Profile":
         selected_user_id = user_options[selected_user_display]
         if st.session_state.current_user is None or st.session_state.current_user['id'] != selected_user_id:
              # Load the selected user's profile
              user_to_load = user_manager.get_user_by_id(selected_user_id)
              if user_to_load:
                   st.session_state.current_user = user_to_load
                   st.success(f"Switched to profile for {st.session_state.current_user['name']}.")
                   # Load their report history metadata ONLY if DB is available
                   if vector_db and vector_db.is_available():
                       st.session_state.user_reports_metadata = vector_db.get_user_reports_metadata(st.session_state.current_user['id'])
                   else:
                       st.session_state.user_reports_metadata = {} # Cannot load history if DB is down

                   # Clear state related to previous interactions
                   st.session_state.qna_history = []
                   st.session_state.extracted_text = None
                   st.session_state.analysis_result = None
                   st.session_state.active_report_id = None
                   st.session_state.pdf_preview_images = []
                   st.session_state.processing_upload = False # Reset upload flag
                   st.session_state.current_qna_question_value = "" # Clear Q&A input state


                   st.rerun() # Rerun to update main page with new user data
              else:
                   st.sidebar.error("Could not load selected user profile.")

else:
    st.sidebar.info("No user profiles found. Create one above.")



if st.session_state.current_user is None:
    st.info("Please create or load a user profile in the sidebar to continue.")
elif not components_ready:
    st.warning("Core components (AI analysis, embedding, PDF generation) are not available. Please ensure your OpenAI API key is entered and check potential errors in the sidebar.")
else: # User is logged in AND components are ready
    st.header(f"Profile: {st.session_state.current_user.get('name', 'Unknown User')}")


    # --- Report History for Current User ---
    st.subheader("Your Reports History")
    if st.session_state.user_reports_metadata:
         # Convert metadata dict to a list of dicts for pandas
         reports_list = list(st.session_state.user_reports_metadata.values())
         # Create a pandas DataFrame for better display
         df_history = pd.DataFrame(reports_list)
         # Ensure required columns exist, provide defaults if missing (important if metadata structure changes)
         required_cols = ['upload_date', 'report_type', 'filename', 'report_id']
         for col in required_cols:
              if col not in df_history.columns:
                   df_history[col] = 'N/A'

         # Rearrange columns and format date (optional formatting)
         df_history = df_history[required_cols]
         # Ensure column names match exactly what's in the DataFrame before renaming
         if 'report_type' in df_history.columns:
             df_history.rename(columns={'report_type': 'Type'}, inplace=True)
         if 'filename' in df_history.columns:
              df_history.rename(columns={'filename': 'File Name'}, inplace=True)
         if 'upload_date' in df_history.columns:
              df_history.rename(columns={'upload_date': 'Date'}, inplace=True)
         if 'report_id' in df_history.columns:
              df_history.rename(columns={'report_id': 'Report ID'}, inplace=True)


         # Display a subset of columns in the table for brevity
         display_cols = ['Date', 'Type', 'File Name']
         # Ensure display_cols exist in the dataframe after potential renaming
         display_cols = [col for col in display_cols if col in df_history.columns]
         st.dataframe(df_history[display_cols], height=200, use_container_width=True, hide_index=True)

         # Option to select a historical report for viewing/Q&A
         # Create options showing Date - Type (File Name)
         # Ensure 'Type' and 'File Name' exist after renaming
         if 'Type' in df_history.columns and 'File Name' in df_history.columns:
             history_options_display = [f"{row['Date']} - {row['Type']} ({row['File Name']})" for index, row in df_history.iterrows()]
         else:
             # Fallback if renaming failed or columns missing
             history_options_display = [f"Report ID: {row['Report ID']}" for index, row in df_history.iterrows()]

         history_report_ids = df_history['Report ID'].tolist() # Corresponding Report IDs

         # Add a default "Select a report" option
         select_box_options = ["Select a historical report to view or interact with"] + history_options_display

         # Find the index of the currently active report (if any) in the select box options
         current_active_index = 0 # Default to the "Select..." option
         if st.session_state.active_report_id:
              try:
                   # Find the display string for the active report ID using the metadata stored in session state
                   active_report_meta = st.session_state.user_reports_metadata.get(st.session_state.active_report_id)
                   if active_report_meta:
                       # Construct the expected display string using consistent keys/renaming
                       active_display_str = f"{active_report_meta.get('upload_date', 'N/A')} - {active_report_meta.get('report_type', 'Unknown')} ({active_report_meta.get('filename', 'N/A')})"
                       # Find the index of this string in the *current* history options display list
                       if active_display_str in history_options_display: # Check if the active report is still in the history list
                            current_active_index = history_options_display.index(active_display_str) + 1 # +1 because of "Select a report" option at index 0
              except ValueError:
                   # Active report not found in history options display (e.g., data deleted externally), keep default index
                   current_active_index = 0
                   st.session_state.active_report_id = None # Clear active state if not found


         selected_history_report_display = st.selectbox(
             "Load a Previous Report:",
             select_box_options,
             index=current_active_index, # Set default
             key="history_report_selector" # Add key for selectbox
         )

         # Process selection change ONLY if Vector DB is available (needed to retrieve text)
         if vector_db and vector_db.is_available():
             if selected_history_report_display != "Select a historical report to view or interact with":
                 # Find the report_id corresponding to the selected display string
                 try:
                      # The index in history_options_display is the index in history_report_ids
                      selected_history_report_id = history_report_ids[history_options_display.index(selected_history_report_display)]
                 except ValueError:
                      st.error("Could not find selected historical report ID in the list.")
                      selected_history_report_id = None # Indicate error state


                 if selected_history_report_id and st.session_state.active_report_id != selected_history_report_id:
                     # Load the selected report's data
                     with st.spinner(f"Loading data for report: {selected_history_report_display}..."):
                          # Retrieve the full text from the vector DB
                          retrieved_text = vector_db.get_report_text_by_id(
                               user_id=st.session_state.current_user['id'],
                               report_id=selected_history_report_id
                          )

                          if retrieved_text:
                               st.session_state.extracted_text = retrieved_text
                               # Re-run the analysis for the loaded report text using the LLM analyzer ONLY if LLM is ready
                               if llm_analyzer and llm_analyzer.analysis_chain:
                                    st.session_state.analysis_result = llm_analyzer.analyze_report(st.session_state.extracted_text)
                               else:
                                   st.warning("AI Analyzer not available. Cannot perform analysis for loaded report.")
                                   st.session_state.analysis_result = "AI Analysis not available."


                               st.session_state.active_report_id = selected_history_report_id # Set as active
                               # Clear Q&A history as context is changing
                               st.session_state.qna_history = []
                               st.session_state.pdf_preview_images = [] # Clear PDF preview from previous state (preview is only for new uploads)
                               st.session_state.processing_upload = False # Reset upload flag
                               st.session_state.current_qna_question_value = "" # Clear Q&A input state


                               st.success(f"Loaded report: {selected_history_report_display}.")
                               # Rerun to show the loaded report/analysis and Q&A sections
                               st.rerun()
                          else:
                               st.error(f"Could not load data for report: {selected_history_report_display}. Data might be missing from the database.")
                               # Clear active report state if loading failed
                               st.session_state.active_report_id = None
                               st.session_state.extracted_text = None
                               st.session_state.analysis_result = None
                               st.session_state.qna_history = []
                               st.session_state.pdf_preview_images = []
                               st.session_state.processing_upload = False # Reset upload flag
                               st.session_state.current_qna_question_value = "" # Clear Q&A input state

             # Handle case where "Select a report" is selected and it's currently active
             elif selected_history_report_display == "Select a historical report to view or interact with" and st.session_state.active_report_id is not None:
                  # Clear active report state
                  st.session_state.active_report_id = None
                  st.session_state.extracted_text = None
                  st.session_state.analysis_result = None
                  st.session_state.qna_history = []
                  st.session_state.pdf_preview_images = []
                  st.session_state.processing_upload = False # Reset upload flag
                  st.session_state.current_qna_question_value = "" # Clear Q&A input state
                  st.info("Cleared active report.") # Optional message
                  st.rerun()

         else: # Vector DB not available
              st.warning("Vector Database is not ready. Cannot load report data from history.")


         st.markdown("---") # Separator


    else:
         st.info("No reports uploaded yet for this profile. Upload one below.")
         st.markdown("---") # Separator


    # --- Upload New Report ---
    st.subheader("Upload a New Medical Report")
    # Use a unique key per user/upload count to ensure the uploader resets correctly
    # Using user ID and the number of reports in history to make key unique
    upload_key = f"new_report_uploader_{st.session_state.current_user['id']}_{len(st.session_state.user_reports_metadata)}"

    uploaded_file = st.file_uploader("Choose a file (PNG, JPG, PDF)", type=["png", "jpg", "jpeg", "pdf"], key=upload_key)

    # Manual Report Type Selection - Ensure this is shown before upload processing
    report_type = st.selectbox(
        "Select Report Type",
        ["Unknown", "Blood Test", "Radiology Scan (X-Ray, CT, MRI)", "Pathology Report", "Specialist Letter", "Other"],
        key="report_type_selector" # Add key
    )

    # Process the uploaded file only if a file is present and we are not already processing
    if uploaded_file is not None and not st.session_state.processing_upload:
        st.session_state.processing_upload = True # Set flag immediately to prevent re-entry

        # Check if AI components are ready for processing
        if not components_ready:
            st.error("AI components are not ready to process your upload. Please ensure your API key is entered and check sidebar messages for errors.")
            st.session_state.processing_upload = False # Reset flag as processing didn't start
            st.rerun() # Rerun to clear upload state and display error
        else:
            file_bytes = uploaded_file.getvalue()
            file_type = uploaded_file.type
            file_extension = get_file_extension(uploaded_file.name)

            # Clear previous active report state before processing new one
            st.session_state.extracted_text = None
            st.session_state.analysis_result = None
            st.session_state.qna_history = []
            st.session_state.active_report_id = None # No active report until successfully stored
            st.session_state.pdf_preview_images = [] # Clear previous preview
            st.session_state.current_qna_question_value = "" # Clear Q&A input state for new report


            st.info(f"Processing new upload: {uploaded_file.name} ({uploaded_file.type})...")

            # --- Display PDF Preview if it's a PDF ---
            if file_type == "application/pdf":
                 # Only attempt PDF preview if OCR processor is available
                 if ocr_processor:
                      with st.spinner("Generating PDF page previews..."):
                           st.session_state.pdf_preview_images = ocr_processor.get_pdf_images(file_bytes)

                      if st.session_state.pdf_preview_images:
                          st.subheader("PDF Preview:")
                          # Display images in columns
                          cols = st.columns(4) # Adjust number of columns as needed
                          for i, img in enumerate(st.session_state.pdf_preview_images):
                              # Create a small container for each image to manage size/spacing
                              with cols[i % 4]: # Cycle through columns
                                  st.image(img, caption=f"Page {i+1}", use_column_width="always")
                          st.markdown("---") # Separator
                 else:
                     st.warning("OCR Processor not available. Cannot generate PDF page previews.")


            # --- Step 3: Extract Text ---
            # Only attempt text extraction if OCR processor is available
            if ocr_processor:
                 with st.spinner("Extracting text from the report..."):
                     st.session_state.extracted_text = ocr_processor.process(file_bytes, file_type)
            else:
                 st.error("OCR Processor not available. Cannot extract text.")
                 st.session_state.extracted_text = None # Ensure it's None if OCR failed


            if st.session_state.extracted_text:
                # Optionally show raw text for debugging
                # st.subheader("Extracted Text Preview:")
                # st.text_area("Raw Extracted Text", st.session_state.extracted_text, height=300, disabled=True)
                st.write("Text extraction successful. Proceeding to analysis and storage.")

                # --- Step 4: Store in Vector DB ---
                # Only attempt storage if Vector DB is available
                if vector_db and vector_db.is_available():
                     report_id = generate_uuid() # Generate unique ID for this new report
                     upload_date = get_current_timestamp() # Get current timestamp
                     report_metadata_to_store = {
                         "filename": uploaded_file.name,
                         "file_type": file_type,
                         "report_type": report_type, # Use the selected type
                         "upload_date": upload_date,
                         "report_id": report_id # Include report_id in metadata upfront
                     }

                     with st.spinner("Storing data embeddings..."):
                         # Call vector_db.add_document and CHECK its return value
                         added_doc_ids = vector_db.add_document(
                              user_id=st.session_state.current_user['id'],
                              report_id=report_id, # Use the generated report_id
                              report_metadata=report_metadata_to_store, # Pass the metadata dictionary
                              text=st.session_state.extracted_text
                         )

                     # --- IMPORTANT: Check if data was successfully added before proceeding ---
                     if added_doc_ids: # add_document returns list of IDs if successful, empty list if failed
                          st.success(f"Report data successfully stored in vector database (as {len(added_doc_ids)} chunks).")
                          st.session_state.active_report_id = report_id # Set the new report as active
                          # Update user's report history metadata in session state
                          # This is only added to session state IF storage was successful
                          st.session_state.user_reports_metadata[st.session_state.active_report_id] = report_metadata_to_store

                          # --- Step 5: Analyze with LLM ---
                          # Only attempt analysis if LLM Analyzer component is ready
                          if llm_analyzer and llm_analyzer.analysis_chain:
                               with st.spinner("Analyzing report text with AI..."):
                                    # Run analysis using the LLM analyzer
                                    st.session_state.analysis_result = llm_analyzer.analyze_report(st.session_state.extracted_text)

                               if st.session_state.analysis_result:
                                    st.success("Initial AI analysis complete.")
                               else:
                                    st.error("AI analysis failed after text extraction.")
                                    # Clear analysis result state if LLM failed
                                    st.session_state.analysis_result = None
                          else:
                              st.warning("AI Analyzer not available. Cannot perform analysis.")
                              st.session_state.analysis_result = "AI Analysis not available." # Set a placeholder message


                     else:
                          # Handle the case where add_document failed
                          st.error("Failed to store report data in vector database. Analysis cannot proceed.")
                          # Clear extracted text and analysis as storage failed
                          st.session_state.extracted_text = None
                          st.session_state.analysis_result = None
                          st.session_state.active_report_id = None # Ensure no report is active if storage fails
                          # Do NOT add to user_reports_metadata if storage failed
                else:
                     st.error("Vector Database not available. Cannot store report data.")
                     # Clear extracted text and analysis as storage failed
                     st.session_state.extracted_text = None
                     st.session_state.analysis_result = None
                     st.session_state.active_report_id = None # Ensure no report is active if storage fails


            else:
                st.error("Text extraction failed. Could not process the report.")
                # Clear all state related to this failed upload
                st.session_state.extracted_text = None
                st.session_state.analysis_result = None
                st.session_state.active_report_id = None
                st.session_state.pdf_preview_images = [] # Clear preview images if OCR failed
                st.session_state.current_qna_question_value = "" # Clear Q&A input state


            # Unset the processing flag regardless of success/failure in processing steps
            st.session_state.processing_upload = False

            # Rerun the app after processing to update the UI with analysis/Q&A/history based on new state
            st.rerun()

    # --- Display Active Report Info (Analysis, Q&A) ---
    # This section is displayed if a report is currently active (either just uploaded successfully or loaded from history)
    # Check if active_report_id is set and if that ID exists in the user's reports metadata (ensures it's a valid report)
    if st.session_state.active_report_id and st.session_state.active_report_id in st.session_state.user_reports_metadata:
         # Get metadata for the currently active report from session state history
         current_report_metadata = st.session_state.user_reports_metadata.get(st.session_state.active_report_id, {})
         report_display_name = current_report_metadata.get('filename', 'Currently Loaded Report')

         # --- Step 5/Show Analysis ---
         if st.session_state.analysis_result:
             st.subheader(f"AI Analysis for Report: {report_display_name}")
             # Display the analysis result formatted with markdown
             st.markdown(st.session_state.analysis_result)

             # --- Step 6: Generate Downloadable Report ---
             st.subheader("Download Analysis Summary")
             # Check if analysis was successfully generated AND PDF generator is available
             if st.session_state.analysis_result and st.session_state.analysis_result != "AI Analysis not available." and pdf_generator:
                  if st.button("Generate and Download PDF Summary", key="generate_pdf_button"):
                       with st.spinner("Preparing PDF report..."):
                           # Pass current user profile and report metadata to the PDF generator instance
                           pdf_buffer = pdf_generator.generate_pdf( # <-- Calling on the instance
                                analysis_text=st.session_state.analysis_result,
                                user_profile=st.session_state.current_user,
                                report_metadata=current_report_metadata
                           )

                       if pdf_buffer:
                           # Create a more descriptive filename, sanitizing potentially problematic characters
                           user_name_slug = st.session_state.current_user.get('name', 'User').replace(' ', '_').replace('.', '').replace('/', '').replace('\\', '')
                           report_type_slug = current_report_metadata.get('report_type', 'Report').replace(' ', '_').replace('.', '').replace('/', '').replace('\\', '')
                           report_date_slug = current_report_metadata.get('upload_date', 'NODATE').split(' ')[0].replace('-', '').replace('.', '').replace('/', '').replace('\\', '')
                           download_filename = f"{user_name_slug}_{report_type_slug}_{report_date_slug}_Analysis.pdf"

                           st.download_button(
                               label="Download PDF",
                               data=pdf_buffer,
                               file_name=download_filename,
                               mime="application/pdf",
                               key="download_pdf_button"
                           )
                       else:
                            st.error("Could not generate PDF.")
             else:
                  st.warning("Analysis not available or PDF Generator not ready to generate PDF.")

             st.markdown("---") # Separator

         # --- Step 7: Q&A with the Report ---
         # Q&A is available if components are ready AND a report's text is loaded AND stored in DB (active_report_id)
         # And LLM Analyzer must be available and have the Q&A prompt
         if components_ready and st.session_state.extracted_text and st.session_state.active_report_id and llm_analyzer.qna_prompt_template:
              st.subheader(f"Ask a Question About Report: {report_display_name}")

              # Define a unique key for the text input using the active report ID
              question_input_key = f"qna_question_input_{st.session_state.active_report_id}"

              # The text input's value is controlled by st.session_state.current_qna_question_value
              user_question = st.text_input(
                   "Enter your question:",
                   key=question_input_key, # Still use a key to track the widget instance
                   value=st.session_state.current_qna_question_value # Value is read from this session state key
              )

              col_qna_buttons1, col_qna_buttons2 = st.columns(2)

              with col_qna_buttons1:
                  # Clear Q&A history and the input box value
                  if st.button("Clear Q&A History", key="clear_qna_button"):
                       st.session_state.qna_history = [] # Clear history
                       st.session_state.current_qna_question_value = "" # Clear the input value state
                       st.rerun() # Rerun to update the display


              with col_qna_buttons2:
                  # Ask AI button
                  if st.button("Ask AI about this report", key="ask_qna_button"):
                      if user_question:
                          # Add user question to history
                          st.session_state.qna_history.append({"role": "user", "content": user_question})

                          with st.spinner("Generating AI response..."):
                              # Call the Q&A method, passing the user ID and the active report ID to filter context
                              ai_answer = llm_analyzer.qna_with_report(
                                   user_id=st.session_state.current_user['id'],
                                   user_question=user_question,
                                   relevant_report_ids=st.session_state.active_report_id # Filter context to active report
                              )

                              # Add AI answer to history
                              st.session_state.qna_history.append({"role": "assistant", "content": ai_answer})

                          # Clear the input box by updating its value in session state
                          st.session_state.current_qna_question_value = ""
                          st.rerun() # Trigger rerun to display the updated history and clear input
                      else:
                           st.warning("Please enter a question to ask.")


              # Display Q&A history
              if st.session_state.qna_history:
                  st.subheader("Conversation History")
                  # Display history in chronological order
                  for chat in st.session_state.qna_history:
                      if chat["role"] == "user":
                          st.markdown(f"**You:** {chat['content']}")
                      else:
                          st.markdown(f"**AI Assistant:** {chat['content']}")
                  # Add a separator at the end of the conversation section
                  st.markdown("---")
              else:
                   st.info("Enter a question above to start the conversation about this report.")

         elif st.session_state.extracted_text and st.session_state.active_report_id:
             st.warning("AI components are not ready or Q&A feature is disabled. Cannot ask questions.")


    elif st.session_state.current_user:
         # Message shown if user is logged in but no report is active/processed/loaded
         st.info("Upload a report above or select one from your history to get AI analysis and ask questions.")


# --- Final Disclaimer (Always visible) ---
st.markdown("---")
st.markdown("""
**Important Notice:** This application is an AI-powered tool designed to assist in understanding medical reports by providing simplified interpretations based on extracted text. It is **NOT** a diagnostic tool and **does not** provide medical advice. The information provided should not be used as a substitute for professional medical judgement. Always consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.
""")