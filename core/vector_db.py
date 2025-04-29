import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import os
import shutil
import traceback # Import traceback to print full stack trace

# Define the directory for persistent storage
CHROMA_DB_PATH = "./chroma_db_data"

class VectorDB:
    def __init__(self, openai_api_key):
        # Ensure the API key is provided before proceeding with client/embedding function init
        if not openai_api_key:
             st.error("OpenAI API key is required to initialize VectorDB.")
             self.client = None # Indicate failure
             self.collection = None
             self.embedding_function = None
             self.text_splitter = None
             return # Exit initialization early

        self.client = None
        self.collection = None
        self.embedding_function = None
        self.text_splitter = None # Initialize attributes to None

        # Use a persistent client to store data across sessions
        try:
            # Ensure the directory exists
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            st.info(f"Using persistent ChromaDB at {CHROMA_DB_PATH}")
        except Exception as e:
            st.error(f"Error initializing persistent ChromaDB at {CHROMA_DB_PATH}: {e}")
            st.info("Falling back to in-memory client. Data will NOT persist across sessions!")
            self.client = chromadb.Client() # Fallback to in-memory


        # OpenAI Embedding Function for ChromaDB
        try:
             # Check if the API key is actually valid for the embedding call
             # A small test call can confirm this, but is slow during init
             # Relying on the first embedding call in add_document to catch issues might be ok
             self.embedding_function = OpenAIEmbeddingFunction(
                 api_key=openai_api_key,
                 model_name="text-embedding-ada-002" # Standard embedding model
             )
        except Exception as e:
             st.error(f"Error initializing OpenAI Embedding Function: {e}")
             self.client = None # Cannot function without embedding function
             self.collection = None
             self.embedding_function = None
             st.error("VectorDB initialization failed due to embedding function error.")
             return # Exit initialization early


        self.collection_name = "medical_reports_collection"

        try:
            
             self.collection = self.client.get_or_create_collection(
                 name=self.collection_name,
                 embedding_function=self.embedding_function # Assign embedding function here
            )
             st.info(f"Loaded or created ChromaDB collection: {self.collection_name}")
        except Exception as e:
             st.error(f"Error getting or creating ChromaDB collection '{self.collection_name}': {e}")
             self.collection = None # Indicate collection is not available
             # Client might still be valid, but collection failed

        # Text splitter for creating document chunks - always create if we got this far
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


    def is_available(self):
        """Check if the ChromaDB collection was successfully initialized."""
        # Ensure client, collection, and embedding function are all valid
        return self.client is not None and self.collection is not None and self.embedding_function is not None

    def add_document(self, user_id, report_id, report_metadata, text):
        """Adds document text chunks with user and report metadata to the vector store."""
        if not self.is_available():
            st.error("ChromaDB is not available. Cannot add document.")
            return []
        if not text or not text.strip():
            st.warning("No text to add to vector database.")
            return []
        if not user_id:
             st.error("User ID is required to add document.")
             return []
        if not report_id:
             st.error("Report ID is required to add document.")
             return []

        st.info("Starting add_document process...")
        st.info(f"Report ID: {report_id}, User ID: {user_id}")


        try:
            st.info("Splitting text into chunks...")
            # Create Langchain Document objects
            docs = self.text_splitter.create_documents([text])
            st.info(f"Created {len(docs)} chunks.")


            # Prepare data for ChromaDB
            ids = [f"{user_id}_{report_id}_chunk_{i}" for i in range(len(docs))]
            documents = [doc.page_content for doc in docs]
            metadatas_list = [{**report_metadata, "user_id": user_id, "report_id": report_id, "chunk_index": i} for i, doc in enumerate(docs)]

            st.info(f"Prepared {len(ids)} IDs, documents, and metadatas for adding.")
            # st.info(f"Sample Metadata: {metadatas_list[0] if metadatas_list else 'N/A'}") # Optional, can be verbose

            # Check for potential duplicate IDs before adding (important for persistence)
            try:
                 st.info("Checking for existing IDs...")
                 # Include=[] makes this call faster as it doesn't retrieve documents or embeddings
                 existing_items = self.collection.get(ids=ids, include=[])
                 st.info(f"Finished checking for existing IDs. Found {len(existing_items.get('ids', []))} existing.")

                 # Check if any returned ID is actually in the requested list of IDs
                 found_ids = set(existing_items.get('ids', []))
                 requested_ids = set(ids)
                 overlapping_ids = list(found_ids.intersection(requested_ids))

                 if overlapping_ids:
                      st.warning(f"Attempted to add documents with existing IDs: {overlapping_ids}. Skipping add operation for these IDs.")
                      # If ALL requested IDs already exist, return empty list
                      if len(overlapping_ids) == len(ids):
                          return []
                      else:
                          # If only some exist, filter out the ones that exist and add the rest
                          st.info("Adding only the non-existing chunks.")
                          new_ids = [id for id in ids if id not in found_ids]
                          new_documents = [doc for doc, id in zip(documents, ids) if id in new_ids]
                          new_metadatas = [meta for meta, id in zip(metadatas_list, ids) if id in new_ids]
                          ids = new_ids
                          documents = new_documents
                          metadatas_list = new_metadatas
                          if not ids: return [] # Nothing left to add

            except Exception as e:
                 st.warning(f"Error checking for existing IDs in ChromaDB: {e}. Proceeding with add, might cause errors if IDs exist.")
                 # Continue anyway, add() might raise an error if IDs are duplicates.


            st.info(f"Attempting to add {len(ids)} chunks to ChromaDB...")
            # Add documents to the collection
            # This call triggers embedding generation by OpenAIEmbeddingFunction internally
            self.collection.add(
                documents=documents,
                metadatas=metadatas_list,
                ids=ids
            )
            st.success(f"Added {len(ids)} text chunks for report {report_id} to vector database.") # Log number of added IDs
            return ids
        except Exception as e:
            # --- DETAILED ERROR LOGGING FOR ADD FAILURE ---
            st.error(f"!!! FATAL ERROR during add_document process for report {report_id}: {e}")
            print(f"!!! FATAL ERROR during add_document process for report {report_id}: {e}")
            traceback.print_exc() # Print the full traceback to console
            # --- END OF DETAILED ERROR LOGGING ---

            # Attempt to clean up if partial add failed
            try:
                 st.warning("Attempting cleanup after add failure...")
                 # Use the corrected delete filter syntax here with explicit $and operator
                 # This should now be correct: {"$and": [{"report_id": report_id}, {"user_id": user_id}]}
                 self.collection.delete(where={"$and": [{"report_id": report_id}, {"user_id": user_id}]})
                 st.info(f"Attempted to delete potentially partial data for report {report_id} due to error.")
            except Exception as delete_e:
                 st.warning(f"Failed to clean up data for report {report_id} after add error: {delete_e}")
            return []


    def query_user_data(self, user_id, query_text, report_ids=None, n_results=10):
        """
        Queries the vector store for text relevant to a specific user, optionally filtered by report ID(s).
        report_ids can be a single ID string or a list of ID strings.
        """
        st.info("Starting query_user_data process...") # --- ADDED LOG ---
        st.info(f"User ID: {user_id}, Query: '{query_text[:50]}...'") # --- ADDED LOG ---
        st.info(f"Filtering by Report ID(s): {report_ids}") # --- ADDED LOG ---


        if not self.is_available():
            st.error("ChromaDB is not available. Cannot query.")
            return []
        if not user_id:
             st.error("User ID is required for query.")
             return []
        # Check if collection is empty
        if self.collection.count() == 0:
            st.warning("Vector database is empty. Cannot query.")
            return []

        # Build the query filter for the `where` clause using explicit operators
        # The filter must always include the user_id condition
        base_user_filter = {"user_id": user_id}

        # Initialize the final filter. Start with the user filter.
        final_filter = base_user_filter

        # Add report ID filter(s) if specified
        if report_ids:
            report_filter_part = None # This will hold the filter part for report_ids

            if isinstance(report_ids, str):
                # Filter for a single report ID using equality
                report_filter_part = {"report_id": report_ids}
            elif isinstance(report_ids, list) and report_ids: # Check if list is not empty
                 if len(report_ids) == 1:
                      # Filter for a single report ID using equality
                      report_filter_part = {"report_id": report_ids[0]}
                 else:
                      # Filter for multiple report IDs using $in operator
                      report_filter_part = {"report_id": {"$in": report_ids}}

            # If we have a report filter part, combine it with the user filter using $and
            if report_filter_part:
                 final_filter = {"$and": [base_user_filter, report_filter_part]}

        st.info(f"ChromaDB query filter: {final_filter}") # Show the final structured filter


        try:
            st.spinner(f"Querying ChromaDB for user {user_id}...")
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=final_filter, # Use the correctly structured filter here
                include=['documents', 'metadatas', 'distances'] # Include metadata to see report_id, etc.
            )
            st.info(f"Raw ChromaDB query results (first 5 IDs): {results.get('ids', [[]])[0][:5] if results and results.get('ids') and results.get('ids')[0] else 'N/A'}") # Log first 5 IDs if available
            st.info(f"Raw ChromaDB query results count (first list): {len(results.get('ids', [[]])[0]) if results and results.get('ids') and results.get('ids')[0] else 0}") # Log number of results


            # Reformat results for easier use (list of dicts)
            formatted_results = []
            # ChromaDB query results are nested lists, flatten them
            if results and results.get('ids') and len(results['ids']) > 0:
                # Check if the first list of results is not empty
                if results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        # Check if the index is valid across all included lists
                        if i < len(results.get('documents', [[]])[0]) and \
                           i < len(results.get('metadatas', [[]])[0]) and \
                           i < len(results.get('distances', [[]])[0]):
                            formatted_results.append({
                                "id": results['ids'][0][i],
                                "document": results['documents'][0][i],
                                "metadata": results['metadatas'][0][i],
                                "distance": results['distances'][0][i]
                            })

            st.success(f"ChromaDB query complete. Found {len(formatted_results)} relevant chunks.")
            return formatted_results
        except Exception as e:
            st.error(f"Error querying ChromaDB: {e}")
            print(f"Error querying ChromaDB: {e}") # Also print to console
            traceback.print_exc()
            return []

    def delete_user_data(self, user_id):
        """Deletes all data associated with a specific user ID from ChromaDB."""
        st.info(f"Starting delete_user_data process for user {user_id}...")
        if not self.is_available():
            st.warning("ChromaDB is not available. Cannot delete data.")
            return False
        if not user_id:
             st.error("User ID is required for deletion.")
             return False
        # Check if collection is empty before attempting delete
        if self.collection.count() == 0:
             st.info(f"ChromaDB collection is empty. No data to delete for user {user_id}.")
             return True # Consider it successfully deleted if nothing was there

        try:
            st.spinner(f"Deleting ChromaDB data for user {user_id}...")
            # Delete by filtering on user_id metadata - this should work with `delete`
            # {"user_id": user_id} is a single condition filter, should be OK for delete
            self.collection.delete(where={"user_id": user_id})
            st.success(f"Deleted data associated with user {user_id} from ChromaDB.")
            return True
        except Exception as e:
            st.error(f"Error deleting data for user {user_id} from ChromaDB: {e}")
            print(f"Error deleting data for user {user_id} from ChromaDB: {e}") # Also print to console
            traceback.print_exc()
            return False

    def delete_chroma_storage(self):
        """Deletes the entire persistent ChromaDB directory."""
        st.warning("!!! Starting delete_chroma_storage process (DELETES ALL DATA) !!!")
        if self.client is None:
            st.warning("ChromaDB client was not initialized. No persistent storage to delete.")
            return False

        st.warning(f"Attempting to delete the entire ChromaDB directory: {CHROMA_DB_PATH}")
        if os.path.exists(CHROMA_DB_PATH):
            try:
                shutil.rmtree(CHROMA_DB_PATH)
                st.success(f"Deleted ChromaDB directory: {CHROMA_DB_PATH}. Restart the app to clear memory.")
                return True
            except Exception as e:
                st.error(f"Error deleting ChromaDB directory {CHROMA_DB_PATH}: {e}. Please stop the app and delete the folder manually.")
                print(f"Error deleting ChromaDB directory {CHROMA_DB_PATH}: {e}") # Also print to console
                traceback.print_exc()
                return False
        else:
            st.info(f"ChromaDB directory not found: {CHROMA_DB_PATH}.")
            return False


    def get_user_reports_metadata(self, user_id):
        """Retrieves metadata for all reports belonging to a user."""
        st.info(f"Starting get_user_reports_metadata process for user {user_id}...")
        if not self.is_available():
            st.warning("ChromaDB is not available. Cannot retrieve report history.")
            return {}
        if not user_id:
             st.error("User ID is required to retrieve history.")
             return {}
        # Check if collection is empty
        if self.collection.count() == 0:
             st.info("ChromaDB collection is empty. No report history found.")
             return {}

        st.spinner(f"Fetching report history for user {user_id} from ChromaDB...")
        try:
            # Retrieve *all* items for the user using the filter.
            st.info(f"Attempting collection.get with filter: {{\"user_id\": \"{user_id}\"}}")
            # A single condition filter like {"user_id": user_id} is expected to work for collection.get
            results = self.collection.get(where={"user_id": user_id}, include=['metadatas'])

            st.info(f"Raw results from collection.get (first 5 IDs): {results.get('ids', [])[:5]}")
            st.info(f"Raw results count from collection.get: {len(results.get('ids', []))}")


            reports_metadata = {}
            if results and results.get('metadatas'):
                # Iterate through the metadata of all chunks belonging to the user
                # Using zip to align ids and metadatas correctly
                ids_list = results.get('ids', [])
                metas_list = results.get('metadatas', [])
                if ids_list and len(ids_list) > 0 and metas_list and len(metas_list) > 0 and len(ids_list) == len(metas_list):
                    for i in range(len(ids_list)):
                        metadata = metas_list[i]
                        report_id = metadata.get('report_id')
                        # Add report metadata if it's a new report ID encountered.
                        if report_id and report_id not in reports_metadata:
                            reports_metadata[report_id] = {
                                "report_id": report_id,
                                "filename": metadata.get('filename', 'N/A'),
                                "file_type": metadata.get('file_type', 'N/A'),
                                "report_type": metadata.get('report_type', 'Unknown'),
                                "upload_date": metadata.get('upload_date', 'N/A')
                                # Add any other relevant report-level metadata stored with chunks here
                            }
                elif results.get('ids') is not None and len(results.get('ids', [])) > 0:
                     st.warning(f"ChromaDB returned {len(results['ids'])} IDs but metadata list is inconsistent or empty. Cannot retrieve full history.")


            st.success(f"Retrieved metadata for {len(reports_metadata)} reports for user {user_id}.")
            return reports_metadata
        except Exception as e:
            st.error(f"Error retrieving report history for user {user_id}: {e}")
            print(f"Error retrieving report history for user {user_id}: {e}") # Also print to console
            traceback.print_exc()
            return {}

    def get_report_text_by_id(self, user_id, report_id):
        """Retrieves the full text of a specific report for a user by assembling chunks."""
        st.info(f"Starting get_report_text_by_id process for report {report_id} (user {user_id})...")
        if not self.is_available():
            st.warning("ChromaDB is not available. Cannot retrieve report text.")
            return ""
        if not user_id or not report_id:
             st.error("User ID and Report ID are required to retrieve report text.")
             return ""
         # Check if collection is empty
        if self.collection.count() == 0:
             st.warning("ChromaDB collection is empty. Cannot retrieve report text.")
             return ""


        st.spinner(f"Retrieving text for report {report_id} for user {user_id}...")
        try:
            
            st.info(f"Attempting collection.get with filter: {{\"user_id\": \"{user_id}\"}}")
            all_user_chunks = self.collection.get(
                where={"user_id": user_id}, # <-- Only filtering by user_id here in the ChromaDB get call
                include=['documents', 'metadatas']
            )

            st.info(f"Raw results from collection.get (first 5 IDs): {all_user_chunks.get('ids', [])[:5]}")
            st.info(f"Raw results count from collection.get: {len(all_user_chunks.get('ids', []))}")


            if not all_user_chunks or not all_user_chunks.get('documents'):
                st.warning(f"No data found for user {user_id} in the database.")
                return ""

            # Filter the retrieved chunks by the specific report_id in Python
            report_chunks = []
            # zip results together for easier processing (ids, documents, metadatas lists are aligned)
            ids_list = all_user_chunks.get('ids', [])
            docs_list = all_user_chunks.get('documents', [])
            metas_list = all_user_chunks.get('metadatas', [])

            if ids_list and len(ids_list) > 0 and docs_list and len(docs_list) == len(ids_list) and metas_list and len(metas_list) == len(ids_list):
                 st.info(f"Filtering {len(ids_list)} chunks by report_id '{report_id}' in Python...")
                 for i in range(len(ids_list)):
                      doc = docs_list[i]
                      meta = metas_list[i]
                      # Check if the chunk belongs to the desired report_id
                      if meta and meta.get('report_id') == report_id:
                           report_chunks.append({'document': doc, 'metadata': meta})
            elif ids_list and len(ids_list) > 0:
                 st.warning(f"ChromaDB returned {len(ids_list)} IDs but documents or metadata lists are inconsistent. Cannot retrieve report text.")


            st.info(f"Found {len(report_chunks)} chunks matching report_id '{report_id}' after Python filtering.")

            if not report_chunks:
                 st.warning(f"No data found for report {report_id} within user {user_id}'s data.")
                 return ""

            
            st.info("Sorting chunks by index...")
            sorted_chunks = sorted(report_chunks, key=lambda x: x['metadata'].get('chunk_index', 0))
            st.info("Finished sorting chunks.")


            # Join document content from the sorted chunks
            full_text = "".join([chunk['document'] for chunk in sorted_chunks])

            st.success(f"Successfully retrieved and assembled text for report {report_id}.")
            return full_text

        except Exception as e:
            st.error(f"Error retrieving text for report {report_id}: {e}")
            print(f"Error retrieving text for report {report_id}: {e}") # Also print to console
            traceback.print_exc()
            return ""