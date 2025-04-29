from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document # Used for formatting context
import streamlit as st
import os # Needed to check prompt file existence

# Define prompt file paths
ANALYSIS_PROMPT_PATH = "prompts/analysis_prompt.txt"
QNA_PROMPT_PATH = "prompts/qna_prompt.txt"
COMPARISON_PROMPT_PATH = "prompts/comparison_prompt.txt"

class LLMAnalyzer:
    def __init__(self, openai_api_key, vector_db_instance):
        if not openai_api_key:
             raise ValueError("OpenAI API key is not provided.")
        if not vector_db_instance or not vector_db_instance.is_available():
             st.warning("Vector database not provided or not available. Q&A and context retrieval features may not work.")
        self.vector_db = vector_db_instance # Keep reference to the VectorDB

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key) # Using gpt-4o

        # Load prompt templates from files
        self.analysis_prompt_template = self._load_prompt_template(ANALYSIS_PROMPT_PATH, "analysis")
        self.qna_prompt_template = self._load_prompt_template(QNA_PROMPT_PATH, "Q&A")
        self.comparison_prompt_template = self._load_prompt_template(COMPARISON_PROMPT_PATH, "comparison") # Load comparison prompt

        # Define analysis chain (basic chain without retrieval yet)
        if self.analysis_prompt_template:
             self.analysis_prompt = PromptTemplate(
                 input_variables=["report_text"],
                 template=self.analysis_prompt_template,
             )
             self.analysis_chain = self.analysis_prompt | self.llm | StrOutputParser()
        else:
             self.analysis_chain = None
             st.error("Analysis prompt template not loaded. Analysis will not work.")

        # Q&A chain setup
        # The Q&A chain will need context retrieved from the vector DB
        # We will build this chain dynamically in the qna_with_report method or define a retrieval chain pattern
        # Let's define a basic Q&A prompt template first
        if self.qna_prompt_template:
             self.qna_prompt = PromptTemplate(
                 input_variables=["context", "question"],
                 template=self.qna_prompt_template,
             )
             # Retrieval-augmented generation (RAG) chain structure
             # This is a simplified representation. Actual retrieval happens before calling this part.
             # chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | output_parser)
             # We'll implement the retrieval part in the qna_with_report method.
        else:
             st.error("Q&A prompt template not loaded. Q&A will not work.")

        # Comparison chain setup (Similar to Q&A, needs two sets of text)
        if self.comparison_prompt_template:
             self.comparison_prompt = PromptTemplate(
                 input_variables=["report1_text", "report2_text", "comparison_query"], # Or structure differently
                 template=self.comparison_prompt_template,
             )
             # Comparison chain could be prompt | llm | output_parser
             self.comparison_chain = self.comparison_prompt | self.llm | StrOutputParser()
        else:
            st.warning("Comparison prompt template not loaded. Comparison analysis feature may be limited.")


    def _load_prompt_template(self, file_path, prompt_name):
         """Helper to load prompt templates from files."""
         try:
             with open(file_path, "r", encoding='utf-8') as f:
                 st.info(f"Loaded {prompt_name} prompt template from {file_path}.")
                 return f.read()
         except FileNotFoundError:
             st.error(f"Prompt template file not found: {file_path}! {prompt_name} feature disabled.")
             return None
         except Exception as e:
             st.error(f"Error loading prompt template {file_path}: {e}")
             return None


    def analyze_report(self, report_text):
        """Runs the initial analysis chain on the report text."""
        if not self.analysis_chain:
             return "Analysis feature is not available due to missing prompt template."

        if not report_text or not report_text.strip():
            st.warning("No text provided for analysis.")
            return "Could not analyze the report text."

        st.info("Sending report text to LLM for initial analysis...")
        try:
            # Invoke the chain
            analysis_result = self.analysis_chain.invoke({"report_text": report_text})
            st.success("Initial analysis complete.")
            return analysis_result
        except Exception as e:
            st.error(f"Error during LLM analysis: {e}")
            return f"An error occurred during initial analysis: {e}"

    def qna_with_report(self, user_id, user_question, relevant_report_ids=None):
         """
         Answers user questions using context retrieved from the vector DB
         for the specific user and optionally specific reports.
         """
         if not self.vector_db or not self.vector_db.is_available():
             return "Q&A feature is not available because the vector database is not set up."
         if not self.qna_prompt_template:
              return "Q&A feature is not available due to missing prompt template."
         if not user_id:
              return "User must be logged in to ask questions."
         if not user_question or not user_question.strip():
             return "Please enter a question."

         st.info(f"Searching vector DB for relevant context for user {user_id}...")

         # Retrieve relevant text chunks based on the user's question and user ID,
         # optionally filtered by report_ids for multi-report Q&A or comparisons
         retrieved_docs = self.vector_db.query_user_data(
             user_id=user_id,
             query_text=user_question,
             report_ids=relevant_report_ids, # Pass the filter
             n_results=10 # Retrieve top N chunks
        )

         if not retrieved_docs:
             st.warning("Could not find relevant information in your reports to answer the question.")
             # Fallback: Ask the LLM based on general knowledge? Or just say cannot answer?
             # Let's say cannot answer from the reports.
             return "Based on the available reports, I cannot find information to answer this question. Please consult a medical professional."

         # Format the retrieved documents into a single context string for the LLM
         # Include metadata like filename/page number to help the LLM reference source (optional but good)
         context_text = "\n---\n".join([
             f"Report: {doc['metadata'].get('filename', 'N/A')} (Page {doc['metadata'].get('chunk_index', -1) + 1}) - Chunk ID: {doc['id']}\n{doc['document']}"
             for doc in retrieved_docs
         ])

         st.info("Sending question and context to LLM for answer generation...")

         try:
             # Create prompt with retrieved context and user question
             qna_prompt_formatted = self.qna_prompt.format(context=context_text, question=user_question)

             # Invoke the LLM
             qna_answer = self.llm.invoke(qna_prompt_formatted)
             st.success("Q&A complete.")
             return qna_answer.content # Return the content string

         except Exception as e:
             st.error(f"Error during LLM Q&A: {e}")
             return f"An error occurred while trying to answer your question: {e}"

    # Placeholder for a dedicated comparison function if needed,
    # but Q&A function with relevant_report_ids can handle comparisons too.
    # def compare_reports_qna(self, user_id, report_id_1, report_id_2, comparison_query):
    #     """Retrieves text from two reports and performs a comparison analysis."""
    #     if not self.vector_db or not self.vector_db.is_available():
    #          return "Comparison feature is not available because the vector database is not set up."
    #     if not self.comparison_chain:
    #          return "Comparison feature is not available due to missing prompt template."
    #     if not user_id or not report_id_1 or not report_id_2 or not comparison_query:
    #          return "Insufficient information for comparison."

    #     # Retrieve full text for both reports (using the new method in VectorDB)
    #     report1_text = self.vector_db.get_report_text_by_id(user_id, report_id_1)
    #     report2_text = self.vector_db.get_report_text_by_id(user_id, report_id_2)

    #     if not report1_text or not report2_text:
    #          return "Could not retrieve text for one or both reports for comparison."

    #     st.info(f"Sending reports {report_id_1} and {report_id_2} for comparison analysis...")
    #     try:
    #         comparison_result = self.comparison_chain.invoke({
    #             "report1_text": report1_text,
    #             "report2_text": report2_text,
    #             "comparison_query": comparison_query # Or just use the user's question
    #         })
    #         st.success("Comparison analysis complete.")
    #         return comparison_result
    #     except Exception as e:
    #          st.error(f"Error during comparison analysis: {e}")
    #          return f"An error occurred during comparison analysis: {e}"