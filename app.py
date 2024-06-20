import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Define constants
CHROMA_DB_DIR = "./chroma_db"  # Directory to store Chroma database

def get_db_connection(collection_name):
    """Returns a Chroma DB connection object"""
    embeddings = OpenAIEmbeddings()
    return Chroma(collection_name=collection_name, persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def get_similar_docs(query: str, collection_name: str):
    """Fetches similar text from the vector db"""
    vector_db = get_db_connection(collection_name)
    return vector_db.similarity_search(query, k=3)

def fetch_answer_from_llm(query: str, collection_name: str):
    """Fetches relevant answer from LLM"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0.6,
                     max_tokens=1024)
    chain = load_qa_chain(llm, "stuff")
    similar_docs = get_similar_docs(query, collection_name)
    docs = [doc for doc in similar_docs]
    chain_response = chain.invoke(input={"input_documents": docs, "question": query})
    return chain_response["output_text"]

def load_docs(file_path):
    """Loads a PDF document"""
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_docs(documents):
    """Splits a document into small chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
    return text_splitter.split_documents(documents)

def file_exists_in_db(collection_name):
    """Checks if a file already exists in the Chroma database"""
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(collection_name=collection_name, persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    results = vector_db.similarity_search("dummy query", k=1)
    return len(results) > 0

def insert_data(file_path):
    """Creates embeddings for document chunks and inserts them into Chroma DB"""
    collection_name = f"collection_{os.path.basename(file_path).replace('.pdf', '')}"
    if file_exists_in_db(collection_name):
        print(f"File '{file_path}' already exists in the database.")
        return
    
    try:
        documents = load_docs(file_path)
        docs = split_docs(documents)
        embeddings = OpenAIEmbeddings()
        for doc in docs:
            doc.metadata["collection"] = collection_name
        vectorstore = Chroma.from_documents(collection_name=collection_name, documents=docs, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
        vectorstore.persist()
        print(f"File '{file_path}' inserted into vector database successfully")
    except Exception as exception_message:
        print(str(exception_message))

def process_existing_pdfs():
    """Processes all existing PDFs in the folder into embeddings"""
    pdf_folder = "./pdf_documents"
    os.makedirs(pdf_folder, exist_ok=True)
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            insert_data(file_path)

def main():
    st.title("PDF Question Answering System")

    # Process existing PDFs when the app starts
    with st.spinner('Processing existing PDF files...'):
        process_existing_pdfs()

    st.markdown("### Upload New PDF File")
    new_uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if new_uploaded_file:
        file_path = f"./pdf_documents/{new_uploaded_file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(new_uploaded_file.getbuffer())
        with st.spinner('Processing new PDF file...'):
            insert_data(file_path)
        st.success("New file processed and inserted into vector database successfully")
        st.experimental_rerun()

    uploaded_files = os.listdir("./pdf_documents")
    selected_file = st.selectbox("Select a PDF file", uploaded_files)

    if selected_file:
        collection_name = f"collection_{selected_file.replace('.pdf', '')}"
        
        if "previous_selected_file" not in st.session_state:
            st.session_state.previous_selected_file = None
        
        # Reset conversation if a new file is selected
        if st.session_state.previous_selected_file != selected_file:
            st.session_state.conversation_history = ""
            st.session_state.previous_selected_file = selected_file

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = ""

        initial_query = st.text_input("Enter your question:")
        
        if st.button("Ask Question"):
            if initial_query:
                context = st.session_state.conversation_history + " " + initial_query
                with st.spinner('Fetching answer...'):
                    answer = fetch_answer_from_llm(context, collection_name)
                st.session_state.conversation_history += f" {initial_query} {answer}"
                st.markdown(f"**Question:** {initial_query}")
                st.markdown(f"**Answer:** {answer}")
            else:
                st.warning("Please enter a question.")
    
        if st.button("Reset"):
            st.session_state.conversation_history = ""
            st.experimental_rerun()

    if "conversation_history" in st.session_state and st.session_state.conversation_history:
        st.markdown("**Conversation History:**")
        st.markdown(st.session_state.conversation_history)

if __name__ == "__main__":
    main()
