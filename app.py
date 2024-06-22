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
CHROMA_DB_DIR = "./chroma_db"

# Cache for already checked files
checked_files = set()

def get_db_connection(collection_name):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return Chroma(collection_name=collection_name, persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def get_similar_docs(query, collection_name):
    return get_db_connection(collection_name).similarity_search(query, k=10)

def fetch_answer_from_llm(query, collection_name):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8, max_tokens=1524)
    chain = load_qa_chain(llm, "stuff")
    docs = get_similar_docs(query, collection_name)
    return chain.invoke(input={"input_documents": docs, "question": query})["output_text"]

def load_and_split_docs(file_path):
    documents = PyPDFLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def file_exists_in_db(collection_name):
    if collection_name in checked_files:
        return True
    vector_db = get_db_connection(collection_name)
    if vector_db.similarity_search("dummy query", k=1):
        checked_files.add(collection_name)
        return True
    return False

def insert_data(file_path):
    collection_name = f"collection_{os.path.basename(file_path).replace('.pdf', '')}"
    if file_exists_in_db(collection_name):
        st.warning(f"File '{file_path}' already exists in the database.", icon="⚠️")
        return
    
    try:
        docs = load_and_split_docs(file_path)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        for doc in docs:
            doc.metadata["collection"] = collection_name
        vectorstore = Chroma.from_documents(collection_name=collection_name, documents=docs, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
        vectorstore.persist()
        st.success(f"File '{file_path}' inserted into vector database successfully")
    except Exception as e:
        st.error(str(e))

def process_existing_pdfs():
    pdf_folder = "./pdf_documents"
    os.makedirs(pdf_folder, exist_ok=True)
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            insert_data(os.path.join(pdf_folder, file_name))

def main():
    st.title("PDF Question Answering System")

    if 'pdfs_processed' not in st.session_state:
        st.session_state.pdfs_processed = False

    if not st.session_state.pdfs_processed:
        with st.spinner('Processing existing PDF files...'):
            process_existing_pdfs()
        st.session_state.pdfs_processed = True

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
        new_uploaded_file = None

    uploaded_files = os.listdir("./pdf_documents")
    selected_file = st.selectbox("Select a PDF file", uploaded_files)

    if selected_file:
        collection_name = f"collection_{selected_file.replace('.pdf', '')}"

        if "previous_selected_file" not in st.session_state:
            st.session_state.previous_selected_file = None

        if st.session_state.previous_selected_file != selected_file:
            st.session_state.conversation_history = ""
            st.session_state.previous_selected_file = selected_file

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = ""

        initial_query = st.text_input("Enter your question:")

        if st.button("Ask Question"):
            if initial_query:
                context = f"Context: {st.session_state.conversation_history} {initial_query} Please find correct and detailed information to answer the question from the provided context"
                with st.spinner('Fetching answer...'):
                    answer = fetch_answer_from_llm(context, collection_name)
                st.session_state.conversation_history += f" {initial_query} {answer}"
                st.markdown(f"**Question:** {initial_query}")
                st.markdown(f"**Answer:** {answer}")
            else:
                st.warning("Please enter a question.")
        
        if st.button("Reset"):
            st.session_state.conversation_history = ""
            st.rerun()

if __name__ == "__main__":
    main()
