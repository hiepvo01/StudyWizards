import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile

# Load environment variables
load_dotenv()

# Define constants
CHROMA_DB_DIR = "./chroma_db"  # Directory to store Chroma database
PDF_DIR = "./pdf_documents"  # Directory to store PDF documents

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
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.6, max_tokens=1024)
    chain = load_qa_chain(llm, "stuff")
    similar_docs = get_similar_docs(query, collection_name)
    docs = [doc for doc in similar_docs]
    chain_response = chain.invoke(input={"input_documents": docs, "question": query})
    return chain_response["output_text"]

def load_docs(uploaded_file):
    """Loads a document from a Streamlit UploadedFile object"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    os.remove(tmp_file_path)  # Clean up the temporary file
    return documents

def split_docs(documents):
    """Splits a document into small chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def extract_content(uploaded_file):
    """Extracts content from the uploaded file"""
    documents = load_docs(uploaded_file)
    chunks = split_docs(documents)
    content = " ".join([chunk.page_content for chunk in chunks])
    return content

def file_exists_in_db(collection_name):
    """Checks if a file already exists in the Chroma database"""
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(collection_name=collection_name, persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    results = vector_db.similarity_search("dummy query", k=1)
    return len(results) > 0

def insert_data(file_path):
    """Inserts data into the vector db"""
    collection_name = f"collection_{os.path.basename(file_path).replace('.pdf', '').replace('.txt', '').replace('.json', '')}"
    documents = load_docs(file_path)
    documents = split_docs(documents)
    vector_db = get_db_connection(collection_name)
    vector_db.add_documents(documents)
    vector_db.persist()

def process_existing_pdfs():
    """Processes existing PDF files in the directory into embeddings"""
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            collection_name = f"collection_{pdf_file.replace('.pdf', '')}"
            if not file_exists_in_db(collection_name):
                file_path = os.path.join(PDF_DIR, pdf_file)
                insert_data(file_path)

def delete_file_and_embeddings(selected_file):
    """Deletes the selected PDF file and its embeddings"""
    collection_name = f"collection_{selected_file.replace('.pdf', '').replace('.txt', '').replace('.json', '')}"
    vector_db = get_db_connection(collection_name)
    vector_db.delete_collection()
    
    file_path = os.path.join(PDF_DIR, selected_file)
    if os.path.exists(file_path):
        os.remove(file_path)

def get_context_retriever_chain(vector_store, selected_file):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", f"Based on the above conversation and the selected PDF file '{selected_file}', look up relevant information strictly from the selected document"),
        ("user", "{input}")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain, selected_file):
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Answer the user's questions based on the context from the selected PDF file '{selected_file}':\n\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, collection_name, selected_file):
    vector_store = get_db_connection(collection_name)
    retriever_chain = get_context_retriever_chain(vector_store, selected_file)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain, selected_file)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

def main():
    st.set_page_config(page_title="StudyWizards Chatbot", page_icon="🤖")
    st.title("StudyWizards Chatbot")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "previous_selected_file" not in st.session_state:
        st.session_state.previous_selected_file = None

    # Process existing PDF files in the folder into embeddings
    process_existing_pdfs()

    with st.sidebar:
        st.header("Add Textbook PDF for Questioning")
        uploaded_pdf_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_pdf_file is not None:
            file_path = os.path.join(PDF_DIR, uploaded_pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf_file.getbuffer())
            with st.spinner('Processing new PDF file...'):
                insert_data(file_path)
                st.success("New PDF file processed and inserted into vector database successfully")

    uploaded_files = os.listdir(PDF_DIR)
    selected_file = st.selectbox("Select a PDF file", uploaded_files)

    if selected_file:
        collection_name = f"collection_{selected_file.replace('.pdf', '').replace('.txt', '').replace('.json', '')}"
        
        # Reset conversation if a new file is selected
        if st.session_state.previous_selected_file != selected_file:
            st.session_state.chat_history = []
            st.session_state.previous_selected_file = selected_file

        initial_query = st.text_input("Enter your question:")
        uploaded_context_file = st.file_uploader("Attach a context file", type=["pdf", "txt", "json"], key="context_file")

        if st.button("Ask Question"):
            additional_context = ""
            if uploaded_context_file is not None:
                additional_context = extract_content(uploaded_context_file)
            
            if initial_query:
                context = " ".join(st.session_state.chat_history) + " " + initial_query + " .Here is the content of the extra attached file: " + additional_context
                with st.spinner('Fetching answer...'):
                    answer = get_response(context, collection_name, selected_file)
                st.session_state.chat_history.append(f"**Question:** {initial_query}")
                st.session_state.chat_history.append(f"**Answer:** {answer}")
                st.markdown(f"**Question:** {initial_query}")
                st.markdown(f"**Answer:** {answer}")
            else:
                st.warning("Please enter a question.")
    
        if st.button("Reset"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("Delete Selected File"):
            delete_file_and_embeddings(selected_file)
            st.success(f"{selected_file} and its embeddings have been deleted.")
            st.rerun()

    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.markdown("**Conversation History:**")
        for message in st.session_state.chat_history:
            st.markdown(message)

if __name__ == "__main__":
    main()
