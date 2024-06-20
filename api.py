from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)

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

def delete_data(collection_name):
    """Deletes embeddings for a document from Chroma DB and removes the file"""
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(collection_name=collection_name, persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    vector_db.delete_collection()

def process_existing_pdfs():
    """Processes all existing PDFs in the folder into embeddings"""
    pdf_folder = "./pdf_documents"
    os.makedirs(pdf_folder, exist_ok=True)
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            insert_data(file_path)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Uploads a new PDF file and processes it into embeddings"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        file_path = f"./pdf_documents/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        insert_data(file_path)
        return jsonify({"status": "File processed and inserted into vector database successfully"}), 200
    else:
        return jsonify({"error": "Invalid file type, only PDFs are allowed"}), 400

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Fetches an answer to a question based on the selected PDF"""
    data = request.get_json()
    if not data or 'question' not in data or 'file_name' not in data:
        return jsonify({"error": "Invalid request"}), 400
    
    question = data['question']
    file_name = data['file_name']
    collection_name = f"collection_{file_name.replace('.pdf', '')}"
    
    if not file_exists_in_db(collection_name):
        return jsonify({"error": "File not found in the database"}), 404

    answer = fetch_answer_from_llm(question, collection_name)
    return jsonify({"question": question, "answer": answer}), 200

@app.route('/delete_file', methods=['DELETE'])
def delete_file():
    """Deletes a file and its embeddings from the database"""
    data = request.get_json()
    if not data or 'file_name' not in data:
        return jsonify({"error": "Invalid request"}), 400

    file_name = data['file_name']
    collection_name = f"collection_{file_name.replace('.pdf', '')}"
    file_path = f"./pdf_documents/{file_name}"
    
    if not file_exists_in_db(collection_name):
        return jsonify({"error": "File not found in the database"}), 404

    try:
        delete_data(collection_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"status": f"File '{file_name}' and its embeddings deleted successfully"}), 200
    except Exception as exception_message:
        return jsonify({"error": str(exception_message)}), 500

if __name__ == '__main__':
    # Automatically process existing PDFs when the server starts
    print("Processing existing PDF files...")
    process_existing_pdfs()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
