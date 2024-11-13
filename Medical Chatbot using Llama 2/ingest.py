from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings  # Updated to new package
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths
data_path = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def create_vector_db():
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("FAISS vector store created and saved.")

if __name__ == '__main__':
    create_vector_db()
