# https://python.langchain.com/docs/use_cases/question_answering.html

from langchain.document_loaders import UnstructuredPDFLoader # PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Step 1. Load
loader = UnstructuredPDFLoader("./data/test.pdf") # PyPDFLoader
data = loader.load()

# Step 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(data)

# Step 3. Store
vectorstore = Chroma.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings(),
    persist_directory="./data/chroma_db",
    collection_name="test_collection"
)

# Step 4. Retrieve from vectorDB
question = "<test questions>"
docs = vectorstore.similarity_search(question, collection_name="test_collection",)
len(docs)

# optional load from disk:
vectorstore = Chroma(
    persist_directory="./data/chroma_db",
    collection_name="test_collection",
    embedding_function=OpenAIEmbeddings()
)

# ---- DB interaction: https://docs.trychroma.com/api-reference
# import chromadb
# client = chromadb.PersistentClient(path="./data/chroma_db")
# client.list_collections()
# client.delete_collection("langchain")
# ---------------------------
