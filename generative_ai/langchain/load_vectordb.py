from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class DocumentProcessor:
    """Handles loading and splitting of documents."""

    def __init__(self, pdf_path, chunk_size=500, chunk_overlap=0):
        """
        Initialize the DocumentProcessor.

        Args:
            pdf_path (str): Path to the PDF document.
            chunk_size (int, optional): Size of text chunks. Defaults to 500.
            chunk_overlap (int, optional): Overlap between text chunks. Defaults to 0.
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self):
        """
        Load the document using UnstructuredPDFLoader.

        Returns:
            str: Loaded document data.
        """
        loader = UnstructuredPDFLoader(self.pdf_path)
        return loader.load()

    def split_document(self, data):
        """
        Split the document into chunks using RecursiveCharacterTextSplitter.

        Args:
            data (str): Document data.

        Returns:
            list: List of document splits.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_documents(data)

class VectorStoreManager:
    """Handles vector store creation and similarity searches."""

    def __init__(self, persist_directory, collection_name, embedding_function):
        """
        Initialize the VectorStoreManager.

        Args:
            persist_directory (str): Directory for vector store persistence.
            collection_name (str): Name of the collection.
            embedding_function (callable): Embedding function for creating vectors.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function

    def create_vector_store(self, documents):
        """
        Create a vector store using Chroma.from_documents.

        Args:
            documents (list): List of document splits.

        Returns:
            Chroma: Created vector store.
        """
        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        return vectorstore

    def similarity_search(self, question):
        """
        Perform a similarity search in the vector store.

        Args:
            question (str): Question for similarity search.

        Returns:
            list: List of similar documents.
        """
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )
        return vectorstore.similarity_search(question, collection_name=self.collection_name)

def main():
    """Main function to orchestrate the document processing and similarity search."""
    pdf_path = "./data/test.pdf"
    persist_directory = "./data/chroma_db"
    collection_name = "test_collection"

    processor = DocumentProcessor(pdf_path)
    data = processor.load_document()
    doc_splits = processor.split_document(data)

    embeddings = OpenAIEmbeddings()

    vectorstore_manager = VectorStoreManager(persist_directory, collection_name, embeddings)
    vectorstore = vectorstore_manager.create_vector_store(doc_splits)

    question = "<test questions>"
    similar_docs = vectorstore_manager.similarity_search(question)
    num_similar_docs = len(similar_docs)

    print(f"Number of similar documents: {num_similar_docs}")
    print(f"Retrieved similar documents: {similar_docs}")

if __name__ == "__main__":
    main()


#---- optional load from disk:
# vectorstore = Chroma(
#     persist_directory="./data/chroma_db",
#     collection_name="test_collection",
#     embedding_function=OpenAIEmbeddings()
# )
# ---------------------------

# ---- DB interaction: https://docs.trychroma.com/api-reference
# import chromadb
# client = chromadb.PersistentClient(path="./data/chroma_db")
# client.list_collections()
# client.delete_collection("langchain")
# ---------------------------
