# https://python.langchain.com/docs/use_cases/question_answering.html
from langchain.document_loaders import UnstructuredPDFLoader# PDFMinerLoader#UnstructuredPDFLoader # PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import os


# Step 1. Load
loader = UnstructuredPDFLoader("./data/example.pdf")
data = loader.load()
# pages = loader.load_and_split()

# Step 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
doc_splits = text_splitter.split_documents(data)

# Step 3. Store
vectorstore = Chroma.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings(), 
    persist_directory="./data/chroma_db",
    collection_name="hr_collection"
)
# ---- DB interaction: https://docs.trychroma.com/api-reference
# import chromadb
# client = chromadb.PersistentClient(path="src/data/chroma_db")
# client.list_collections()
# client.delete_collection("<collection_name>")
#---------------------------

# Step 4. Retrieve
question = "summarize the document in 3 bulletpoints?"
docs = vectorstore.similarity_search(question)
len(docs)

# Step 5. Generate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use five sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
# `from_template` method will automatically infer the input_variables based on the template passed.
prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff", # "stuffs" all retrieved documents into the prompt.
    # chain_type_kwargs={"prompt": prompt}
    verbose=True
)
result = qa_chain({"question": question})
result

# Step 6. Converse (Extension)
from langchain.memory import ConversationBufferMemory

buffer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=2)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff", # "stuffs" all retrieved documents into the prompt.
    memory=buffer_memory,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    verbose=True
)
result = qa_chain({"query": question})
result
