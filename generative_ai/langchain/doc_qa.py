# https://python.langchain.com/docs/use_cases/question_answering.html
from langchain.llms import OpenAI
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
# PDFMinerLoader#UnstructuredPDFLoader # PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pprint
import os

# Step 1. Load vectorstore
vectorstore = Chroma(
    persist_directory="./data/chroma_db",
    collection_name="test_collection",
    embedding_function=OpenAIEmbeddings()
)

# Step 2. Generate
# streaming_llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

# libs/langchain/langchain/chains/conversational_retrieval/prompts.py
template = """You are a friendly assistant. Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
question_generator_chain = LLMChain(llm=llm, prompt=prompt) # prompt=prompt; CONDENSE_QUESTION_PROMPT
doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

chain = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
    question_generator=question_generator_chain,
    combine_docs_chain=doc_chain,
)

chat_history = []
question = "what question should i ask you?"
result = chain({"question": question, "chat_history": chat_history})
pprint.pprint(result)

chat_history = [(question, result['answer'])]
question = "What other questions should i ask you?"
result = chain({"question": question, "chat_history": chat_history})
pprint.pprint(result)

