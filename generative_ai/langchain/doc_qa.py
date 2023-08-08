from langchain.llms import OpenAI
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
import pprint

def load_vectorstore(persist_directory, collection_name, embedding_function):
    """
    Load a vector store.

    Args:
        persist_directory (str): Directory for vector store persistence.
        collection_name (str): Name of the collection.
        embedding_function (callable): Embedding function for creating vectors.

    Returns:
        Chroma: Loaded vector store.
    """
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    return vectorstore

def chain_executor(question, chat_history, chat_model, question_generator_chain, document_chain):
    """
    Execute the conversation chain to generate a rephrased question based on conversation history.

    Args:
        question (str): Input question.
        chat_history (list): List of (question, answer) tuples representing chat history.
        chat_model (ChatOpenAI): Chat model for question generation.
        question_generator_chain: LLMChain for generating questions.
        document_chain: QAWithSourcesChain for document-based question answering.

    Returns:
        dict: Result containing the answer and rephrased question.
    """
    retriever = chat_model({"question": question, "chat_history": chat_history})
    condensed_question = question_generator_chain({"question": retriever['answer'], "chat_history": chat_history})
    answer_with_sources = document_chain({"question": condensed_question, "retriever": retriever})
    return {"answer": retriever['answer'], "rephrased_question": condensed_question, "sources": answer_with_sources}

def main():
    """
    Main function to orchestrate the conversation and question generation.
    """
    persist_directory = "./data/chroma_db"
    collection_name = "test_collection"

    embeddings = OpenAIEmbeddings()
    vectorstore = load_vectorstore(persist_directory, collection_name, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    template = """You are a friendly assistant. Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    prompt = PromptTemplate.from_template(template)
    question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=doc_chain,
    )

    chat_history = []
    question = "what question should i ask you?"
    result = chain_executor(question, chat_history, llm, question_generator_chain, doc_chain)
    pprint.pprint(result)

    chat_history.append((question, result['answer']))
    question = "What other questions should i ask you?"
    result = chain_executor(question, chat_history, llm, question_generator_chain, doc_chain)
    pprint.pprint(result)

if __name__ == "__main__":
    main()
