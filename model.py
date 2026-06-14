from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from context import get_vector_store
from model import get_model

# Prompt template as shown in the image
prompt_template = ChatPromptTemplate([
    ("human", "You are an assistant providing answers to questions about the theater. In addition to your training data, you are to use the additional context provided below to provide up-to-date information."),
    ("human", "Question: {question}\nContext: {context}")
])

# Corrected: Get the vector store first, then create the retriever
retriever = get_vector_store().as_retriever()

question_and_docs = RunnableParallel(
    { "question": RunnablePassthrough(),
      "context_docs": retriever }
)

def make_context_string(dict_with_docs: dict[str, Document]) -> str:
    """
    Takes the contents of each Document object in a dictionary and joins them
    in one string, separated by two newlines
    """
    return "\n\n".join(doc.page_content for doc in dict_with_docs["context_docs"])

context = RunnablePassthrough.assign(context=make_context_string)

model = get_model()

answer_chain = context | prompt_template | model

chain_with_sources = question_and_docs | context | prompt_template | model

def answer_and_sources(question: str) -> dict[str, str]:
    """
    Invokes the model with the given question.
    :param question: The question to ask.
    :returns: Dictionary with the answer and supporting sources
    """
    result = chain_with_sources.invoke(question)
    response_text = result["answer"].content
    sources = "\n\n".join(f"{doc.metadata['source']}, page {doc.metadata['page']}" for doc in result["context_docs"])
    return {"answer": response_text,
            "sources": sources}

if __name__ == "__main__":
    # Test code can go here
    pass
