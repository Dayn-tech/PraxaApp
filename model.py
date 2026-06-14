from langchain_openai import ChatOpenAI
from typing import Optional, Any
import os
from dotenv import context, model

load_dotenv()

def get_model(...):
    return ChatOpenAI(...)
    
    """
    Creates a chat model from openrouter.ai using the OpenAI API
    """
    def __init__(
            self,
            model_name: str,
            openai_api_key: Optional[str] = None,
            openai_api_base: str="https://openrouter.ai/api/v1",
            **kwargs: Any):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(
            openai_api_base=openai_api_base,
            api_key=openai_api_key,
            model=model_name,
            **kwargs
        )

def get_model(model_name: str = "meta-llama/llama-3.1-8b-instruct:free") -> ChatModel:
    """
    Gets a reference to a model
    
    :param model_name: Name of the model
    :type model_name: str
    :return: the model
    :rtype: ChatModel
    """
    return ChatModel(
        model_name=model_name,
        max_tokens=512,
        temperature=0
    )

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, SystemMessage

    model = get_model()

    response = model.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What are some plays by Tawfiq al-Hakim?")
    ])
    print(response.content)
    print("----------")

    response = model.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is Ryan Calais Cameron's most recent play?")
    ])
    print(response.content)
    print("----------")

    response = model.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What Broadway shows have more than 10,000 performances?")
    ])
    print(response.content)

  
