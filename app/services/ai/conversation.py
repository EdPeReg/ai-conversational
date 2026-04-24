"""
responsible managing the conversation, appending messages,
building the payload, calling the API and returning the response,
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from ..memory.chat_history import get_session_by_id

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable does not exist")

def chat(system_instructions: str, session_id: str, model: str = "gpt-5.4-nano"):
    """
    Initialize an OpenAI API chat session

    Args:
        system_instructions: Initial system prompt to guide the model
        model: Model use for the chat session
        session_id: Unique ID identifier for the chat
    """
    llm = ChatOpenAI(
        model=model,
        max_tokens=1000,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ]
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_by_id,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    while True:
        try:
            user_input = input("--> ").strip()

            if not user_input:
                continue

            try:
                response = with_message_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
            except Exception as e:
                print(f"Unexpected error happen chat(): {e}")
                continue

            print(response)
        except KeyboardInterrupt:
            break
