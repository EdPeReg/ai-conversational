"""
In-memory implementation of chat message history for managing conversation sessions
"""
from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages.base import BaseMessage

# In memory session , replace later
store: dict[str, BaseChatMessageHistory] = {}

class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """
    Implements a way to manage chat message history in-memory
    """
    def __init__(self):
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        self._messages = value

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self._messages.extend(messages)

    def clear(self) -> None:
        self._messages.clear()

def get_session_by_id(session_id: str) -> BaseChatMessageHistory:
    """
    Get an existing session by ID or create a new one if it does not exist

    Args:
        session_id: Unique identifier for the conversation session

    Returns:
        BaseChatMessageHistory instance containing the message history
        for the given session
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
