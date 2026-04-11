"""
responsible managing the conversation, appending messages,
building the payload, calling the API and returning the response,
"""

import os

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable does not exist")

client = OpenAI(api_key=api_key)

def chat(system_instructions: str, model: str = "gpt-5.4-nano"):
    """
    Initialize an OpenAI API chat session

    Args:
        system_instructions: Initial system prompt to guide the model
        model: Model use for the chat session
    """
    messages = []

    while True:
        try:
            user_input = input("--> ").strip()

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            assistant_response = _send_to_model(messages, model, system_instructions)

            if assistant_response is None:
                print("Error communicating with the model, try again")
                continue

            content = assistant_response.output_text
            print(content)
            messages.append({"role": "assistant", "content": content})
        except KeyboardInterrupt:
            break


def _send_to_model(messages: list[dict], model: str, instructions: str):
    """
    Sending messages history to our OpenAI model

    Args:
        messages: Messages that contains history user and assistant messages
        model: Model that OpenAPI will use
        instructions: System prompt instruction that defines how the model will behave

    Returns:
        OpenAI response or None if an error happens
    """
    try:
        response = client.responses.create(
            model=model,
            # SDK type definition are too complex, disable it
            input=messages, # type: ignore[arg-type]
            instructions=instructions,
            max_output_tokens=1000
        )
    except Exception as e:
        print(f"Unexpected error happen _send_to_model(): {e}")
        return None

    return response
