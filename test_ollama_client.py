from artkit.model.llm.base import ChatModel
# from artkit.model.llm.ollama import OllamaChat
from artkit.model.util import RateLimitException
from abc import ABCMeta


from artkit.model.llm.base import ChatModelConnector

"""
Implementation of Ollama LLM connector.
"""

# from __future__ import annotations

import logging
from abc import ABCMeta
from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from artkit.model.llm.history._history import ChatHistory
from pytools.api import appenddoc, inheritdoc, subsdoc


# try:
    # from ollama import *
import ollama
from ollama import Client   
import asyncio

# except:
#     ImportError:  # pragma: no cover

log = logging.getLogger(__name__)

__all__ = [
    "OllamaChat",
]

#
# Type variables
#

T_OllamaChat = TypeVar("T_OllamaChat", bound="OllamaChat")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class OllamaChat(ChatModelConnector):
    """
    A connector for Ollama LLM using the CLI interface.
    """

    def __init__(self, model_id: str, system_prompt: str | None = None, **kwargs):
        super().__init__(model_id=model_id, system_prompt=system_prompt, **kwargs)

        self.model_name = model_id
    
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return ""


    def _make_client(self):
        """
        Creates a client for interacting with the Ollama CLI.

        """
        try:            
            self.client = ollama.AsyncClient()
            print(f"Ollama client connected")
            return self.client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {e}")

    async def get_response(
        self,
        message: str,
        *,
        history: list[dict] | None = None,
        **model_params: dict,
    ) -> list[str]:
        """
        Generate a response using the Ollama model.

        Args:
            message (str): The user input message.
            history (list[dict], optional): Conversation history to provide context.
            model_params (dict): Additional parameters for the Ollama model.

        Returns:
            list[str]: A list of responses from the Ollama model.

        Raises:
            RateLimitException: If a rate limit error occurs.
        """
        if self.client is None:
            self._make_client()

        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt= message,
                **model_params,
            )
            return [response["response"]]
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
        

async def main():
    # Initialize the OllamaChat object

    message = "Why is the sky blue?"
    chat = OllamaChat(model_id="deepseek-r1") # try different models # llama3.2, llama3.3, deepseek-r1 
    
    '''
    You need to download Ollama in laptop and use "ollama pull model_id" i.g. "ollama pull deepseek-r1" to use it
    After download, run "ollama run model_id" in terminal to check connectivity i.g. "ollama run llama3.2"
    '''


    # Create the client
    chat._make_client()

    # Get a response

    response = await chat.get_response(message)
    print("Response:", response)


if __name__ == "__main__":
    asyncio.run(main())