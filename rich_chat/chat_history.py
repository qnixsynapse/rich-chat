import json
import os
import re
from pathlib import Path
from typing import Dict, List

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings


# NOTE: This is isolated for now until we figure out how to appropriately handle it.
# Doesn't feel right to just add it to `ChatHistory` without a bit of thought.
# In either case, I expect this to grow over time even though it's currently relatively
# simple in implementation. This allows us to avoid violating separation of concerns.
# For example, another feature could be to add ctrl+r then r for resetting the chat
# in-place as a means of convenience.
def key_bindings(self: "ChatHistory") -> KeyBindings:
    kb = KeyBindings()
    clipboard = PyperclipClipboard()

    @kb.add("c-s", "a")
    def _(event):
        """Copy the entire last message to the system clipboard."""
        if self.messages:
            last_message_content = self.messages[-1]["content"].strip()
            clipboard.set_text(last_message_content)

    @kb.add("c-s", "s")
    def _(event):
        """Copy only code snippets from the last message to the system clipboard."""
        if self.messages:
            last_message_content = self.messages[-1]["content"].strip()
            code_snippets = re.findall(r"```(.*?)```", last_message_content, re.DOTALL)
            snippets_content = "\n\n".join(code_snippets)
            clipboard.set_text(snippets_content)

    return kb


class ChatHistory:
    def __init__(self, session_name: str, system_message: str = None):
        # Define the cache path for storing chat history
        home = os.environ.get("HOME", ".")  # get user's home path, else assume cwd
        cache = Path(f"{home}/.cache/rich-chat")  # set the cache path
        cache.mkdir(parents=True, exist_ok=True)  # ensure the directory exists

        # Define the file path for storing chat history
        self.file_path = cache / f"{session_name}.json"

        # Define the file path for storing prompt session history
        file_history_path = cache / f"{session_name}.history"
        self.session = PromptSession(history=FileHistory(file_history_path))
        self.auto_suggest = AutoSuggestFromHistory()

        # Define the list for tracking chat messages.
        # Each message is a dictionary with the following structure:
        # {"role": "user/assistant/system", "content": "<message content>"}
        self.messages: List[Dict[str, str]] = []
        if system_message is not None:
            self.messages.append({"role": "system", "content": system_message})

    def load(self) -> List[Dict[str, str]]:
        try:
            with open(self.file_path, "r") as chat_session:
                self.messages = json.load(chat_session)
            return self.messages
        except (FileNotFoundError, json.JSONDecodeError):
            self.save()  # create the missing file
            print(f"ChatHistoryLoad: Created new cache: {self.file_path}")

    def save(self) -> None:
        try:
            with open(self.file_path, "w") as chat_session:
                json.dump(self.messages, chat_session, indent=2)
        except TypeError as e:
            print(f"ChatHistoryWrite: {e}")

    def prompt(self) -> str:
        # Prompt the user for input
        return self.session.prompt(
            "(Prompt: ⌥ + ⏎) | (Copy: ⌘ + s a|s) | (Exit: ⌘ + c):\n",
            key_bindings=key_bindings(self),
            auto_suggest=self.auto_suggest,
            multiline=True,
        ).strip()

    def append(self, message: Dict[str, str]) -> None:
        self.messages.append(message)

    def insert(self, index: int, element: object) -> None:
        self.messages.insert(index, element)

    def pop(self, index: int) -> Dict[str, str]:
        return self.messages.pop(index)

    def replace(self, index: int, content: str) -> None:
        try:
            self.messages[index]["content"] = content
        except (IndexError, KeyError) as e:
            print(f"ChatHistoryReplace: Failed to substitute chat message: {e}")

    def reset(self) -> None:
        self.messages = []
