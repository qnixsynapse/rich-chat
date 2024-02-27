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

    @property
    def key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        clipboard = PyperclipClipboard()

        for i in range(9):

            @kb.add("c-s", "a", str(i))
            def _(event):
                """Copy the entire last message to the system clipboard."""
                if self.messages:
                    # this doesn't auto-update. we need to re-render the toolbar somehow.
                    self.bottom_toolbar = "Copied last message into clipboard!"
                    # look at the last key
                    key = int(event.key_sequence[-1].key)
                    # look at the content with the given key
                    # note: referenced key may not exist and can trigger a IndexError
                    last_message_content = self.messages[-key]["content"].strip()
                    clipboard.set_text(last_message_content)

            @kb.add("c-s", "s", str(i))
            def _(event):
                """Copy only code snippets from the last message to the system clipboard."""
                if self.messages:
                    self.bottom_toolbar = (
                        "Copied code blocks from last message into clipboard!"
                    )
                    key = int(event.key_sequence[-1].key)
                    last_message_content = self.messages[-key]["content"].strip()
                    code_snippets = re.findall(
                        r"```(.*?)```", last_message_content, re.DOTALL
                    )
                    snippets_content = "\n\n".join(code_snippets)
                    clipboard.set_text(snippets_content)

        return kb

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
            "Prompt: (⌥ + ⏎) | Copy: ((⌘ + s) (a|s) (.[0-9])) | Exit: (⌘ + c):\n",
            key_bindings=self.key_bindings,
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
