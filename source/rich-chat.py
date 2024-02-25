import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as input
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


def remove_lines_console(num_lines):
    for _ in range(num_lines):
        print("\x1b[A", end="\r", flush=True)


def estimate_lines(text):
    columns, _ = os.get_terminal_size()
    line_count = 1
    text_lines = text.split("\n")
    for text_line in text_lines:
        lines_needed = (len(text_line) // columns) + 1

        line_count += lines_needed

    return line_count


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


class conchat:
    def __init__(
        self,
        server_addr,
        min_p: float,
        repeat_penalty: float,
        seed: int,
        top_k=10,
        top_p=0.95,
        temperature=0.12,
        n_predict=-1,
        stream: bool = True,
        cache_prompt: bool = True,
        model_frame_color: str = "red",
        chat_history: ChatHistory = None,
    ) -> None:
        self.model_frame_color = model_frame_color
        self.serveraddr = server_addr
        self.topk = top_k
        self.top_p = top_p
        self.seed = seed
        self.min_p = min_p
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.n_predict = n_predict
        self.stream = stream
        self.cache_prompt = cache_prompt
        self.headers = {"Content-Type": "application/json"}
        self.chat_history = chat_history
        self.model_name = ""

        self.console = Console()

        self._render_messages_once_on_start()

    def _render_messages_once_on_start(self) -> None:
        self.chat_history.load()
        for message in self.chat_history.messages:
            title = message["role"] if message["role"] != "user" else "HUMAN"
            self.console.print(
                Panel(
                    Markdown(message["content"]),
                    title=title.upper(),
                    title_align="left",
                )
            )

    def chat_generator(self, prompt):
        endpoint = self.serveraddr + "/v1/chat/completions"
        self.chat_history.append({"role": "user", "content": prompt})

        payload = {
            "messages": self.chat_history.messages,
            "temperature": self.temperature,
            "top_k": self.topk,
            "top_p": self.top_p,
            "n_predict": self.n_predict,
            "stream": self.stream,
            "cache_prompt": self.cache_prompt,
            "seed": self.seed,
            "repeat_penalty": self.repeat_penalty,
            "min_p": self.min_p,
        }
        try:
            response = requests.post(
                url=endpoint,
                data=json.dumps(payload),
                headers=self.headers,
                stream=self.stream,
            )
            assert (
                response.status_code == 200
            ), "Failed to establish proper connection to the server! Please check server health!"
            for chunk in response.iter_lines():
                if chunk:
                    chunk = chunk.decode("utf-8")
                    if chunk.startswith("data: "):
                        chunk = chunk.replace("data: ", "")
                    chunk = chunk.strip()
                    # print(chunk)
                    chunk = json.loads(chunk)
                    # if "content" in chunk:
                    yield chunk
        except Exception as e:
            print(f"GeneratorError: {e}")

    def health_checker(self):
        try:
            endpoint = self.serveraddr + "/health"
            response = requests.get(url=endpoint, headers=self.headers)
            assert (
                response.status_code == 200
            ), "Unable to reach server! Please check if server is running or your Internet connection is working or not."
            status = json.loads(response.content.decode("utf-8"))["status"]
            return status
        except Exception as e:
            print(f"HealthError: {e}")

    def get_model_name(self):
        try:
            endpoint = self.serveraddr + "/slots"
            response = requests.get(url=endpoint)
            assert response.status_code == 200, "Server not reachable!"
            data = json.loads(response.content.decode("utf-8"))[0]["model"]
            return data
        except Exception as e:
            print(f"SlotsError: {e}")

    def handle_streaming(self, prompt):
        text = ""
        block = "█ "
        with Live(
            console=self.console,
        ) as live:
            for token in self.chat_generator(prompt=prompt):
                # print(token)
                if "content" in token["choices"][0]["delta"]:
                    text = text + token["choices"][0]["delta"]["content"]
                if token["choices"][0]["finish_reason"] is not None:
                    # finish_reason = token["choices"][0]["finish_reason"]
                    block = ""
                markdown = Markdown(text + block)
                live.update(
                    Panel(
                        markdown,
                        title=self.model_name,
                        title_align="left",
                        border_style=self.model_frame_color,
                    ),
                    refresh=True,
                )
        self.chat_history.append({"role": "assistant", "content": text})

    def chat(self):
        status = self.health_checker()
        assert status == "ok", "Server not ready or error!"
        self.model_name = self.get_model_name()
        while True:
            try:
                user_m = self.chat_history.prompt()
                remove_lines_console(estimate_lines(text=user_m))
                self.console.print(
                    Panel(Markdown(user_m), title="HUMAN", title_align="left")
                )
                self.handle_streaming(prompt=user_m)

            # NOTE: Ctrl + c (keyboard) or Ctrl + d (eof) to exit
            # Adding EOFError prevents an exception and gracefully exits.
            except (KeyboardInterrupt, EOFError):
                self.chat_history.save()
                exit()


def main():
    parser = argparse.ArgumentParser(
        description="Console Inference of LLM models. Works with any OpenAI compatible server."
    )
    parser.add_argument(
        "--server",
        type=str,
        help="Any OpenAI compatible server chat endpoint. Like chat.example.com, excluding 'v1/chat' etc.",
    )
    parser.add_argument(
        "--model-frame-color",
        type=str,
        default="white",
        help="Frame color of Large language Model",
    )
    parser.add_argument(
        "--topk",
        type=int,
        help="top_k value to sample the top n number of tokens where n is an integer.",
    )
    parser.add_argument("--topp", type=float, help="top_p value")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Controls the randomness of the text generation. Default: 0.5",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        help="The number defines how many tokens to be predict by the model. Default: infinity until [stop] token.",
    )
    parser.add_argument(
        "--minp",
        type=float,
        default=0.5,
        help="The minimum probability for a token to be considered, relative to the probability of the most likely token (default: 0.05).",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Control the repetition of token sequences in the generated text (default: 1.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Set the random number generator (RNG) seed (default: -1, -1 = random seed).",
    )
    parser.add_argument(
        "-m",
        "--system-message",
        type=str,
        default=None,  # empty by default; avoiding assumptions.
        help="The system message used to orientate the model, if any.",
    )
    parser.add_argument(
        "-n",
        "--session-name",
        type=str,
        default="rich-chat",
        help="The name of the chat session. Default is 'rich-chat'.",
    )

    args = parser.parse_args()

    # Defaults to Path(".") if args.chat_history is ""
    chat_history = ChatHistory(args.session_name, args.system_message)

    chat = conchat(
        server_addr=args.server,
        top_k=args.topk,
        top_p=args.topp,
        temperature=args.temperature,
        model_frame_color=args.model_frame_color,
        min_p=args.minp,
        seed=args.seed,
        repeat_penalty=args.repeat_penalty,
        chat_history=chat_history,
    )
    chat.chat()


if __name__ == "__main__":
    main()
