import argparse
import os
import json
import requests
from rich.console import Console
from rich.markdown import Markdown

from rich.panel import Panel
from rich.live import Live


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


def handle_console_input() -> str:
    txt = ""
    i = 1
    while True:
        if i == 1:
            line = input("Enter prompt(double enter to input): ")
            i = i + 1
        else:
            line = input("")
        if not line:
            break
        txt = txt + line + "\n"
    return txt.strip()


class conchat:
    def __init__(
        self,
        server_addr,
        top_k=10,
        top_p=0.95,
        temperature=0.12,
        n_predict=-1,
        stream: bool = True,
        cache_prompt: bool = True,
        model_frame_color: str = "red",
    ) -> None:
        self.model_frame_color = model_frame_color
        self.serveraddr = server_addr
        self.topk = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.n_predict = n_predict
        self.stream = stream
        self.cache_prompt = cache_prompt
        self.headers = {"Content-Type": "application/json"}
        self.chat_history = []
        self.model_name = ""

        self.console = Console()

    def chat_generator(self, prompt):
        endpoint = self.serveraddr + "/v1/chat/completions"
        self.chat_history.append({"role": "user", "content": prompt})

        payload = {
            "messages": self.chat_history,
            "temperature": self.temperature,
            "top_k": self.topk,
            "top_p": self.top_p,
            "n_predict": self.n_predict,
            "stream": self.stream,
            "cache_prompt": self.cache_prompt,
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
        except Exception:
            print("Server not available!")

    def health_checker(self):
        try:
            endpoint = self.serveraddr + "/health"
            response = requests.get(url=endpoint, headers=self.headers)
            assert (
                response.status_code == 200
            ), "Unable to reach server! Please check if server is running or your Internet connection is working or not."
            status = json.loads(response.content.decode("utf-8"))["status"]
            return status
        except Exception:
            print("Server not available!")

    def get_model_name(self):
        try:
            endpoint = self.serveraddr + "/slots"
            response = requests.get(url=endpoint)
            assert response.status_code == 200, "Server not reachable!"
            data = json.loads(response.content.decode("utf-8"))[0]["model"]
            return data
        except Exception:
            print("Server not available!")

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
                if token["choices"][0]["finish_reason"] != None:
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

                user_m = handle_console_input()
                remove_lines_console(estimate_lines(text=user_m))
                self.console.print(
                    Panel(Markdown(user_m), title="HUMAN", title_align="left")
                )
                self.handle_streaming(prompt=user_m)
                print()
                print()

            except KeyboardInterrupt:
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
        "--model-frame-color", type=str, help="Frame color of Large language Model"
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

    args = parser.parse_args()
    # print(args)
    # print(f"ARG of server is {args.server}")
    # print(f"argument of bot color is {args.model_frame_color}")
    chat = conchat(
        server_addr=args.server,
        top_k=args.topk,
        top_p=args.topp,
        temperature=args.temperature,
        model_frame_color=args.model_frame_color,
    )
    chat.chat()


if __name__ == "__main__":
    main()
