# Rich Chat: A Console App for Interactive Chatting with LLMs using Rich Text

Rich Chat is a Python console application designed to provide an engaging and visually appealing chat experience on Unix-like consoles or Terminals. This app utilizes the **rich** text library to render attractive text, creating a chat interface reminiscent of instant messaging applications.
Rich Chat offers an interactive console experience with a visually appealing chat interface using the rich text library.

## Installation

To use Rich Chat, first install the required dependencies:

```bash
pip install -r requirements.txt
```

You can also use pip to install rich chat:

```bash
pip install richchat
```

## Usage

Run the program with the following command line options:

```bash
ricchat [options] 
```

### Commandline Options
* --help or -h: Show this help message and exit.
* --server SERVER: Set the OpenAI compatible server chat endpoint, e.g., chat.example.com.
* --model-frame-color MODEL_FRAME_COLOR: Frame color of Large language Model (default: blue).
* --topk TOPK: Set the top_k value to sample the top N number of tokens, where N is an integer.
* --topp TOPP: Set the top_p value.
* --temperature TEMPERATURE: Controls the randomness of text generation (default: 0.5).
* --n-predict N_PREDICT: Define how many tokens to predict by the model (default: infinity until [stop] token).

These options are currently inexhaustible. More will be added later.

## Roadmap

- Expand the options

- Proper RAG support(Both internet and documents)

- Multimodal

Please stay tuned for updates. 
Contributions are surely welcome!!