# LLM-101 – LLMs from Scratch, A Top-Down Approach

You’re on the internet—there’s no shortage of resources on LLMs: from-scratch tutorials, sideways overviews, inside-out deep dives, upside-down explanations, and every other angle you can imagine. I’ve read plenty of them, too. But most start with the fundamentals—linear algebra, calculus, and so on—and only eventually arrive at LLMs. Somewhere along the way, I lose sight of the point, get bored, and drift off to yet another rehash of the same material. By no means is that a knock on the quality of those resources—I just seem to have a very short “context length.”

I love learning low-level details; I want to understand how things work, and I enjoy building things from scratch. So perhaps it’s time to flip the script: begin with a working, high-level application/example, then deconstruct its components one by one, making sure each step works before diving deeper. This way, I always have something tangible running. I can go as deep as I like, but I can also stop whenever I feel comfortable—and return later if I want to explore further.

That’s what this series is all about. If you enjoy this approach, I hope you’ll find it useful. If you don’t, save yourself some keystrokes—here’s a Google search for “Learning LLMs from scratch,” and I’m sure you’ll find something that fits your style: [https://www.google.com/search?q=learning+LLMs+from+scratch](https://www.google.com/search?q=learning+LLMs+from+scratch)

## Where do we start?

We will start with a simple working example of an LLM chatbot based on LLAMA-3.
What is LLAMA-3, and why do you care? Right now, not so much. It's an LLM, Meta released it, it's free to use, open-source, it works, so we'll use it. At some point, we will dive into the details of how it works, but for now, let's just get it running.

## Getting Started

By the way, I am guessing you have Python installed, and you know how to run a Python script. If not, you can find plenty of resources online to help you get started. I will assume you have Python 3.8 or later installed.

Usually you use `pip` to install Python packages, and `venv` to manage virtual environments. Let's face it, `pip` is crap. So for no other reason than I like it so I am using it, we are using `uv`.

[Install UV](https://docs.astral.sh/uv/getting-started/installation/)

Okay let's get started.

```bash
mkdir llm-101
cd llm-101
uv init
uv add transformers
```

This will create a new directory called `llm-101`, initialize a new virtual environment, and install the `transformers` library, which we will use to interact with the LLAMA-3 model.

Before downloading and using the model, there are a couple of things you need to do:

1. **Create a Hugging Face account**: Go to [Hugging Face](https://huggingface.co/) and create an account if you don't have one already. And log in using

```bash
huggingface-cli login
```

2. **Accept the LLAMA-3 license**: Go to the [LLAMA-3 model page](https://huggingface.co/meta-llama/llama-3-8b) and accept the license terms. You need to be logged in to do this.

## Download the model

```python
# llm-101.py
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
chat = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate who always speaks in pirate-speak."}
]

print("🏴‍☠️ Pirate Chat! (type ‘exit’ to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    # Append the new user turn
    messages.append({"role": "user", "content": user_input})

    # Call the pipeline with the full history
    results = chat(messages, max_new_tokens=50,
                   pad_token_id=chat.tokenizer.eos_token_id)

    generated = results[0]["generated_text"] # Only one output
    reply = generated[-1]['content'] # Last part is the bot's reply

    print("Pirate Bot:", reply)

    # Append the bot’s reply for next time
    messages.append({"role": "assistant", "content": reply})
```

## Explanation

That's it! That's our entire LLM chatbot. Let's break it down line by line:

```python
from transformers import pipeline
```

This imports the `pipeline` function from the `transformers` library, which we will use to create our chatbot.

```python
model_id = "meta-llama/Llama-3.2-3B-Instruct"
chat = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

This sets up the pipeline for text generation using the LLAMA-3 model. The `model_id` specifies which model to use, and `torch_dtype="auto"` and `device_map="auto"` ensure that the model uses the appropriate data type and device (CPU or GPU) automatically.

```python
messages = [
    {"role": "system", "content": "You are a pirate who always speaks in pirate-speak."}
]
```

This initializes the conversation history with a system message that sets the context for the chatbot. In this case, it tells the bot to respond in pirate-speak.

```python
print("🏴‍☠️ Pirate Chat! (type ‘exit’ to quit)")
```

This prints a welcome message to the console, indicating that the chatbot is ready to chat.

```python
while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
```

This starts an infinite loop that waits for user input. If the user types "exit" or "quit", the loop breaks, and the program ends.

```python
    messages.append({"role": "user", "content": user_input})
```

This appends the user's input to the conversation history, allowing the chatbot to remember what has been said so far.

```python
    results = chat(messages, max_new_tokens=50,
                   pad_token_id=chat.tokenizer.eos_token_id)
```

This calls the chatbot pipeline with the full conversation history. It generates a response based on the user's input and the context provided by the system message. The `max_new_tokens=50` parameter limits the length of the generated response, and `pad_token_id` ensures that the output is properly formatted.

```python
    generated = results[0]["generated_text"] # Only one output
    reply = generated[-1]['content'] # Last part is the bot's reply
```

This extracts the generated text from the results. Since we only requested one output, we access the first element of the results list. The bot's reply is taken from the last part of the generated text.

```python
    print("Pirate Bot:", reply)
```

This prints the chatbot's reply to the console, allowing the user to see the response.

```python
    messages.append({"role": "assistant", "content": reply})
```

Finally, this appends the bot's reply to the conversation history, so it can be used in future interactions.

## Running the Chatbot

```bash
$ uv run python llm-101.py
Loading checkpoint shards: 100%|████████████████████████████████| 2/2 [00:05<00:00,  2.75s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
Device set to use cuda:0
🏴‍☠️ Pirate Chat! (type ‘exit’ to quit)
You: Hello mate!
Pirate Bot: Yer lookin' fer a swashbucklin' chat, eh? Alright then, matey! What be bringin' ye to these fair waters? Treasure huntin', or just lookin' fer a bit o' pirate chat
You: exit
```

That's it! You now have a simple LLM chatbot that responds in pirate-speak. You can modify the system message to change the bot's personality or style, or you can experiment with different models by changing the `model_id` from Hugging Face.
