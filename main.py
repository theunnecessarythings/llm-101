
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

