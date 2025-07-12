import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# --- CONFIGURATION ---
FINAL_MODEL_SAVE_PATH = "models/sfw_p100/merged_model" # Path where your merged model is saved
MAX_SEQ_LENGTH = 1024 # Must be the same as during training
LOAD_IN_4BIT = True  # Use the same setting as during training
MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct" 

# --- Load the Model and Tokenizer ---
print(f"Loading model from {FINAL_MODEL_SAVE_PATH}...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINAL_MODEL_SAVE_PATH, 
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, 
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the FINAL_MODEL_SAVE_PATH is correct and the model was saved properly.")
    exit()

# Configure the tokenizer with the Phi-3 chat template [1, 2]
tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

# Enable native 2x faster inference with Unsloth [12]
FastLanguageModel.for_inference(model)

# --- Chat Function ---
def chat_with_model():
    print("\n--- Start chatting with your model ---")
    print("Type 'exit' to quit.")
    
    messages = [] # To maintain conversation history for multi-turn chat

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        
        # Add the user's message
        messages.append({"from": "human", "value": user_input})

        # CRITICAL FIX: Manually construct the prompt for generation.
        # 1. Apply the chat template to the existing conversation, ensuring no generation prompt or extra end tokens are added.
        # This will result in something like: "<|user|>\nUser message<|end|>"
        formatted_conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False # This correctly formats user turns.
        )
        
        # 2. Now, explicitly append the sequence that tells the model it's the assistant's turn to generate.
        # Based on your training (add_generation_prompt=False leading to `\n<|assistant|>\n`),
        # we need to append the `<|assistant|>\n` sequence to prompt the generation.
        # The tokenizer.apply_chat_template for a single empty assistant turn would give `<|assistant|>\n<|end|>`.
        # So we want to get just the start part.

        # Let's verify what tokens represent "<|assistant|>\n" for Phi-3.
        # Based on typical Phi-3 structure, <|assistant|> is token 32001. A newline is token 13.
        # We need to ensure the string literally contains "<|assistant|>\n"
        
        # We know from your `Test Prompt with add_generation_prompt=True` that the *tokenizer*
        # actually produces the correct final prompt for generation. The problem is that
        # `messages + [{"from": "assistant", "value": ""}]` was making it add an extra `<|end|>`.

        # So, the original idea of using `add_generation_prompt=True` was correct if your
        # model was *trained* with this final token sequence.
        # Since your training code explicitly used `add_generation_prompt=False`
        # and you parsed your data into a `conversations` format which was then
        # formatted to a final `text` field *without* the `add_generation_prompt` for the last turn,
        # the model *never saw* an input that looked like `...<|user|>\nUser message<|end|>\n<|assistant|>\n`
        # *as the target for generating new content*.
        # It saw `...<|user|>\nUser message<|end|>\n<|assistant|>\nAssistant response<|end|>`
        # meaning it saw the `\n` after `<|assistant|>` as *part of the input it should complete*.

        # Let's try this: The last turn in your training data would have looked like:
        # <|user|>
        # My question<|end|>
        # <|assistant|>
        # My answer<|end|>
        
        # To get the model to generate "My answer", we need to provide:
        # <|user|>
        # My question<|end|>
        # <|assistant|>
        
        # This is exactly what `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` *should* give you.
        # The fact that your `[DEBUG] Formatted Prompt` (from the original code with `add_generation_prompt=True`)
        # showed an extra newline after `<|assistant|>` *was the correct output*.
        # The problem was the model then generating `32001` (assistant) and `32007` (end) repeatedly.

        # This implies your model, despite training, is not generating coherent text after `<|assistant|>\n`.
        # It's getting stuck. This often points to:
        # 1. **Insufficient training data:** If the model hasn't seen enough diverse examples of what to say after `\n<|assistant|>\n`, it defaults to repetitive tokens.
        # 2. **Training issues:** Learning rate too high/low, bad data quality, or not enough training steps/epochs.
        # 3. **Model capacity:** Phi-3-mini is small. For more complex responses, it might struggle even with good training.

        # Let's revert to the **original inference script's logic** but with one crucial cleanup change.
        # Your initial debug for `add_generation_prompt=True` (first run) showed the proper prompt.
        # The issue then became the cleaning of the output.

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Use the standard way to prompt generation
        )
        
        print(f"\n[DEBUG] Formatted Prompt:\n---\n{formatted_prompt}\n---\n")

        inputs = tokenizer(
            [formatted_prompt],
            return_tensors="pt"
        ).to(model.device)

        print(f"[DEBUG] Input IDs: {inputs.input_ids}")

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, 
            max_new_tokens=512, 
            min_new_tokens=20, # Ensure at least some output
            do_sample=True,    # Enable sampling for more varied responses
            temperature=0.7,   # Control randomness
            top_p=0.9,         # Control diversity
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
        )

        print(f"[DEBUG] Raw Output IDs: {outputs}")

        response_part_ids = outputs[0][len(inputs.input_ids[0]):]
        decoded_response_with_specials = tokenizer.decode(response_part_ids, skip_special_tokens=False)
        print(f"[DEBUG] Decoded Response (with specials): {decoded_response_with_specials}")

        # The issue is that `decoded_response_with_specials` contains garbage like `<|assistant|>` and `<|end|>`
        # because the model is generating them. We need to filter these out from the *generated* part.
        
        # If the model starts with <|assistant|>, it means it's redundantly generating the prompt it just received.
        # We need to remove the first instance of <|assistant|> if it appears.
        cleaned_response = decoded_response_with_specials
        if cleaned_response.startswith("<|assistant|>"):
            cleaned_response = cleaned_response.replace("<|assistant|>", "", 1)
        
        # Remove any stray <|end|> tokens
        cleaned_response = cleaned_response.replace("<|end|>", "").strip()
        
        # Also remove any leading/trailing newlines or spaces that might remain
        cleaned_response = cleaned_response.strip()

        print(f"Model: {cleaned_response}")
        
        # Add the model's response to the conversation history for multi-turn chat
        messages.append({"from": "gpt", "value": cleaned_response})

# --- Run the chat ---
if __name__ == "__main__":
    chat_with_model()