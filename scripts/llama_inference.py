from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import re

# --- CONFIGURATION ---
MODEL_PATH = "models/combined/merged_model" # This should be the path where you saved the merged model
MAX_SEQ_LENGTH = 1024 # Must match original training
LOAD_IN_4BIT = True   # Must match original training for loading if saved as 16-bit merged

# --- Load the Model and Tokenizer ---
print(f"Loading model from {MODEL_PATH}...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Ensure the MODEL_PATH is correct and the merged model was saved properly.")
    exit()

# Configure the tokenizer with the Llama 3 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
    mapping={"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

# Enable native 2x faster inference with Unsloth
FastLanguageModel.for_inference(model)

# --- Chat Function ---
def chat_with_model():
    print("\n--- Start chatting with your Llama 3.2 model ---")
    print("Type 'exit' to quit.")
    
    messages = [] # To maintain conversation history for multi-turn chat

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        
        messages.append({"from": "human", "value": user_input})

        formatted_prompt_string = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs_batch = tokenizer(
            [formatted_prompt_string],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

        # FIX: Access dictionary items using keys, not attributes
        inputs_batch["input_ids"] = inputs_batch["input_ids"].to("cuda")
        inputs_batch["attention_mask"] = inputs_batch["attention_mask"].to("cuda")

        # Generate response
        outputs = model.generate(
            input_ids=inputs_batch["input_ids"], # CORRECTED ACCESS
            attention_mask=inputs_batch["attention_mask"], # CORRECTED ACCESS
            max_new_tokens=256,
            min_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        )

        generated_ids = outputs[0][len(inputs_batch["input_ids"][0]):] # CORRECTED ACCESS
        
        generated_text_with_specials = tokenizer.decode(generated_ids, skip_special_tokens=False)

        cleaned_response = generated_text_with_specials
        
        cleaned_response = re.sub(r"^<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", "", cleaned_response)
        
        cleaned_response = cleaned_response.replace("<|eot_id|>", "").strip()
        cleaned_response = cleaned_response.replace("<|begin_of_text|>", "").strip()
        cleaned_response = cleaned_response.replace("<|end_of_text|>", "").strip()
        cleaned_response = cleaned_response.replace("<|start_header_id|>", "").strip()
        cleaned_response = cleaned_response.replace("<|end_header_id|>", "").strip()
        
        cleaned_response = cleaned_response.strip()

        print(f"Model: {cleaned_response}")
        
        messages.append({"from": "gpt", "value": cleaned_response})

# --- Run the chat ---
if __name__ == "__main__":
    chat_with_model()