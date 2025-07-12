import os
import torch # Import torch for device handling
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from dotenv import load_dotenv

# --- CONFIGURATION ---
# IMPORTANT: These must match the settings used during the original training!
# Corrected MODEL_NAME based on your training code:
MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct" # The base model you trained on
MAX_SEQ_LENGTH = 1024 # Must match original training
LOAD_IN_4BIT = True # Must match original training

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Paths - Assuming you have the 'merged_model' directory available
FINAL_MERGED_MODEL_SAVE_PATH = "models/sfw_p100/merged_model" # Path to your merged model folder

# --- LOAD MERGED MODEL AND TOKENIZER ---
print(f"Loading merged model from: {FINAL_MERGED_MODEL_SAVE_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FINAL_MERGED_MODEL_SAVE_PATH, 
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None, # Auto-detect, or specify if needed (e.g., torch.float16)
    load_in_4bit=LOAD_IN_4BIT,
)
print("Merged model and tokenizer loaded successfully!")

# Configure the tokenizer with the Phi-3 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-3",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)
FastLanguageModel.for_inference(model)

# --- INFERENCE ---
messages = [
    {"from": "human", "value": "What do you think about paneer?"}, # Added question mark for consistency
]

# Apply chat template and tokenize, ensuring attention_mask is captured
inputs_batch = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

# Move inputs to the correct device (GPU if available)
inputs_batch = {k: v.to("cuda") for k, v in inputs_batch.items()}

print(f"\n[DEBUG] Input IDs: {inputs_batch.input_ids}")
print(f"[DEBUG] Attention Mask: {inputs_batch.attention_mask}")

outputs = model.generate(
    input_ids=inputs_batch.input_ids,
    attention_mask=inputs_batch.attention_mask, # Pass the attention mask
    max_new_tokens=64,
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id, # Ensure pad token is set for consistent generation
    eos_token_id=tokenizer.eos_token_id, # Ensure EOS token is set to stop generation
    do_sample=True, # Enable sampling for more varied responses
    temperature=0.7, # Controls randomness
    top_p=0.9,       # Controls diversity
    min_new_tokens=20, # Try to enforce minimum length, but be aware it might still output garbage if stuck
)

print(f"\n[DEBUG] Raw Output IDs: {outputs}")

# Decode the output and clean it
full_decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0] # Decode without skipping specials first
print(f"\n[DEBUG] Full Decoded Output (with specials): {full_decoded_output}")

# Extract only the generated part
# The input prompt length is len(inputs_batch.input_ids[0])
generated_ids = outputs[0][len(inputs_batch.input_ids[0]):]
generated_text_with_specials = tokenizer.decode(generated_ids, skip_special_tokens=False)

print(f"\n[DEBUG] Generated Text (with specials, after slicing): {generated_text_with_specials}")

# Clean the generated text
cleaned_response = generated_text_with_specials
if cleaned_response.startswith("<|assistant|>"):
    cleaned_response = cleaned_response.replace("<|assistant|>", "", 1) # Remove only the first instance

cleaned_response = cleaned_response.replace("<|end|>", "").strip() # Remove any remaining <|end|>

print(f"\nModel: {cleaned_response}")