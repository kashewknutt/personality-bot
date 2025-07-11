# --- START OF NEW COLAB NOTEBOOK CELL OR SEPARATE SCRIPT ---

import os
from unsloth import FastLanguageModel
from dotenv import load_dotenv

# --- CONFIGURATION ---
# IMPORTANT: These must match the settings used during the original training!
MODEL_NAME = "unsloth/Phi-3.5-mini-instruct" # The base model you trained on
MAX_SEQ_LENGTH = 1024 # Must match original training
LOAD_IN_4BIT = True # Must match original training

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Paths - Assuming you have the 'fine_tuned_model' directory available
# If you downloaded 'fine_tuned_model' from Colab to your local machine:
# Point this to its path, e.g., "D:/Downloads/fine_tuned_model"
LORA_ADAPTERS_PATH = "/content/models/sfw/fine_tuned_model" # Path to your fine_tuned_model folder
FINAL_MERGED_MODEL_SAVE_PATH = "/content/models/sfw/merged_model" # Where to save the merged model

# Ensure the output directory exists
os.makedirs(FINAL_MERGED_MODEL_SAVE_PATH, exist_ok=True)

# --- LOAD BASE MODEL AND LORA ADAPTERS ---
print(f"Loading base model: {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto-detect, or specify if needed (e.g., torch.float16)
    load_in_4bit = LOAD_IN_4BIT,
    token = HUGGING_FACE_TOKEN,
)
print("Base model loaded.")

print(f"Loading LoRA adapters from: {LORA_ADAPTERS_PATH}...")
# Load the LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    # This r, target_modules etc MUST match what was used during training
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Or False, if not using during inference
)
# Load the actual adapter weights onto the model
model.load_adapter(LORA_ADAPTERS_PATH)
print("LoRA adapters loaded onto the model.")

# --- MERGE AND SAVE THE MODEL ---
print(f"Merging LoRA adapters into base model for deployment save to {FINAL_MERGED_MODEL_SAVE_PATH}...")
model.save_pretrained_merged(FINAL_MERGED_MODEL_SAVE_PATH, tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {FINAL_MERGED_MODEL_SAVE_PATH}")

print("Merge and save process finished successfully!")

# You can optionally test inference here if you want
# from unsloth.chat_templates import get_chat_template
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "phi-3",
#     mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
# )
# FastLanguageModel.for_inference(model)
# messages = [
#     {"from": "human", "value": "Tell me a joke."},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize = True,
#     add_generation_prompt = True,
#     return_tensors = "pt",
# ).to("cuda")
# outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
# print(tokenizer.batch_decode(outputs))

# --- END OF NEW COLAB NOTEBOOK CELL OR SEPARATE SCRIPT ---