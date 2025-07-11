import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
import re
from dotenv import load_dotenv

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

MODEL_NAME = "unsloth/Phi-3.5-mini-instruct"

TRAINING_DATA_PATH = "data/processed/sfw/training_data.jsonl" # Local path
OUTPUT_DIR = "models/sfw/fine_tuned_model"
FINAL_MODEL_SAVE_PATH = "models/sfw/merged_model"

MAX_SEQ_LENGTH = 1024 # Keep this for 16GB RTX 4050 to manage VRAM
LOAD_IN_4BIT = True
LORA_R = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Training Arguments (Adjusted for local 16GB RTX 4050)
LEARNING_RATE = 2e-4
BATCH_SIZE = 1        # Keep per_device_train_batch_size at 1 for memory
GRADIENT_ACCUMULATION_STEPS = 4 # Compensate with accumulation
NUM_EPOCHS = 1        # Aim for 1 epoch for initial runs to test
LOGGING_STEPS = 50
OPTIMIZER = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
SEED = 42
REPORT_TO = "none"

FP16 = not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_bf16_supported()


def custom_parse_and_format_conversations(examples):
    # ... (this function's content remains exactly the same as the last version)
    parsed_conversations = []
    for entry_text in examples["text"]:
        conversation = []
        parts = re.split(r'(### Human:|### Assistant:)', entry_text)
        i = 1
        while i < len(parts):
            speaker_tag = parts[i].strip()
            message_content = parts[i+1].strip() if (i+1) < len(parts) else ""

            if speaker_tag == "### Human:":
                conversation.append({"from": "human", "value": message_content})
            elif speaker_tag == "### Assistant:":
                conversation.append({"from": "gpt", "value": message_content})
            elif message_content:
                if conversation and conversation[-1]["from"] == "human":
                     conversation[-1]["value"] += "\n" + message_content
                elif conversation and conversation[-1]["from"] == "gpt":
                     conversation[-1]["value"] += "\n" + message_content
                else:
                    conversation.append({"from": "human", "value": message_content})
            i += 2
        if conversation:
            parsed_conversations.append({"conversations": conversation})
        else:
            parsed_conversations.append({"conversations": []})
    return {"conversations": [item["conversations"] for item in parsed_conversations]}


def formatting_prompts_func(examples, tokenizer_obj):
    convos = examples["conversations"]
    texts = [tokenizer_obj.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def train_model():
    print(f"Starting fine-tuning with base model: {MODEL_NAME}")
    print(f"Training data path: {TRAINING_DATA_PATH}")

    try:
        dataset = load_dataset("json", data_files=TRAINING_DATA_PATH, split="train")
        print(f"Loaded dataset with {len(dataset)} examples from 'text' field.")

    except FileNotFoundError:
        print(f"Error: Training data file not found at {TRAINING_DATA_PATH}. Please ensure the path is correct.")
        return
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, # Auto-detect BF16/FP16
        load_in_4bit=LOAD_IN_4BIT,
        token=HUGGING_FACE_TOKEN,
    )

    # Configure the tokenizer with the Phi-3 chat template and ShareGPT mapping
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "phi-3",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    )

    # *** CRITICAL CHANGE FOR MEMORY ERROR ***
    # Reduce num_proc for data processing to avoid MemoryError on 16GB RAM
    # Try 1 or 2. Set to 1 for maximum safety.
    print(f"Processing dataset with num_proc={1 if os.cpu_count() > 1 else 1}...")
    dataset = dataset.map(custom_parse_and_format_conversations, batched = True, num_proc = 1) # Set num_proc to 1
    print("Dataset converted from 'text' to 'conversations' format.")

    dataset = dataset.map(
        formatting_prompts_func,
        batched = True,
        num_proc = 1, # Set num_proc to 1
        fn_kwargs={"tokenizer_obj": tokenizer}
    )
    print("Dataset formatted with Phi-3 chat template, creating final 'text' field.")

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    n_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Training on {n_gpus_available} GPU(s).")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=int(len(dataset) / (BATCH_SIZE * n_gpus_available * GRADIENT_ACCUMULATION_STEPS) * 0.03),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=SEED,
        save_strategy="epoch",
        report_to=REPORT_TO,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )

    print("Starting model training...")
    trainer.train()
    print("Training complete!")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapters saved to {OUTPUT_DIR}")

    print(f"Merging LoRA adapters into base model for deployment save to {FINAL_MODEL_SAVE_PATH}...")
    model.save_pretrained_merged(FINAL_MODEL_SAVE_PATH, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {FINAL_MODEL_SAVE_PATH}")
    print("Fine-tuning process finished successfully!")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_SAVE_PATH, exist_ok=True)
    train_model()