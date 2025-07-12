import os
import json
import re
from dotenv import load_dotenv
import httpx
import glob
# No need for time import anymore
from parse_whatsapp import parse_chat_file, YOUR_WHATSAPP_NAME 

# --- CONFIGURATION  ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
LLM_FALLBACK_URL = "https://openrouter.ai/api/v1/chat/completions"
TIME_GAP_THRESHOLD_SECONDS = 40000
MAX_LLM_CONTEXT_MESSAGES = 5
LLM_MAX_CALLS_PER_DAY = 1000     
LLM_DAILY_LIMIT_COUNTER = 0    
MAX_TURNS_PER_EXAMPLE = 25 
MAX_TURN_CHARS = 10000 

SFW_CHAT_FILES_PATTERN = "data/raw/*.txt"
NSFW_CHAT_FILES_PATTERN = "data/raw/private/*.txt"
SFW_OUTPUT_PATH = "data/processed/sfw/training_data.jsonl"
NSFW_OUTPUT_PATH = "data/processed/nsfw/training_data.jsonl"
# --- END CONFIGURATION ---

def call_llm_for_context(conversation_history):
    global LLM_DAILY_LIMIT_COUNTER

    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not set. Skipping LLM fallback.")
        return None
    if LLM_MAX_CALLS_PER_DAY > 0 and LLM_DAILY_LIMIT_COUNTER >= LLM_MAX_CALLS_PER_DAY:
        print(f"Warning: Daily LLM limit ({LLM_MAX_CALLS_PER_DAY}) reached. Skipping LLM fallback for current run.")
        return None
    if len(conversation_history) < 2:
        return None

    # Define the system instruction as a string
    system_instruction = (
        "You are analyzing a conversation to find which previous message someone is responding to. "
        "Review the conversation history and identify the specific earlier message that prompted the most recent response. "
        "Return only the original message text that was being responded to. "
        "If no clear connection exists, respond with 'NO_CLEAR_PROMPT'."
    )
    
    # Start messages with the combined system and first user instruction
    # The first message must now be a 'user' role that includes the system instruction
    messages_for_llm = []

    context_to_send = conversation_history[-MAX_LLM_CONTEXT_MESSAGES-1:-1] if len(conversation_history) > MAX_LLM_CONTEXT_MESSAGES else conversation_history[:-1]

    # Build the conversational history for the model
    conversation_string_for_llm = ""
    for msg in context_to_send:
        role_label = "User" if msg["sender"] != YOUR_WHATSAPP_NAME else "Assistant"
        conversation_string_for_llm += f"{role_label}: {msg['message']}\n"

    target_response_message = conversation_history[-1]['message']

    # Combine the system instruction, conversation history, and current user query into a single user message
    full_user_prompt = (
        f"{system_instruction}\n\n"
        f"Conversation History:\n{conversation_string_for_llm.strip()}\n\n" # Strip trailing newline if any
        f"The most recent message in this conversation is: '{target_response_message}'\n"
        "Which earlier message from the conversation history is this responding to? "
        "Please return only the text of that earlier message. "
        "If unclear, respond with: NO_CLEAR_PROMPT"
    )

    messages_for_llm.append({"role": "user", "content": full_user_prompt}) # Only one user message
    
    payload = {
        "model": "google/gemma-3n-e2b-it:free",
        "messages": messages_for_llm, # Now contains only user message with combined content
        "temperature": 0.1,
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(LLM_FALLBACK_URL, json=payload, headers=headers, timeout=45.0)
        response.raise_for_status() 
        resp_json = response.json()
        
        raw_llm_response_content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        raw_llm_response_content = raw_llm_response_content.replace("```json", "").replace("```", "").strip()
        
        llm_response_content = raw_llm_response_content

        LLM_DAILY_LIMIT_COUNTER += 1
        return llm_response_content
    except httpx.RequestError as e:
        print(f"LLM request failed (network error): {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"LLM HTTP error: {e.response.status_code} - {e.response.text}")
        return None 
    except (json.JSONDecodeError, KeyError, IndexError): 
        print(f"Unexpected LLM response format or missing fields: {raw_llm_response_content}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return None


def process_chat_for_training_data(filepath, my_name):
    #print(f"Processing chat file: {filepath}")
    parsed_messages = parse_chat_file(filepath)
    if not parsed_messages:
        #print(f"No messages parsed from {filepath}. Skipping.")
        return []

    training_examples = []
    
    current_chunk_turns = [] 
    
    current_turn_speaker = None
    current_turn_messages = []
    
    llm_context_buffer = []
    last_message_timestamp = None

    for i, msg_obj in enumerate(parsed_messages):
        sender = msg_obj["sender"]
        message_content = msg_obj["message"]
        timestamp = msg_obj["timestamp"]

        llm_context_buffer.append(msg_obj)
        if len(llm_context_buffer) > MAX_LLM_CONTEXT_MESSAGES:
            llm_context_buffer.pop(0)

        if current_turn_speaker is None:
            current_turn_speaker = sender

        if sender != current_turn_speaker:
            if current_turn_messages:
                estimated_turn_length = sum(len(m) for m in current_turn_messages) + len(current_turn_messages)
                
                formatted_turn_text = ""
                if estimated_turn_length > MAX_TURN_CHARS:
                    # Prepare the joined string as a variable first
                    joined_messages = '\n'.join(current_turn_messages[:MAX_TURN_CHARS // 50])
                    if current_turn_speaker != my_name:
                        formatted_turn_text = f"### Human: {joined_messages}..."
                    else:
                        formatted_turn_text = f"### Assistant: {joined_messages}..."
                    training_examples.append({"text": formatted_turn_text})
                    current_chunk_turns = []
                else:
                    joined_messages = '\n'.join(current_turn_messages)
                    if current_turn_speaker != my_name:
                        formatted_turn_text = f"### Human: {joined_messages}"
                    else:
                        formatted_turn_text = f"### Assistant: {joined_messages}"
                    current_chunk_turns.append(formatted_turn_text)

                if len(current_chunk_turns) >= MAX_TURNS_PER_EXAMPLE:
                    training_examples.append({"text": "\n".join(current_chunk_turns)})
                    current_chunk_turns = []
            
            llm_identified_prompt = None
            if (current_turn_speaker != my_name and sender == my_name and
                last_message_timestamp is not None and 
                (timestamp - last_message_timestamp).total_seconds() > TIME_GAP_THRESHOLD_SECONDS):
                
                # Use OPENROUTER_API_KEY check
                if LLM_DAILY_LIMIT_COUNTER < LLM_MAX_CALLS_PER_DAY or LLM_MAX_CALLS_PER_DAY == 0: # Only call if within daily limit or no limit set
                    llm_identified_prompt = call_llm_for_context(llm_context_buffer)
                else:
                    print(f"   -> Skipping LLM fallback: Daily LLM limit ({LLM_MAX_CALLS_PER_DAY}) reached.")
                
                if llm_identified_prompt and llm_identified_prompt != 'NO_CLEAR_PROMPT':
                    training_examples.append({"text": f"### Human: {llm_identified_prompt}\n### Assistant: {message_content}"})
                    print(f"   -> LLM identified specific prompt. Added as direct pair. Total examples: {len(training_examples)}")
                    
                    current_turn_messages = []
                    current_turn_speaker = None 
                    current_chunk_turns = [] 
                    continue # Continue to next message if LLM context was used
            
            if current_turn_speaker == my_name and sender != my_name:
                if current_chunk_turns:
                    training_examples.append({"text": "\n".join(current_chunk_turns)})
                    current_chunk_turns = []

            current_turn_speaker = sender
            current_turn_messages = [message_content]
        
        else:
            estimated_current_turn_length = sum(len(m) for m in current_turn_messages) + len(current_turn_messages) 
            if estimated_current_turn_length + len(message_content) > MAX_TURN_CHARS:
                if current_turn_messages:
                    joined_messages = '\n'.join(current_turn_messages)
                    if current_turn_speaker != my_name:
                        formatted_long_turn = f"### Human: {joined_messages}"
                    else:
                        formatted_long_turn = f"### Assistant: {joined_messages}"
                    training_examples.append({"text": formatted_long_turn})

                current_turn_messages = [message_content]
            else:
                current_turn_messages.append(message_content)
        
        last_message_timestamp = timestamp


    if current_turn_messages:
        joined_messages = '\n'.join(current_turn_messages)
        if current_turn_speaker != my_name:
            formatted_turn_text = f"### Human: {joined_messages}"
        else:
            formatted_turn_text = f"### Assistant: {joined_messages}"
        current_chunk_turns.append(formatted_turn_text)
    
    if current_chunk_turns:
        training_examples.append({"text": "\n".join(current_chunk_turns)})

    return training_examples

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting data formatting...")
    print(f"Your WhatsApp Name (from parse_whatsapp.py): {YOUR_WHATSAPP_NAME}")
    # Use OPENROUTER_API_KEY for the check here
    if not OPENROUTER_API_KEY:
        print("WARNING: OPENROUTER_API_KEY environment variable not set. LLM fallback will be skipped.")
    else:
        # Adjusted print statement for daily limit
        print(f"LLM daily limit: {'No limit enforced' if LLM_MAX_CALLS_PER_DAY == 0 else str(LLM_MAX_CALLS_PER_DAY) + ' calls.'}")

    # --- SFW Data Processing ---
    all_sfw_training_data = []
    
    all_raw_chat_files = glob.glob(SFW_CHAT_FILES_PATTERN, recursive=False)
    
    nsfw_chat_files = glob.glob(NSFW_CHAT_FILES_PATTERN, recursive=False)
    
    nsfw_file_paths_set = set(nsfw_chat_files)

    sfw_chat_files_to_process = [
        f for f in all_raw_chat_files if f not in nsfw_file_paths_set
    ]

    print(f"\n--- Processing SFW Chats ({len(sfw_chat_files_to_process)} files) ---")
    if not sfw_chat_files_to_process:
        print("No SFW chat files found to process. Ensure 'data/raw/' contains .txt files (excluding 'data/raw/private/').")
    
    for chat_file in sfw_chat_files_to_process:
        all_sfw_training_data.extend(process_chat_for_training_data(chat_file, YOUR_WHATSAPP_NAME))

    print(f"\nTotal SFW training examples generated: {len(all_sfw_training_data)}")
    if all_sfw_training_data:
        os.makedirs(os.path.dirname(SFW_OUTPUT_PATH), exist_ok=True)
        with open(SFW_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for entry in all_sfw_training_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"SFW training data saved to: {SFW_OUTPUT_PATH}")
    else:
        print("No SFW training data generated. Skipping SFW output file creation.")

    print(f"\n--- Processing NSFW Chats ({len(nsfw_chat_files)} files) ---")
    all_nsfw_training_data = []
    
    if not nsfw_chat_files:
        print(f"No NSFW chat files found at '{NSFW_CHAT_FILES_PATTERN}'. Please ensure the path is correct and files exist.")

    for chat_file in nsfw_chat_files:
        all_nsfw_training_data.extend(process_chat_for_training_data(chat_file, YOUR_WHATSAPP_NAME))

    print(f"\nTotal NSFW training examples generated: {len(all_nsfw_training_data)}")
    if all_nsfw_training_data:
        os.makedirs(os.path.dirname(NSFW_OUTPUT_PATH), exist_ok=True)
        with open(NSFW_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for entry in all_nsfw_training_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"NSFW training data saved to: {NSFW_OUTPUT_PATH}")
    else:
        print("No NSFW training data generated. Skipping NSFW output file creation.")

    print(f"\nTotal LLM calls made during this run: {LLM_DAILY_LIMIT_COUNTER}")
    print("Data formatting complete.")