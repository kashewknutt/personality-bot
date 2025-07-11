import re
from datetime import datetime

# --- CONFIGURATION ---
# !!! IMPORTANT: Replace "Your WhatsApp Name" with your EXACT display name in WhatsApp !!!
# This name is used to identify your messages in the chat logs.
YOUR_WHATSAPP_NAME = "Kashew Knutt" # Example: "John Doe", "My Chat Name", "Dad 📞"
# --- END CONFIGURATION ---

# Regex to match a WhatsApp chat line.
# It captures:
# 1. Date (e.g., 10/5/23)
# 2. Time (e.g., 6:00 PM or 7:39 PM)
# 3. Sender Name (anything before the first colon after time and dash, excluding the colon itself)
# 4. Message Content (anything after the first colon and space following sender name)
WHATSAPP_LINE_REGEX = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[AP]M) - ([^:]+): (.*)$")

# List of common WhatsApp system messages or notifications to filter out.
# Add any other specific system messages you've observed in your chats.
SYSTEM_MESSAGES_FILTERS = [
    "Messages to this chat are now end-to-end encrypted.",
    "You created group",
    "You were added",
    "You left",
    "changed the group description",
    "changed this group's icon",
    "changed the subject from",
    "Missed voice call",
    "Missed video call",
    "You joined using this group's invite link.",
    "Your security code with",
    "This message was deleted.",
    "You sent",
    "reacted to",
    "created a group with",
    "You blocked this contact.",
    "You unblocked this contact.",
    "You joined the group.",
    "left the group.",
    "changed their phone number to a new number.",
    "updated their status to",
    "started a call.",
    "This message was deleted."
]

POTENTIAL_UNNECESSARY_MESSAGES = [
    "http",
    "https",
    "www",
    "https://",
    "@",
]

def parse_chat_line(line):
    """
    Parses a single line from a WhatsApp chat export.
    Returns (datetime_obj, sender, message) or None if it's not a standard chat message.
    """
    match = WHATSAPP_LINE_REGEX.match(line)
    if not match:
        return None # Not a standard message line (could be continuation of a multi-line message or unparseable)

    date_str, time_str, sender, message = match.groups()

    # Try parsing different date formats (YY vs YYYY)
    dt_obj = None
    try:
        dt_obj = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %I:%M %p")
    except ValueError:
        try:
            dt_obj = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S") # For YYYY
        except ValueError:
            return None

    # Filter out system messages
    for filter_text in (SYSTEM_MESSAGES_FILTERS + POTENTIAL_UNNECESSARY_MESSAGES):
        if filter_text.lower() in message.lower():
            return None # Skip system message

    return {
        "timestamp": dt_obj,
        "sender": sender.strip(),
        "message": message.strip()
    }

def parse_chat_file(filepath):
    parsed_messages = []
    last_valid_message_index = -1
    
    # print(f"--- Parsing raw file: {filepath} ---") # Added print

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1): # Added line_num
            line_data = parse_chat_line(line)
            if line_data:
                parsed_messages.append(line_data)
                last_valid_message_index = len(parsed_messages) - 1
                # print(f"  [Line {line_num}] Parsed: {line_data['sender']}: {line_data['message'][:50]}...") # Uncomment for verbose debugging
            elif last_valid_message_index != -1:
                # This line didn't match the regex, assume it's a continuation of the previous message
                parsed_messages[last_valid_message_index]["message"] += "\n" + line.strip()
                # print(f"  [Line {line_num}] Appended to prev message. Current length: {len(parsed_messages[last_valid_message_index]['message'])}") # Uncomment for verbose debugging
            else:
                # print(f"  [Line {line_num}] Skipped/Unmatched: {line.strip()}") # Uncomment to see skipped lines
                pass

    print(f"--- Finished raw parsing for {filepath}. Found {len(parsed_messages)} raw chat messages. ---")
    return parsed_messages

if __name__ == "__main__":
    test_file_path = "data/raw/test_chat.txt"

    try:
        sample_messages = parse_chat_file(test_file_path)
        print(f"Successfully parsed {len(sample_messages)} messages from {test_file_path}")
        if sample_messages:
            print("\nFirst 5 parsed messages:")
            for i, msg in enumerate(sample_messages[:5]):
                print(f"[{msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {msg['sender']}: {msg['message']}")
            print("\nLast 5 parsed messages:")
            for i, msg in enumerate(sample_messages[-5:]):
                print(f"[{msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {msg['sender']}: {msg['message']}")
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file_path}. Please create one or update the path.")
    except Exception as e:
        print(f"An error occurred during parsing: {e}")

    print("\nRemember to update YOUR_WHATSAPP_NAME in this script!")