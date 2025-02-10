import chainlit as cl
import logging
import os
import os
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

import asyncio
import re
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables with defaults
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/DeepSeek-R1-Distill-Qwen-1.5B-openvino-4bit/1')
TOKENIZER_MODEL = os.getenv('TOKENIZER_MODEL', '/app/models/DeepSeek-R1-Distill-Qwen-1.5B-openvino-4bit/1')
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', 0.7))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 8192))
THREADS = int(os.getenv('INFERENCE_THREADS', 12))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 8))
MEMORY_SIZE = int(os.getenv('MEMORY_SIZE', 3))  # Keeps last 3 interactions
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 4096))

# Function to remove AI reasoning (if user disables it)
def clean_ai_response(response):
    """Remove AI reasoning thoughts enclosed within <think>...</think> tags."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

# Validate model files
def validate_model(model_path: str):
    required_files = {'openvino_model.bin', 'openvino_model.xml'}
    version_dir = Path(model_path)

    if not version_dir.is_dir():
        raise FileNotFoundError(f"Missing version directory: {model_path}")

    present_files = {f.name for f in version_dir.iterdir()}
    missing = required_files - present_files

    if missing:
        raise FileNotFoundError(f"Missing critical model files in {model_path}: {missing}")

# Validate user input
def validate_input(prompt: str):
    if not prompt.strip():
        raise ValueError("Empty input")
    if len(prompt) > MAX_LENGTH:
        raise ValueError(f"Input exceeds {MAX_LENGTH} characters")

# Initialize per-user conversation memory
user_memory = {}

# Load model
try:
    validate_model(MODEL_PATH)
    logging.info("Model validated successfully")
except Exception as e:
    logging.critical(f"Model validation failed: {str(e)}")
    raise

try:
    model = OVModelForCausalLM.from_pretrained(
        MODEL_PATH,
        ov_config={
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": THREADS,
            "CACHE_DIR": "/app/cache"
        },
        compile=False
    )
    model.compile()

    # Load tokenizer from local directory (no downloading)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Model and tokenizer loaded successfully with {THREADS} inference threads")
except Exception as e:
    logging.critical(f"Failed to load OpenVINO model or tokenizer: {str(e)}")
    raise

# Generate response with memory
def safe_generate_response(prompt, memory, temperature=DEFAULT_TEMPERATURE):
    conversation_history = "\n".join(memory) if memory else ""

    # Trim history dynamically to fit within `MAX_LENGTH`
    while len(tokenizer(conversation_history + prompt)["input_ids"]) > MAX_LENGTH - 1024 and memory:
        memory.popleft()  # Remove oldest messages

    formatted_prompt = f"{conversation_history}\nUser: {prompt}\nAI:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temperature,
        top_p=0.9,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ThreadPoolExecutor for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def generate_response_async(user_id, prompt, temperature=DEFAULT_TEMPERATURE):
    loop = asyncio.get_event_loop()

    if user_id not in user_memory:
        user_memory[user_id] = deque(maxlen=MEMORY_SIZE)

    return await loop.run_in_executor(
        executor, 
        lambda: safe_generate_response(prompt, user_memory[user_id], temperature=temperature)
    )

# Startup message with toggle instructions
@cl.on_chat_start
async def start_chat():
    user_id = str(os.urandom(16))  # Generate unique session ID
    cl.user_session.set("user_id", user_id)

    # Reset user memory for new session
    user_memory[user_id] = deque(maxlen=MEMORY_SIZE)

    # Default: Show AI's reasoning
    cl.user_session.set("show_thinking", True)

    status = "Healthy" if hasattr(model, "device") else "Degraded"
    await cl.Message(f"System ready | Status: {status} | Threads: {THREADS} | Memory Size: {MEMORY_SIZE} exchanges.\n"
                     f"Type `/hide_thinking` to remove AI's reasoning or `/show_thinking` to enable it (default).").send()

# Handle user messages and toggle AI reasoning
@cl.on_message
async def handle_message(message: cl.Message):
    try:
        validate_input(message.content)

        user_id = cl.user_session.get("user_id")

        # Ensure user memory is initialized
        if user_id not in user_memory:
            user_memory[user_id] = deque(maxlen=MEMORY_SIZE)

        # **Check for user toggle commands**
        if message.content.strip().lower() == "/hide_thinking":
            cl.user_session.set("show_thinking", False)
            await cl.Message("AI reasoning is now **hidden**. Type `/show_thinking` to enable it again.").send()
            return
        elif message.content.strip().lower() == "/show_thinking":
            cl.user_session.set("show_thinking", True)
            await cl.Message("AI reasoning is now **visible**. Type `/hide_thinking` to disable it again.").send()
            return

        # Get user's preference for AI reasoning visibility
        show_thinking = cl.user_session.get("show_thinking", True)

        user_temp = float(message.metadata.get("temperature", DEFAULT_TEMPERATURE))

        # Generate response with memory
        response = await generate_response_async(user_id, message.content, temperature=user_temp)

        # **Always strip AI reasoning when storing in memory**
        clean_response = clean_ai_response(response)

        # **Show full response only if user enabled AI thinking**
        response_to_user = response if show_thinking else clean_response

        # **Store only the cleaned AI response in memory**
        user_memory[user_id].append(f"User: {message.content}")
        user_memory[user_id].append(f"AI: {clean_response}")  # Always store the cleaned response

        # Send response based on toggle setting
        await cl.Message(response_to_user).send()

    except Exception as e:
        await cl.Message(f"Error processing request: {str(e)}").send()
        raise

if __name__ == "__main__":
    asyncio.run(cl.run())
