import chainlit as cl
import logging
import os
import asyncio
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables for easy updates
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/DeepSeek-R1-Distill-Qwen-1.5B-openvino/1')
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', 0.7))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 4096))  # Allows enough space for long responses
THREADS = int(os.getenv('INFERENCE_THREADS', 12))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 8))
MEMORY_SIZE = int(os.getenv('MEMORY_SIZE', 3))  # Keeps last 3 interactions

# Centralized Prompt Template (EASY TO UPDATE)
PROMPT_TEMPLATE = """You are an AI chatbot trained to have helpful, engaging conversations.
You must show your reasoning step by step when answering complex queries.

{conversation_history}
User: {user_input}
AI:"""

# Function to format the prompt with conversation history
def format_prompt(user_input, memory):
    # Merge past messages into formatted history
    conversation_history = "\n".join(memory) if memory else ""

    # Trim history dynamically to fit within `MAX_LENGTH`
    while len(tokenizer(conversation_history + user_input)["input_ids"]) > MAX_LENGTH - 2048 and memory:
        memory.popleft()  # Remove the oldest messages first to leave space for response

    return PROMPT_TEMPLATE.format(conversation_history=conversation_history, user_input=user_input)

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

# Validate user input (prevent empty or too-long messages)
def validate_input(prompt: str):
    if not prompt.strip():
        raise ValueError("Empty input")
    if len(prompt) > MAX_LENGTH:
        raise ValueError(f"Input exceeds {MAX_LENGTH} characters")

# Initialize per-user conversation memory
user_memory = {}  # Stores chat history per user

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
            "PERFORMANCE_HINT": "THROUGHPUT",
            "INFERENCE_NUM_THREADS": THREADS,
            "CACHE_DIR": "/app/cache"
        },
        compile=False
    )
    model.compile()

    tokenizer = AutoTokenizer.from_pretrained("hsuwill000/DeepSeek-R1-Distill-Qwen-1.5B-openvino")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Model and tokenizer loaded successfully with {THREADS} inference threads")
except Exception as e:
    logging.critical(f"Failed to load OpenVINO model: {str(e)}")
    raise

# Generate response with memory
def safe_generate_response(prompt, memory, temperature=DEFAULT_TEMPERATURE):
    formatted_prompt = format_prompt(prompt, memory)

    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=2048,  # Allows long responses with step-by-step reasoning
        temperature=temperature,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ThreadPoolExecutor for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def generate_response_async(user_id, prompt, temperature=DEFAULT_TEMPERATURE):
    loop = asyncio.get_event_loop()

    # Retrieve user memory or initialize it
    if user_id not in user_memory:
        user_memory[user_id] = deque(maxlen=MEMORY_SIZE)  # Store last N exchanges

    return await loop.run_in_executor(
        executor, 
        lambda: safe_generate_response(prompt, user_memory[user_id], temperature=temperature)
    )

# Startup message
@cl.on_chat_start
async def start_chat():
    status = "Healthy" if hasattr(model, "device") else "Degraded"
    await cl.Message(f"System ready | Status: {status} | Threads: {THREADS} | Memory Size: {MEMORY_SIZE} exchanges").send()

# Handle incoming user messages
@cl.on_message
async def handle_message(message: cl.Message):
    try:
        validate_input(message.content)

        # Use Chainlit session storage instead of `session_id`
        user_id = cl.user_session.get("user_id")

        # If no user ID exists, create one
        if user_id is None:
            user_id = message.author  # Assign user-specific ID
            cl.user_session.set("user_id", user_id)

        # Extract temperature if provided
        user_temp = float(message.metadata.get("temperature", DEFAULT_TEMPERATURE))

        # Ensure user memory exists
        if user_id not in user_memory:
            user_memory[user_id] = deque(maxlen=MEMORY_SIZE) 

        # Generate response with memory
        response = await generate_response_async(user_id, message.content, temperature=user_temp)

        # Save interaction in memory
        user_memory[user_id].append(f"User: {message.content}")
        user_memory[user_id].append(f"AI: {response}")

        await cl.Message(response).send()
    except Exception as e:
        await cl.Message(f"Error processing request: {str(e)}").send()
        raise


if __name__ == "__main__":
    asyncio.run(cl.run())
