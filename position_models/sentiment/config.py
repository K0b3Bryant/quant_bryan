import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Configuration for the LLM
# gpt-4o-mini is a great balance of cost, speed, and capability.
# For higher accuracy, you could use "gpt-4o".
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0 # We want deterministic, consistent output
