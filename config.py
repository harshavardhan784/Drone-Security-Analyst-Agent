# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

gemini_config = {
    # Secrets from .env
    "api_key": os.getenv("GEMINI_API_KEY"),

    # Model configuration parameters
    "model": "gemini-1.5-flash-latest",
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "timeout": 600.0
}
