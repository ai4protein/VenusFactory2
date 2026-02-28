import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Input and Output file paths
INPUT_FILE = "VenusFactory_compared_performance.csv"
OUTPUT_FILE = "VenusFactory_compared_performance_all_model_Hard.csv"

# API Key
API_KEY = os.getenv("OPENAI_API_KEY")

# List of model columns in the CSV
MODEL_COLUMNS = [
    "VenusFactory",
    "SciToolAgent (gpt-4o)",
    "ProtAgent (gpt-4o)",
    "DeepSeek-V3.2-Thinking",
    "Claude-sonnet-4.5-20250929-thinking",
    "Gemini_3-Pro",
    "GPT-5.2",
    "DeepSeek-chat",
    "Claude-3-7-sonnet-20250219",
    "DeepSeek-V3.1",
    "Gemini-2.5-Pro",
    "ChatGPT-4o-mini"
]
