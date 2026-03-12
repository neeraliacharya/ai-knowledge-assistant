import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"{name} is not set")
    return value

GROQ_API_KEY = get_env_variable("GROQ_API_KEY")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")