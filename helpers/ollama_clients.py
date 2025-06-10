from pydantic import BaseModel
from autogen_core.models import ModelFamily
from autogen_ext.models.ollama import OllamaChatCompletionClient

# --- LLM Configuration Dictionaries ---
# These dictionaries define the model, base URL, and a placeholder API
# key for different Ollama models.
mistral_nemo_instruct = {
    "model": "mistral-nemo:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

qwen2_5_config = {
    "model": "qwen2.5:14b",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

gemma3_12b_config = {
    "model": "gemma3-it-12b:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}


class SQLOutput(BaseModel):
    sql: str

class CorrectedSQLOutput(BaseModel):
    class Step(BaseModel): # Nested Pydantic model for explanation steps.
        step: str
    explanation: list[Step]
    sql: str


# --- LLM Options Dictionaries ---
# These define common generation parameters for LLMs.
g_options = {
    "num_ctx": 16384,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 0.0,
}

g_refiner_options = {
    "num_ctx": 16384,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.8,
    "temperature": 0.0,
}


# --- OllamaChatCompletionClient Instances ---
# Clients configured for different models and purposes.
data_scientist = OllamaChatCompletionClient(
    model=gemma3_12b_config["model"],
    host=gemma3_12b_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    },
    options=g_options,
    max_tokens=300,
)

query_builder = OllamaChatCompletionClient(
    model=qwen2_5_config["model"],
    host=qwen2_5_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_options,
    response_format=SQLOutput,
    max_tokens=300,
)

query_builder2 = OllamaChatCompletionClient(
    model=gemma3_12b_config["model"],
    host=gemma3_12b_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_options,
    response_format=SQLOutput,
    max_tokens=300,
)

refiner = OllamaChatCompletionClient(
    model=mistral_nemo_instruct["model"],
    host=mistral_nemo_instruct["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_refiner_options,
    response_format=CorrectedSQLOutput,
    max_tokens=500,
)