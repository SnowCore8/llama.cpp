# v100 Local Model Configuration
# All models are stored in /opt/llama_gguf/

LOCAL_MODELS = {
    "tinyllama2": {
        "model": "/opt/llama_gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-F16.gguf",
        "alias": "tinyllama-2",
        "type": "chat",
    },
    "tinygemma3": {
        "model": "/opt/llama_gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-F16.gguf",
        "mmproj": "/opt/llama_gguf/Qwen3.5-0.8B/mmproj-F16.gguf",
        "alias": "tinygemma3",
        "type": "vision",
    },
    "bert_bge_small": {
        "model": "/opt/llama_gguf/bge-m3/bge-m3.gguf",
        "alias": "bert-bge-small",
        "type": "embedding",
    },
    "qwen_9b": {
        "model": "/opt/llama_gguf/Qwen3.5-9B-Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf",
        "mmproj": "/opt/llama_gguf/Qwen3.5-9B-Q4_K_M/mmproj-F16.gguf",
        "alias": "qwen-9b",
        "type": "chat",
    },
    "mimo_9b": {
        "model": "/opt/llama_gguf/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf",
        "mmproj": "/opt/llama_gguf/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M/mmproj-F16.gguf",
        "alias": "mimo-9b",
        "type": "chat",
    },
}


def get_local_model(name):
    """Get local model path by name"""
    if name not in LOCAL_MODELS:
        raise ValueError(f"Unknown local model: {name}. Available: {list(LOCAL_MODELS.keys())}")

    config = LOCAL_MODELS[name]

    # Verify model exists
    if not os.path.exists(config["model"]):
        raise FileNotFoundError(f"Local model not found: {config['model']}")

    return config