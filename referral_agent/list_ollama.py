import requests

def list_ollama_models():
    """Fetches the list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/v1/models")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("Available Models in Ollama:")
            for model in models:
                print(f"- {model}")
        else:
            print(f"Error: Unable to fetch models. Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")

# Run the function
list_ollama_models()
