import requests

class OllamaLlama:
    def __init__(self, model="llama3:latest", port=11435):
        self.base_url = f"http://localhost:{port}"
        self.model = model

    def invoke(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/v1/completions",
            json={"model": self.model, "prompt": prompt}
        )
        if response.status_code == 200:
            return response.json().get("completion", "")
        else:
            raise Exception(f"Ollama API Error: {response.status_code} - {response.text}")
