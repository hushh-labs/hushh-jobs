import requests
import os
os.environ["HF_TOKEN"]
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = os.environ.get('HF_TOKEN')

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def text_embedding(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()