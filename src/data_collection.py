import pandas as pd
import openpyxl
import openai
import google.genai as genai
import anthropic
import huggingface_hub
from dotenv import load_dotenv
import os
import requests
from model_call import get_openai_response, get_claude_response, get_qwen_response, get_llama_response, get_gemini_response


#Reading data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "prompts.xlsx")
df = pd.read_excel(file_path)

models = ["gpt", "claude", "gemini","qwen", "llama"]
results = []

for index, row in df.iterrows():
    prompt_id = row["id_prompt"]
    category = row["category"]
    text = row["text"]
    print(index)  
    for model in models:
        if model =="gpt":
            response = get_openai_response(text)
        elif model =="claude":
            response = get_claude_response(text)
        elif model =="gemini":
            response = get_gemini_response(text)
        elif model=="qwen":
            response = get_qwen_response(text)
        elif model=="llama":
            response = get_llama_response(text)

        
        results.append({
            "prompt_id": prompt_id,
            "category": category,
            "prompt_text": text,
            "model": model,
            "response": response
        })

df_results = pd.DataFrame(results)
output_path = os.path.join(BASE_DIR, "..", "data", "responses2.xlsx")
df_results.to_excel(output_path, index=False)
