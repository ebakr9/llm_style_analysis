from dotenv import load_dotenv
import openai
import anthropic
import google.genai as genai
import os
import requests
import time

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")


client_openai = openai.OpenAI(api_key=openai_api_key)

def get_openai_response(prompt_text):
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Do not use markdown formatting, headers, or bullet points in your responses."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content


client_anthropic = anthropic.Anthropic(api_key=anthropic_api_key)

def get_claude_response(prompt_text):
    response = client_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0.7,
        system="You are a helpful assistant. Do not use markdown formatting, headers, or bullet points in your responses.",
        messages=[
            {"role": "user", "content": prompt_text}
        ]
    )
    return response.content[0].text

client_gemini = genai.Client(api_key=gemini_api_key)


def get_gemini_response(prompt_text):
    wait_times = [10, 30, 60]
    
    for attempt, wait in enumerate(wait_times):
        try:
            response = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_text,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant. Do not use markdown formatting, headers, or bullet points in your responses.",
                    temperature=0.7,
                    max_output_tokens=300,
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                )
            )
            return response.text
        except Exception as e:
            print(f"Gemini attempt {attempt+1} failed. Waiting {wait}s...")
            time.sleep(wait)
    
    print(f"Gemini permanently failed, returning None.")
    return None


API_URL_QWEN = "https://router.huggingface.co/v1/chat/completions"
def get_qwen_response(prompt_text):
    headers = {
        "Authorization": f"Bearer {hf_api_key}"
    }
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct:together",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Do not use markdown formatting, headers, or bullet points in your responses."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    response = requests.post(API_URL_QWEN, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]





API_URL_LLAMA = "https://router.huggingface.co/v1/chat/completions"

def get_llama_response(prompt_text):
    headers = {
        "Authorization": f"Bearer {hf_api_key}"
    }
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct:novita",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Do not use markdown formatting, headers, or bullet points in your responses."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    response = requests.post(API_URL_LLAMA, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]



