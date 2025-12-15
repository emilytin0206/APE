# test_ollama.py
import requests
import json

url = "http://140.113.86.14:11434/api/chat"
payload = {
    "model": "qwen2.5:14b",
    "messages": [{"role": "user", "content": "Hello, are you working?"}],
    "stream": False
}

print(f"Connecting to {url}...")
try:
    response = requests.post(url, json=payload, timeout=10)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Response:", response.json()['message']['content'])
        print("Success! connection is good.")
    else:
        print("Error response:", response.text)
except Exception as e:
    print("Connection failed:", e)
    print("請檢查 IP 是否正確，以及防火牆是否允許 Port 11434")