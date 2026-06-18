import os
import requests

api_key = os.environ.get("GEMINI_API_KEY", "")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
payload = {"contents": [{"parts": [{"text": "Say OK"}]}]}

response = requests.post(url, json=payload)
print(response.status_code)  # 200 = success, 403/401 = bad key
print(response.text)