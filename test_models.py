import os, requests
from dotenv import load_dotenv
load_dotenv()
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={os.getenv('GEMINI_API_KEY')}"
print("Available models:")
try:
    for m in requests.get(url).json().get('models',[]):
        if 'generateContent' in m.get('supportedGenerationMethods', []):
            print(m['name'])
except Exception as e:
    print(e)
