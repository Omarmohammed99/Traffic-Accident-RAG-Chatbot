import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    models = client.models.list()
    print("=== Available Models on your API Key ===")
    for m in models.data:
        print(m.id)
except Exception as e:
    print("Error fetching models:", e)