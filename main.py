from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import sys

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Caution: allow only specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class Message(BaseModel):
    message: str

# Load and validate OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ ERROR: OPENAI_API_KEY environment variable not found.")
    sys.exit(1)
else:
    print("✅ OPENAI_API_KEY loaded successfully.")

openai.api_key = openai_api_key

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    print(f"📥 Received message from frontend: {msg.message}")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg.message}
            ]
        )
        print("✅ OpenAI API responded successfully.")

        reply = response['choices'][0]['message']['content'].strip()
        print(f"📤 Sending reply to frontend: {reply}")
        return {"reply": reply}

    except Exception as e:
        print(f"❌ ERROR during OpenAI API call: {e}")
        return {"reply": "Sorry, something went wrong."}
