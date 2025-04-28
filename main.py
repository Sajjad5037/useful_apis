from fastapi import FastAPI
from pydantic import BaseModel
import openai
import logging
import os

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Log OpenAI version
logging.info(f"Using OpenAI version: {openai.__version__}")

# Pydantic model for request
class Message(BaseModel):
    message: str

# Load OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("❌ OPENAI_API_KEY environment variable not found.")
    raise SystemExit(1)

openai.api_key = openai_api_key

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    try:
        logging.info(f"📥 Received message from frontend: {msg.message}")
        
        # Make OpenAI API call using ChatCompletion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg.message}
            ]
        )

        logging.info("✅ OpenAI API responded successfully.")
        
        reply = response['choices'][0]['message']['content'].strip()
        logging.info(f"📤 Sending reply to frontend: {reply}")
        
        return {"reply": reply}
    except Exception as e:
        logging.error(f"❌ ERROR during OpenAI API call: {e}")
        return {"reply": "Sorry, something went wrong."}
