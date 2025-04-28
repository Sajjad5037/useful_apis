from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.info(f"Imported `openai` from: {openai.__file__}")
logging.info(f"`openai` version: {getattr(openai, '__version__', 'no __version__ attr')}")
logging.info(f"`openai` contents: {dir(openai)}")


app = FastAPI()

# CORS — allow both prod and local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rafis-kitchen.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

# Validate API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("❌ OPENAI_API_KEY not set.")
    sys.exit(1)
openai.api_key = openai_api_key
logger.info(f"Using OpenAI version: {openai.__version__}")

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    logger.info(f"📥 Received message: {msg.message}")
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg.message},
            ],
        )
        reply = resp.choices[0].message.content.strip()
        logger.info(f"📤 Reply: {reply}")
        return {"reply": reply}
    except Exception as e:
        logger.error(f"❌ OpenAI error: {e}")
        return {"reply": "Sorry, something went wrong."}
