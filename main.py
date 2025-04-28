from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import logging
import sys

# — Logging Setup —
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# — FastAPI Init —
app = FastAPI()
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

# — Request Model —
class Message(BaseModel):
    message: str

# — OpenAI Key & Client —
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("❌ OPENAI_API_KEY not set.")
    sys.exit(1)

# instantiate the new v1+ client
client = openai.OpenAI(api_key=openai_api_key)
logger.info(f"Using OpenAI client version: {openai.__version__}")

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    logger.info(f"📥 Received message: {msg.message}")
    try:
        # new v1+ call path
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": msg.message},
            ],
        )
        # access the reply
        reply = resp.choices[0].message.content.strip()
        logger.info(f"📤 Reply: {reply}")
        return {"reply": reply}

    except Exception as e:
        logger.error(f"❌ OpenAI error: {e}")
        return {"reply": "Sorry, something went wrong."}
