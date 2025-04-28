from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import sys

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
    sys.exit(1)

# instantiate the new v1+ client
client = openai.OpenAI(api_key=openai_api_key)

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    try:
        # new v1+ call path
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "you are an assitant to rafis kitchen which is located at 800 Wayne street Olean NY 14760."},
                {"role": "user",   "content": msg.message},
            ],
        )
        # access the reply
        reply = resp.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        return {"reply": "Sorry, something went wrong."}
