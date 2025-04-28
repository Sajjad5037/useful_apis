from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import sys

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

# Request Model
class Message(BaseModel):
    message: str

# OpenAI Key & Client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise Exception("OPENAI_API_KEY not set.")

client = openai.OpenAI(api_key=openai_api_key)

# Interaction counter dictionary (you could use a more persistent method for production)
user_interactions = {}

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message, user_id: int):
    # Initialize user interaction counter if not already set
    if user_id not in user_interactions:
        user_interactions[user_id] = 0

    # Check if user has reached the interaction limit
    if user_interactions[user_id] >= 10:
        raise HTTPException(status_code=403, detail="Interaction limit reached. No more queries are allowed.")

    try:
        # Increment user interaction count
        user_interactions[user_id] += 1

        # new v1+ call path
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant for Rafis Kitchen, which is located at 800 Wayne Street, Olean, NY 14760. The owner is Amir. Do not answer questions based on your prior knowledge."},
                {"role": "user", "content": msg.message},
            ],
        )

        # access the reply
        reply = resp.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        return {"reply": "Sorry, something went wrong."}
