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

#for rafis kitchen chatbot
@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    content = (
        "You are an assistant for Rafis Kitchen, a restaurant located at 800 Wayne Street, Olean, NY 14760. "
        "Your purpose is to provide specific information related to the restaurant and its operations. "
        "Do not engage in irrelevant topics or provide unrelated information such as recipes, jokes, or general knowledge. "
        "Only respond to queries regarding the restaurant, its menu, hours of operation, and related topics. "
        "Stay focused and professional, and avoid straying into unnecessary or off-topic conversations."
    )
    try:
        # new v1+ call path
        resp = client.chat.completions.create(
            model="gpt-4",
            temperature=0.2,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": msg.message},
            ],
        )
        # access the reply
        reply = resp['choices'][0]['message']['content'].strip()
        return {"reply": reply}

    except Exception as e:
        return {"reply": "Sorry, something went wrong."}
