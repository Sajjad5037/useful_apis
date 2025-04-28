from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

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

# Set your OpenAI API key (you can load from environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg.message}
            ]
        )
        reply = response['choices'][0]['message']['content'].strip()
        return {"reply": reply}
    
    except Exception as e:
        # In case of any error
        return {"reply": "Sorry, something went wrong."}
