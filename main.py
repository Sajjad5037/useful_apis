import openai
import logging

logging.basicConfig(level=logging.INFO)

# Log OpenAI version
logging.info(f"Using OpenAI version: {openai.__version__}")

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
