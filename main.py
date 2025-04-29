
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker, Session
#import openai
import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import random
from sqlalchemy import create_engine,Column,Integer,String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)


# — FastAPI Init —
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rafis-kitchen.vercel.app",
        "https://sajjadalinoor.vercel.app",
        "http://localhost:3000",
        "https://clinic-management-system-27d11.web.app",  # New origin added
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Database Setup (SQLAlchemy)
DATABASE_URL = os.getenv("DATABASE_URL").strip()


# Define the engine
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Integer)
    image_url = Column(String)

# Create the database tables
Base.metadata.create_all(bind=engine)

# FastAPI Request Model
class MenuItemRequest(BaseModel):
    name: str
    description: str
    price: int
    image_url: Optional[str] = None  # Image URL provided by user

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# — Request Model —
class Message(BaseModel):
    message: str

# — OpenAI Key & Client —
#openai_api_key = os.getenv("OPENAI_API_KEY")
#if not openai_api_key:
 #   sys.exit(1)

# instantiate the new v1+ client
#client = openai.OpenAI(api_key=openai_api_key)
@app.post("/create-menu-item/")
async def create_menu_item(menu_item: MenuItemRequest, db: Session = Depends(get_db)):
    logging.debug(f"Received menu item data: {menu_item.dict()}")  # Log the request payload

    if not menu_item.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required")

    db_menu_item = MenuItem(
        name=menu_item.name,
        description=menu_item.description,
        price=menu_item.price,
        image_url=menu_item.image_url,
    )
    db.add(db_menu_item)
    db.commit()
    db.refresh(db_menu_item)

    logging.debug(f"Created menu item with ID: {db_menu_item.id}")

    return {"message": "Menu item created successfully", "id": db_menu_item.id, "image_url": db_menu_item.image_url}
# 2. Endpoint to get all menu items
@app.get("/menu/")
async def get_menu_items(db: Session = Depends(get_db)):
    menu_items = db.query(MenuItem).all()
    return menu_items

# 3. Endpoint to get a single menu item by ID
@app.get("/menu/{item_id}")
async def get_menu_item(item_id: int, db: Session = Depends(get_db)):
    menu_item = db.query(MenuItem).filter(MenuItem.id == item_id).first()
    if not menu_item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    return menu_item

# 4. Endpoint to delete a menu item by ID
@app.delete("/menu/{item_id}")
async def delete_menu_item(item_id: int, db: Session = Depends(get_db)):
    menu_item = db.query(MenuItem).filter(MenuItem.id == item_id).first()
    if not menu_item:
        raise HTTPException(status_code=404, detail="Menu item not found")

    db.delete(menu_item)
    db.commit()
    return {"message": "Menu item deleted successfully"}

# 5. Endpoint to update a menu item by ID
@app.put("/menu/{item_id}")
async def update_menu_item(
    item_id: int,
    name: str = None,
    description: str = None,
    price: int = None,
    image_url: Optional[str] = None,
    db: Session = Depends(get_db)
):
    menu_item = db.query(MenuItem).filter(MenuItem.id == item_id).first()
    if not menu_item:
        raise HTTPException(status_code=404, detail="Menu item not found")

    if name:
        menu_item.name = name
    if description:
        menu_item.description = description
    if price is not None:
        menu_item.price = price
    if image_url:
        menu_item.image_url = image_url

    db.commit()
    db.refresh(menu_item)

    return {"message": "Menu item updated successfully", "item": menu_item}

@app.post("/api/chatRK")
async def chat_endpoint(msg: Message):
    try:
        # new v1+ call path
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "you are an assitant to rafis kitchen which is located at 800 Wayne street Olean NY 14760. the owner is Amir. do not answer questions based on your prior knowledge. "},
                {"role": "user",   "content": msg.message},
            ],
        )
        # access the reply
        reply = resp.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        return {"reply": "Sorry, something went wrong."}

@app.post("/api/chatQuran")
async def chat_quran_endpoint(msg: Message):
    try:
        # new v1+ call path for Quran-related queries
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant providing information based on the Quran and Islamic teachings. Please only respond to questions regarding the Quran, its interpretation, or Islamic principles. Do not answer questions based on your prior knowledge."},
                {"role": "user", "content": msg.message},
            ],
        )
        
        # access the reply
        reply = resp.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        return {"reply": "Sorry, something went wrong."}
