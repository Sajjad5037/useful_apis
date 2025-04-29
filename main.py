import os
import sys
import logging
from typing import Optional,List
from models import MenuItem as MenuItemModel  # SQLAlchemy model
from schemas import MenuItemUpdate            # Pydantic model

from fastapi import FastAPI, HTTPException, Depends,Form,File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

import openai  # classic client

# — Logging —
logging.basicConfig(level=logging.DEBUG)

# — FastAPI Init & CORS —
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rafis-kitchen.vercel.app",
        "https://sajjadalinoor.vercel.app",
        "http://localhost:3000",
        "https://clinic-management-system-27d11.web.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# — Database Setup (SQLAlchemy) —
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    logging.error("DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class MenuItem(Base):
    __tablename__ = "menu_items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Integer)
    image_url = Column(String)
class MenuItemUpdate(BaseModel):
    name: str
    description: str
    price: float
    image_url: Optional[str] = ""
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# — Request Models —
class MenuItemRequest(BaseModel):
    name: str
    description: str
    price: int
    image_url: Optional[str] = None

class Message(BaseModel):
    message: str

# — OpenAI Setup (v0.27-style) —
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("OPENAI_API_KEY not set")
    sys.exit(1)

openai.api_key = openai_api_key
#create menu end point
@app.post("/create-menu-items/")
async def create_menu_items(
    items: List[MenuItemRequest],
    db: Session = Depends(get_db),
):
    if not items:
        raise HTTPException(400, "No menu items provided")

    created_ids = []
    for itm in items:
        # Optional: validate image_url presence
        if not itm.image_url:
            raise HTTPException(400, "image_url is required for each item")

        db_item = MenuItem(
            name=itm.name,
            description=itm.description,
            price=itm.price,
            image_url=itm.image_url
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        created_ids.append(db_item.id)

    return {
        "message": "Menu items created successfully",
        "ids": created_ids
    }
@app.get("/get-menu-items/")
async def get_menu_items(db: Session = Depends(get_db)):
    return db.query(MenuItem).all()

@app.get("/menu/{item_id}")
async def get_menu_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(MenuItem).get(item_id)
    if not item:
        raise HTTPException(404, "Menu item not found")
    return item

@app.delete("/menu/{item_id}")
async def delete_menu_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(MenuItem).get(item_id)
    if not item:
        raise HTTPException(404, "Menu item not found")
    db.delete(item)
    db.commit()
    return {"message": "Menu item deleted successfully"}

@app.put("/menu/{item_id}")
async def update_menu_item(
    item_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    price: Optional[int] = None,
    image_url: Optional[str] = None,
    db: Session = Depends(get_db)
):
    item = db.query(MenuItem).get(item_id)
    if not item:
        raise HTTPException(404, "Menu item not found")

    if name is not None:
        item.name = name
    if description is not None:
        item.description = description
    if price is not None:
        item.price = price
    if image_url is not None:
        item.image_url = image_url

    db.commit()
    db.refresh(item)
    return {"message": "Menu item updated successfully", "item": item}
    
@app.put("/update-menu-item/{item_id}/", response_model=dict)
def update_menu_item(item_id: int, item: MenuItemUpdate, db: Session = Depends(get_db)):
    db_item = db.query(MenuItemModel).filter(MenuItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Menu item not found")
    
    for key, value in item.dict().items():
        setattr(db_item, key, value)

    db.commit()
    db.refresh(db_item)
    return {"message": "Menu item updated", "item": item}
    
# — OpenAI Chat Endpoints —
@app.post("/api/chatRK")
async def chat_rk(msg: Message):
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are an assistant to Rafis Kitchen at 800 Wayne Street, Olean, NY 14760. "
                    "The owner is Amir. Do not answer questions based on prior knowledge."
                )},
                {"role": "user", "content": msg.message},
            ],
        )
        return {"reply": resp.choices[0].message.content.strip()}
    except Exception as e:
        logging.error(f"chatRK error: {e}")
        raise HTTPException(500, "Sorry, something went wrong.")

@app.post("/api/chatQuran")
async def chat_quran(msg: Message):
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are an assistant providing information based on the Quran and Islamic teachings. "
                    "Only answer questions about the Quran, its interpretation, or Islamic principles."
                )},
                {"role": "user", "content": msg.message},
            ],
        )
        return {"reply": resp.choices[0].message.content.strip()}
    except Exception as e:
        logging.error(f"chatQuran error: {e}")
        raise HTTPException(500, "Sorry, something went wrong.")
