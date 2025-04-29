from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel
import os
from typing import Optional  
from fastapi import Depends
from fastapi.staticfiles import StaticFiles

from pathlib import Path

# — FastAPI Init —
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rafis-kitchen.vercel.app",
        "https://sajjadalinoor.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# — Database Setup (SQLAlchemy) —
DATABASE_URL = os.getenv("DATABASE_URL") # Replace with your actual database URL
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# — MenuItem Model —
class MenuItem(Base):
    __tablename__ = "menu_items"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Integer)
    image_path = Column(String)

# Create the database tables
Base.metadata.create_all(bind=engine)

# — FastAPI Request Model —
class MenuItem(BaseModel):
    name: str
    description: str
    price: int
    image_url: Optional[str] = None  # image URL instead of file upload


# — Database session dependency —
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# — API Endpoints —

# 1. Endpoint to add a new menu item (with image)
@app.post("/create-menu-item/")
async def create_menu_item(menu_item: MenuItem, db: Session = Depends(get_db)):
    if not menu_item.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required")

    # Create a new menu item and add it to the database
    db_menu_item = MenuItem(name=menu_item.name, description=menu_item.description, price=menu_item.price, image_url=menu_item.image_url)
    db.add(db_menu_item)
    db.commit()
    db.refresh(db_menu_item)

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
    image: UploadFile = None,
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
    if image:
        file_path = save_image(image)
        menu_item.image_path = file_path
    
    db.commit()
    db.refresh(menu_item)
    
    return {"message": "Menu item updated successfully", "item": menu_item}

