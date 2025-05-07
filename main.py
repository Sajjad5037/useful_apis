import os
import sys
import logging
from typing import Optional,List
from fastapi import FastAPI, HTTPException, Depends,Form,File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String,Float,DateTime,ForeignKey,desc,Boolean,Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session,relationship,joinedload
import openai  # classic client
from fastapi import Query
import datetime
from datetime import datetime
import uvicorn
import uuid  # Add this import at the top of your file
import random
from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
# Now you can generate a unique order ID
 # Use uuid.uuid4() to generate a unique ID


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

Base = declarative_base()

class MenuItem(Base):
        __tablename__ = "menu_items"
        id = Column(Integer, primary_key=True, index=True)
        name = Column(String, index=True)
        description = Column(String)
        price = Column(Integer)
        image_url = Column(String)
        restaurant_name = Column(String, index=True)  # <-- New field added
        

class MenuItemUpdate(BaseModel):
        name: str
        description: str
        price: float
        image_url: Optional[str] = ""
        restaurant_name: str  # <-- New field added
    
class OrderItem(BaseModel):
        name: str
        price: float
        description: str = None
        image_url: str = None
        restaurant_name: str = None
        quantity:int
    
        class Config:
            orm_mode = True    
class OrderItemResponse(BaseModel):
        id: int
        name: str
        price: float
        description: str
        image_url: str
        restaurant_name: str
        quantity:int

        class Config:
            orm_mode = True    
class OrderResponse(BaseModel):
        id: int
        order_id: int
        total: float
        timestamp: datetime
        restaurant_name: str
        items: List[OrderItemResponse]
        

        class Config:
            orm_mode = True
    
class Order(BaseModel):
        items: List[OrderItem]
        total: float
        timestamp: str
        restaurant_name: str  # ✅ Added field
        

class OrderModel(Base):
        __tablename__ = "orders"

        id = Column(Integer, primary_key=True, index=True)
        order_id = Column(Integer, unique=True, index=True)  # Must exist
        total = Column(Float)
        timestamp = Column(DateTime)
        restaurant_name = Column(String)  # Must exist
        
        
    
        items = relationship("OrderItemModel", back_populates="order")
class OrderItemModel(Base):
        __tablename__ = "order_items"
        id = Column(Integer, primary_key=True, index=True)
        # ← reference the PK on orders
        order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
        name = Column(String)
        price = Column(Float)
        description = Column(String)
        image_url = Column(String)
        restaurant_name = Column(String)
        quantity = Column(Integer)
        order = relationship("OrderModel", back_populates="items")        

# — Request Models —
class MenuItemRequest(BaseModel):
    name: str
    description: str
    price: float  # Changed to float to allow decimal prices
    image_url: Optional[str] = None  # Optional URL for the image
    restaurant_name: str  # Added restaurant_name as required field
    
class Message(BaseModel):
    message: str
# Reservation Data Model
class Reservation(BaseModel):
    name: str
    phone: str
    date: str
    time: str
    partySize: int

# — Database Setup (SQLAlchemy) —
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
#DATABASE_URL= "postgresql://postgres:aootkoMsCIGKpgEbGjAjFfilVKEgOShN@switchback.proxy.rlwy.net:24756/railway"

if not DATABASE_URL:
    logging.error("DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL)
#creating a local session
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# — OpenAI Setup (v0.27-style) —
openai_api_key = os.getenv("OPENAI_API_KEY")
#openai_api_key = 123

if not openai_api_key:
    logging.error("OPENAI_API_KEY not set")
    sys.exit(1)

openai.api_key = openai_api_key
#create menu end point


# SMTP configuration (from your scheduling script)
SMTP_HOST       = 'smtp.gmail.com'
SMTP_PORT       = 587
SMTP_USER       = 'proactive1.san@gmail.com'      # from_email
SMTP_PASS       = 'vsjv dmem twvz avhf'           # from_password
MANAGEMENT_EMAIL = 'proactive1@live.com'     # where we send reservations

@app.post("/api/reservationRafisKitchen")
async def make_reservation(reservation: Reservation):
    try:
        # Email content
        subject = "New Reservation Request - Rafi's Kitchen"
        body = f"""
        <h2>New Reservation Details</h2>
        <ul>
            <li><strong>Name:</strong> {reservation.name}</li>
            <li><strong>Phone:</strong> {reservation.phone}</li>
            <li><strong>Date:</strong> {reservation.date}</li>
            <li><strong>Time:</strong> {reservation.time}</li>
            <li><strong>Party Size:</strong> {reservation.partySize}</li>
        </ul>
        """

        # Prepare email message
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = MANAGEMENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        # Send the email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        return { "message": "Reservation email sent successfully." }

    except Exception as e:
        print(f"Error sending email: {e}")
        return { "error": "Failed to send reservation email." }

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
            image_url=itm.image_url,
            restaurant_name=itm.restaurant_name
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
def get_menu_items(
    restaurant_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    print(f"Received request to /get-menu-items/ with restaurant_name = {restaurant_name}")

    q = db.query(MenuItem)

    if restaurant_name:
        print(f"Filtering items where restaurant_name == '{restaurant_name}'")
        q = q.filter(MenuItem.restaurant_name == restaurant_name)
    else:
        print("No restaurant_name provided. Returning all menu items.")
    
    results = q.all()
    print(f"Number of items found: {len(results)}")

    if not results:
        print("No menu items found for the given restaurant_name.")
        raise HTTPException(status_code=404, detail="No menu items found.")
    
    # Print the content of the results
    print("Results content:")
    for item in results:
        print(f"ID: {item.id}, Name: {item.name}, Description: {item.description}, Price: {item.price}, Restaurant: {item.restaurant_name}")

    return results


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

@app.delete("/delete-menu-item/{item_id}/")
async def delete_menu_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(MenuItem).filter(MenuItem.id == item_id).first()

    if not db_item:
        raise HTTPException(status_code=404, detail="Menu item not found")

    db.delete(db_item)
    db.commit()

    return {"message": f"Menu item with ID {item_id} deleted successfully"}

    
@app.put("/update-menu-item/{item_id}/")
def update_menu_item(
    item_id: int,
    updated: MenuItemUpdate,
    db: Session = Depends(get_db),
    restaurant_name: Optional[str] = Query(None),
):
    # first load, ensuring restaurant_name matches:
    item = (
        db.query(MenuItem)
          .filter(MenuItem.id == item_id)
          .filter(MenuItem.restaurant_name == restaurant_name)
          .first()
    )
    if not item:
        raise HTTPException(404, "Menu item not found")

    for field, value in updated.dict().items():
        setattr(item, field, value)
    db.commit()
    db.refresh(item)
    return item


@app.get("/orders", response_model=List[OrderResponse])
def get_orders(restaurant_name: Optional[str] = Query(None), db: Session = Depends(get_db)):
    query = db.query(OrderModel).options(joinedload(OrderModel.items))
    
    # Filter by restaurant_name if provided
    if restaurant_name:
        query = query.filter(OrderModel.restaurant_name == restaurant_name)

    # Execute the query and fetch all orders
    orders = query.all()

    return orders
"""
@app.post("/create_tables")
def create_table(data: CreateTableRequest, db: Session = Depends(get_db)):
    try:
        # Create a new Table instance
        new_table = Table(
            table_number=data.table_number,
            capacity=data.capacity,
            restaurant_name=data.restaurant_name,
            
        )

        db.add(new_table)
        db.commit()
        db.refresh(new_table)

        return {
            "message": "Table created successfully",
            "table_id": new_table.id,
            "table_number": new_table.table_number
        }

    except Exception as e:
        print("Error creating table:", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create table")
"""
@app.post("/place-order")
def place_order(order: Order, db: Session = Depends(get_db)):
    # Get the latest order_id from the database
    last_order = db.query(OrderModel).order_by(desc(OrderModel.order_id)).first()
    if last_order and last_order.order_id:
        new_order_id = last_order.order_id + 1
    else:
        new_order_id = 1  # Starting order_id if table is empty

    # Create the main order record in the database(Order Model is for "orders" table )
    db_order = OrderModel(
        order_id=new_order_id, 
        total=order.total,
        timestamp=order.timestamp,
        restaurant_name=order.restaurant_name,
        
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)

    # Add each order item related to this order (OrderItemModel is for order_items table)
    for item in order.items:
        db_item = OrderItemModel(
            order_id=db_order.id,  # Foreign key to OrderModel.id
            name=item.name,
            price=item.price,
            description=item.description,
            image_url=item.image_url,
            restaurant_name=item.restaurant_name,
            quantity=item.quantity
        )
        db.add(db_item)

    db.commit()

    return {"message": "Order received", "order_id": db_order.id}


@app.delete("/delete-orders/{order_id}")
def delete_order(order_id: int, db: Session = Depends(get_db)):
    order = db.query(OrderModel).filter(OrderModel.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    # Delete order items first
    db.query(OrderItemModel).filter(OrderItemModel.order_id == order.id).delete()

    # Then delete the order
    db.delete(order)
    db.commit()

    return {"message": f"Order {order_id} deleted successfully"}
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
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    



