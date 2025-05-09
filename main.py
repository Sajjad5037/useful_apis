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
import requests
from twilio.rest import Client

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from email.message import EmailMessage
# Now you can generate a unique order ID
 # Use uuid.uuid4() to generate a unique ID


# — Logging —
logging.basicConfig(level=logging.DEBUG)

# — FastAPI Init & CORS —
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
    "http://127.0.0.1:3000",           # Local React dev server (127.0.0.1)
    "http://localhost:3000",           # Local React dev server (localhost)
    "https://rafis-kitchen.vercel.app",
    "https://sajjadalinoor.vercel.app",
    "https://clinic-management-system-27d11.web.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()

class OrderItemHajvery(BaseModel):
    id: str
    name: str
    quantity: float
    price: float
    
class OrderDataHajvery(BaseModel):
    customerInfo: str
    cart: List[OrderItemHajvery]
    totalAmount: float

class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Integer)
    image_url = Column(String)
    restaurant_name = Column(String, index=True)
    dish_type = Column(String, index=True)  # ✅ Clean and clear        

class MenuItemUpdate(BaseModel):
    name: str
    description: str
    price: float
    image_url: Optional[str] = ""
    restaurant_name: str
    dish_type: str    
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
class ItemPizzapoint(BaseModel):
    name: str
    quantity: int
    price: float
class OrderDataPizzaPoint(BaseModel):
    items: List[ItemPizzapoint]
    total: float
    timestamp: str
    restaurant_name: str
    phone: str

class OrderResponse(BaseModel):
        id: int
        order_id: int
        total: float
        timestamp: datetime
        restaurant_name: str
        items: List[OrderItemResponse]
        phone: str 

        class Config:
            orm_mode = True
    
class Order(BaseModel):
        items: List[OrderItem]
        total: float
        timestamp: str
        restaurant_name: str  # ✅ Added field
        phone: str     # new field
        

class OrderModel(Base):
        __tablename__ = "orders"

        id = Column(Integer, primary_key=True, index=True)
        order_id = Column(Integer, unique=True, index=True)  # Must exist
        total = Column(Float)
        timestamp = Column(DateTime)
        restaurant_name = Column(String)  # Must exist
        phone= Column(String, nullable=False)
        
    
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
        phone= Column(String, nullable=False)
        order = relationship("OrderModel", back_populates="items")        

# — Request Models —
class MenuItemRequest(BaseModel):
    name: str
    description: str
    price: float  # Changed to float to allow decimal prices
    image_url: Optional[str] = None  # Optional URL for the image
    restaurant_name: str  # Added restaurant_name as required field
    dish_type: str

    
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
#for railway deployment
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
account_sid = os.getenv("account_sid") 
auth_token= os.getenv("auth_token")
twilio_number=os.getenv("twilio_number")

#for local deploymnet
#DATABASE_URL= "postgresql://postgres:aootkoMsCIGKpgEbGjAjFfilVKEgOShN@switchback.proxy.rlwy.net:24756/railway"
# Twilio credentials (use environment variables in production)
pizzapoint_number="whatsapp:+923004112884"
hajvery_number="whatsapp:+923004112884"
# Initialize Twilio client
client = Client(account_sid, auth_token)

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

def is_relevant_to_programming(message: str) -> bool:
    programming_keywords = [
        "python", "odoo", "data", "automation", "excel", "backend", "frontend",
        "web scraping", "api", "openai", "fastapi", "react", "database", "freelance",
        "deployment", "clinic", "management system", "project", "code", "script"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in programming_keywords)

def is_relevant_to_rafis_kitchen(message: str) -> bool:
    keywords = [
    "rafi", "rafis kitchen", "restaurant", "olean", "wayne street", "800 wayne", "location", "address",
    "contact", "phone", "call", "hours", "timing", "open", "close", "weekends", "seasonal", "may to december",
    "menu", "full menu", "appetizers", "soups", "salads", "pita", "kids menu", "kids meals", "lunch", "dinner",
    "entrees", "pastas", "specials", "popular dishes", "chicken curry", "beef tikka", "lobster mac and cheese",
    "cuisine", "mediterranean", "italian", "lebanese", "pakistani", "food", "dish", "meal",
    "vegetarian", "vegan", "gluten free", "halal", "kosher", "nut free", "dairy free", "customized dishes",
    "takeout", "pickup", "order", "reservation", "book", "table", "private event", "group reservation",
    "parking", "free parking", "family friendly", "pet", "service dogs", "outdoor seating", "covered patio",
    "deck", "wheelchair accessible", "wifi", "drinks", "cocktails", "beer", "wine", "price", "cost", "payment",
    "cash", "card", "casual dining", "dress code", "chef", "amir"
]


    return any(word.lower() in message.lower() for word in keywords)

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

@app.post("/api/send-email-rafis-kitchen")
async def send_email(
    name: str = Form(...),
    email: str = Form(...),
    message: str = Form(...)
):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'New Contact Form Submission'
        msg['From'] = SMTP_USER
        msg['To'] = MANAGEMENT_EMAIL
        msg.set_content(f"""
        New message from contact form:

        Name: {name}
        Email: {email}

        Message:
        {message}
        """)

        # Send the email using TLS
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        return {"success": True, "message": "Email sent successfully."}
    except Exception as e:
        return {"success": False, "message": f"Failed to send email: {str(e)}"}
    
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
            restaurant_name=itm.restaurant_name,
            dish_type=itm.dish_type
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        created_ids.append(db_item.id)

    return {
        "message": "Menu items created successfully",
        "ids": created_ids
    }
@app.post("/api/sendorder_pizzapoint")
async def send_order_pizzapoint(order: OrderDataPizzaPoint):
    try:
        # Build WhatsApp message
        lines = [
            "*🍕 New Pizza Point Order*",
            f"🏬 Restaurant: {order.restaurant_name}",
            f"📞 Customer Phone: {order.phone}",
            f"🕒 Timestamp: {order.timestamp}",
            "",
            "*🛒 Order Items:*"
        ]

        for item in order.items:
            lines.append(
                f"- {item.name} — {item.quantity} × Rs.{item.price:.0f} = Rs.{item.quantity * item.price:.0f}"
            )

        lines.append("")
        lines.append(f"*💰 Total: Rs.{order.total:.0f}*")

        message_body = "\n".join(lines)

        # Send the WhatsApp message
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=pizzapoint_number
        )

        return {"success": True, "message": "Order sent via WhatsApp", "sid": message.sid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send WhatsApp message: {str(e)}")
    
@app.post("/api/sendorder_hajvery")
async def send_order_hajvery(order: OrderDataHajvery):
    try:
        # Build the WhatsApp message content
        lines = [
            "*📦 New Hajvery Milk Shop Order*",
            f"🏠 Address / Vehicle: {order.customerInfo}",
            "",
            "*🛒 Cart Items:*"
        ]

        for item in order.cart:
            lines.append(f"- {item.name} — {item.quantity} × {item.price:.2f} = {(item.quantity * item.price):.2f}")

        lines.append("")
        lines.append(f"*💰 Total: {order.totalAmount:.2f}*")

        message_body = "\n".join(lines)

        # Send the message
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=hajvery_number
        )

        return {"success": True, "message": "Order sent via WhatsApp", "sid": message.sid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send WhatsApp message: {str(e)}")
        
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
        phone=order.phone,
        
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
            quantity=item.quantity,
            phone=order.phone
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
        # Optional: You can implement a relevance check if needed for restaurant-related queries
        """
        if not is_relevant_to_rafis_kitchen(msg.message):
            return {
                "reply": (
                    "I'm sorry, but I can only assist with questions related to Rafi's Kitchen at 800 Wayne Street, Olean, NY, "
                    "including menu, location, and services provided by Amir, the owner."
                )
            }
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.2,
            messages = [
    {
        "role": "system",
        "content": (
            "You are an assistant for Rafi's Kitchen, a seasonal, family-friendly restaurant located at 800 Wayne Street, Olean, NY 14760. "
            "The owner is Amir. The restaurant is open from May to December.\n\n"
            "Operating Hours:\n"
            "- Monday–Thursday: 11:00 AM – 8:00 PM\n"
            "- Friday: 11:00 AM – 9:00 PM\n"
            "- Saturday: 12:00 PM – 9:00 PM\n"
            "- Sunday: Closed\n"
            "The restaurant is open on weekends.\n\n"
            "Key Amenities:\n"
            "- Free parking available\n"
            "- Family-friendly\n"
            "- Service dogs permitted on outdoor upper patio only\n"
            "- Wheelchair accessible\n"
            "- No guest Wi-Fi\n"
            "- Casual dining dress code\n\n"
            "Dining Experience:\n"
            "- Indoor seating and the area’s largest covered outdoor patio (seating for up to 100 people)\n"
            "- Reservations helpful but not required\n"
            "- For reservations or large groups, call (716) 790-8100\n"
            "- Recommend reserving 24–48 hours in advance for weekends\n\n"
            "Menu Highlights:\n"
            "- Cuisine: Mediterranean, Italian, Lebanese, Pakistani\n"
            "- Popular dishes: Chicken Curry, Vegetable Tikka, Beef Tikka Masala, Lobster Mac and Cheese\n"
            "- Vegetarian/Vegan options: Vegetable Curry, Vegetable Tikka, Vegetable Samosas, Vegetable Pakora, Mediterranean & Beet Salads\n"
            "- Gluten-free options: Vegetable Pakora (chickpea flour), curries, kabobs, salmon, grouper, and various salads\n"
            "- Halal/Kosher options: Chicken and lamb dishes\n"
            "- Allergy customization: Nut-free and dairy-free options available on request\n"
            "- Kids Menu includes Chicken Tenders, Mac & Cheese, Pita Pizza, Chicken & Rice, Cavatappi with Marinara\n"
            "- Specialty beers, wines, and cocktails available\n\n"
            "Respond only to questions specifically related to Rafi's Kitchen—its menu, services, hours, policies, or location. "
            "Do not answer questions about general topics or unrelated businesses. If a question is unrelated, politely inform the user that you can only help with inquiries about Rafi's Kitchen."
        )
    },
    {
        "role": "user",
        "content": msg.message
    }
]

        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        logging.error(f"chatRK error: {e}")
        raise HTTPException(status_code=500, detail="Sorry, something went wrong.")
    
@app.post("/api/chatwebsite")
async def chat_website(msg: Message):
    try:
        if not is_relevant_to_programming(msg.message):
            return {
                "reply": (
                    "I'm sorry, but I can only assist with programming-related questions or information about Sajjad's skills "
                    "in Python, Odoo, automation, and backend development."
                )
            }

        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are my virtual assistant, designed to professionally interact with potential clients on my behalf. "
                        "I am Sajjad Ali Noor, a full stack developer with a strong command of Python and Odoo, including custom module development and deployment. "
                        "I have studied books like *Automate the Boring Stuff with Python* and worked extensively on real-world projects involving Excel automation, web scraping, and backend development. "
                        "I have also developed a scalable clinic management system and integrated chatbot features for doctor-patient interaction. "
                        "When responding to clients, highlight my expertise in Python, Odoo, automation, and data handling. "
                        "Only accept projects that align with my skill set to ensure I can deliver excellent results. "
                        "If the client wishes to continue the conversation or hire me, kindly provide my email address: proactive1.san@gmail.com. "
                        "Your goal is to help attract meaningful freelance or contract opportunities that suit my background in software development and AI integration."
                    )
                },
                {"role": "user", "content": msg.message},
            ]
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        logging.error(f"chatwebsite error: {e}")
        raise HTTPException(status_code=500, detail="Sorry, something went wrong.")

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
    


