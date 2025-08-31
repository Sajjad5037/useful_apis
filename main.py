from enum import Enum
import docx2txt
import os
import vertexai
import sys
import fitz 
from google.cloud import storage

from vertexai.generative_models import GenerativeModel, Part
from uuid import uuid4
from sqlalchemy.dialects.postgresql import JSONB
from google.cloud.vision_v1 import types as vision_types
from rapidfuzz import fuzz
import joblib
import sympy
from decimal import Decimal
from fastapi.responses import HTMLResponse
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from urllib.parse import quote
import re
from sqlalchemy.sql import text
from google.cloud import vision

import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import boto3
from botocore.exceptions import ClientError

from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from typing import Optional,List,Union
from fastapi import FastAPI, HTTPException, Depends, Form, File, UploadFile, Request, Header,Body

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    Date,
    Time,
    Numeric,
    desc,
    func,
    text,
    Text,
    Enum as SQLEnum  # <-- added this line
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session,relationship,joinedload
import openai  # classic client
from fastapi import Query,Request
import datetime
from sqlalchemy.types import UserDefinedType
from datetime import datetime,date,time
import uvicorn
import uuid  # Add this import at the top of your file
import random
import requests
from dotenv import load_dotenv
from twilio.rest import Client
from io import BytesIO
from botocore.exceptions import ClientError
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from email.message import EmailMessage
from openai import OpenAI
import boto3
from fastapi.responses import JSONResponse
vectorstore = None
total_pdf = 0
import tempfile

MAX_USER_COST = 0.4 #dollars
MODEL_COSTS = {
    "gpt-4o-mini": {"prompt": 0.00000015, "completion": 0.0000006},  # per token
    "text-embedding-3-small": {"embedding": 0.00000002},   # $0.02 / 1M tokens
    "text-embedding-3-large": {"embedding": 0.00000013},   # $0.13 / 1M tokens
}
MODEL_COST_PER_TOKEN = {
    "gpt-4o-mini": {"prompt": 0.000000056, "completion": 0.000000223},
    "text-embedding-3-small": {"embedding": 0.00000002},
    "text-embedding-3-large": {"embedding": 0.00000013},
}



USAGE_LIMIT_INCREASE = 5.0  # dollars
vertexai.init(project="dazzling-tensor-455512-j1", location="us-central1")
vision_model = GenerativeModel("gemini-1.5-flash")

# Suppose you have a baseline cost stored somewhere or passed in, for demo let's hardcode:
BASELINE_COST = 0.0  # Replace this with your actual baseline cost or fetch it from DB/config
bucket_name_anz_way = "sociology_anz_way"
VECTORSTORE_FOLDER_IN_BUCKET = "vectorstore"
qa_chain_anz_way = None


# Gemini / Vision model
vision_model = GenerativeModel("gemini-1.5-flash")
#for text extractor from image 
# Load the JSON string from the environment variable
json_creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
if not json_creds:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not found")

# Write the JSON to a temp file
with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
    f.write(json_creds)
    temp_path = f.name
    
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
# OCR client
ocr_client = vision.ImageAnnotatorClient()


# Set the correct GOOGLE_APPLICATION_CREDENTIALS to point to that file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path

try:
    client_google_vision_api = vision.ImageAnnotatorClient()
    print("[DEBUG] Google Vision API client initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize Vision client: {e}")
    raise


# — Logging —
logging.basicConfig(level=logging.DEBUG)
load_dotenv()
# — FastAPI Init & CORS —
app = FastAPI()
session_texts = {}     # session_id -> full essay text
session_histories = {} # session_id -> list of messages (chat history)
username_for_interactive_session = None



allowed_origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "https://rafis-kitchen.vercel.app",
    "https://sajjadalinoor.vercel.app",
    "https://clinic-management-system-27d11.web.app",
    "https://shah-rukk-website.vercel.app",
    "https://class-management-system-new.web.app",
    "https://ai-social-campaign.vercel.app",
    "https://anz-way.vercel.app",  # ← only the domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

Base = declarative_base()

# AWS S3 config (use Railway ENV variables in deployment)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")




s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
# for solving math problem after exxtracing from images

class StudentUsageRequest(BaseModel):
    student_name: str

class StudentReflectionSchema(BaseModel):
    id: int
    student_id: int
    question_text: str
    preparedness_level: str
    created_at: datetime

    class Config:
        orm_mode = True
class StudentReportRequest(BaseModel):
    student_id: int
    from_date: str
    to_date: str


class PreparednessLevel(Enum):
    well_prepared = "well_prepared"
    partially_prepared = "partially_prepared"
    needs_improvement = "needs_improvement"

class StudentReflection(Base):
    __tablename__ = "student_reflections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, nullable=False) 
    question_text = Column(Text, nullable=False)
    preparedness_level = Column(SQLEnum(PreparednessLevel), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
class SolveResult(BaseModel):
    problem: str
    solution: str
class TextRequest(BaseModel):
    text: str
#for pdf query chatbot
class PageRange(BaseModel):
    start_page: int
    end_page: int
    user_name:str
    
class ChatRequest_CSS(BaseModel):
    session_id: str
    message: str
    first_message: bool = False

class CommonMistake(Base):
    __tablename__ = "mistake_pattern_essay"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    original_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class StartConversationRequest(BaseModel):
    subject: str
    marks: int
    question_text: str
    username: str  # new field for tracking the user

class SendMessageRequest(BaseModel):
    id: int          # Unique identifier, e.g., student or message ID
    session_id: str
    message: str
    username: str  # <-- added username
    
class CostPerInteraction(Base):
    __tablename__ = "cost_per_interaction"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), nullable=False)
    model = Column(String(50), nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    cost_usd = Column(Numeric(10, 6), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_name = Column(String, nullable=False)
    goal = Column(String, nullable=False)
    tone = Column(String, nullable=False)

    doctor_id = Column(Integer, nullable=False)
    doctor_name = Column(String, nullable=False)

    suggestions = Column(String, nullable=False)  # Store JSON array as string


class CampaignSuggestion2(Base):
    __tablename__ = "campaign_suggestions2"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    status = Column(String, default="pending")
    user_id = Column(Integer)  # just a plain column, no FK


    
# ---------- Pydantic MODELS ----------
class CampaignDoctorInfo(BaseModel):
    id: int
    name: str

class CampaignRequest(BaseModel):
    campaignName: str
    goal: str
    tone: str
    doctorData: CampaignDoctorInfo
    
class CampaignResponse(BaseModel):
    campaignId: int
    message: str


class CommonMistakeSchema(BaseModel):
    id: int
    session_id: str
    original_text: str
    corrected_text: str
    category: Optional[str] = None
    explanation: Optional[str] = None
    created_at: datetime  # <- change from str to datetime

    class Config:
        orm_mode = True

class Vector(UserDefinedType):
    def __init__(self, dimension):
        self.dimension = dimension

    def get_col_spec(self):
        return f'vector({self.dimension})'

    def bind_expression(self, bindvalue):
        return bindvalue

    def column_expression(self, col):
        return col


#for database query
class ChatRequest(BaseModel):
    message: str
class FAQOut(BaseModel):
    question: str
    answer: str

    class Config:
        orm_mode = True
#for database query
# Response model (optional)
class ChatResponse(BaseModel):
    reply: str
class ChatRequest_interactive_pdf(BaseModel):
    message: str
    user_name: str  # <-- added this

from sqlalchemy import Column

class FAQModel(Base):
    __tablename__ = "faqs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    embedding = Column(Vector(1536)) 

class SalesRequest(BaseModel):
    start_date: date
    end_date: date

class ReservationBase(BaseModel):
    table_id: int
    customer_name: str
    customer_contact: str | None = None  # optional field
    date: date
    time_slot: str
    status: str

    class Config:
        orm_mode = True

class ParaphraseRequest(BaseModel):
    original: str
    paraphrase: str
    level: str = "O-Level"  # default if not provided

class ReservationModel_new(Base):
    __tablename__ = "reservations_new"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    table_id = Column(Integer, nullable=False)
    customer_name = Column(String, nullable=False)
    customer_contact = Column(String, nullable=True)
    date = Column(Date, nullable=False)
    time_slot = Column(String, nullable=False)  # If time format is string, else use Time
    status = Column(String, nullable=False)

class SalePizzaPoint(Base):
    __tablename__ = "sales_pizza_point"

    id = Column(Integer, primary_key=True, index=True)
    bill_number = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    total = Column(Float, nullable=False)

class SalesFilterRequest(BaseModel):
    start_date: date
    end_date: date
#for sales for pizza point like restaurants
class SalesResponse(BaseModel):
    bill_number: str
    date: date
    total: float

    class Config:
        orm_mode = True  # Tells Pydantic to treat this as a model


class PizzaOrder(Base):
    __tablename__ = "pizza_order"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)
    items = Column(String, nullable=False)  # JSON string or flat string
    image_url = Column(String)
    total = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PizzaOrderItemResponse(BaseModel):
    name: str
    quantity: int
    price: float
    image_url: Optional[str]  # ✅ allow image_url to be None

    class Config:
        orm_mode = True

class OrderResponsePizzaOrder(BaseModel):
    id: int
    restaurant_name: str
    phone: str
    timestamp: datetime
    items: List[PizzaOrderItemResponse]
    total: float

    class Config:
        orm_mode = True

class OrderItemHajvery(BaseModel):
    id: str
    name: str
    quantity: float
    price: float
    
class OrderDataHajvery(BaseModel):
    customerInfo: str
    cart: List[OrderItemHajvery]
    totalAmount: float
    vendorName: str  # ✅ Added this to match the payload

class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Integer)
    image_url = Column(String)
    restaurant_name = Column(String, index=True)
    dish_type = Column(String, index=True)  # ✅ Clean and clear    
        
class RailwayUsage(Base):
    __tablename__ = "railway_usage"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)               # Date of usage
    vendor_name = Column(String, index=True)      # Name of the vendor
    api_calls = Column(Integer)                   # Number of API calls

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
    id: int
    name: str
    price: float
    description: str
    image_url: str
    restaurant_name: str
    quantity: int

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
DATABASE_URLAWS="postgresql://postgres:shaazZ121024@database-1.ch0wcs62mtif.eu-north-1.rds.amazonaws.com:5432/postgres"

                
hajvery_number = "whatsapp:+923004112884"  
pizzapoint_number="whatsapp:+923004112884"
hajvery_number="whatsapp:+923004112884"
# Initialize Twilio client
client_twilio = Client(account_sid, auth_token)
#inititailize google vision api
client_google_vision_api = vision.ImageAnnotatorClient()


if not DATABASE_URL:
    logging.error("DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL)
engineAWS = create_engine(DATABASE_URLAWS)
#creating a local session
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
SessionAWS = sessionmaker(bind=engineAWS, autoflush=False, autocommit=False)
MODEL_COST = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4o": {"input": 0.005, "output": 0.015},  # example
    "gpt-4o-mini": {"input": 0.003, "output": 0.01},  # 
    "text-embedding-ada-002": {"input": 0.0004, "output": 0.0004},  # Added embedding model cost
}
Base.metadata.create_all(bind=engine)
#Base.metadata.create_all(bind=engineAWS)

# — OpenAI Setup (v0.27-style) —
#openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

if not openai_api_key:
    logging.error("OPENAI_API_KEY not set")
    sys.exit(1)

openai.api_key = openai_api_key
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the total cost for a model call, multiplying the final cost by 3
    before returning (e.g., to account for pricing adjustments or markups).
    """
    try:
        print(f"[DEBUG] Calculating cost for model: {model}")
        print(f"[DEBUG] Prompt tokens: {prompt_tokens}")
        print(f"[DEBUG] Completion tokens: {completion_tokens}")

        # Retrieve cost rates
        input_cost_per_1k = MODEL_COST[model]["input"]
        output_cost_per_1k = MODEL_COST[model]["output"]

        print(f"[DEBUG] Input cost per 1K tokens: {input_cost_per_1k}")
        print(f"[DEBUG] Output cost per 1K tokens: {output_cost_per_1k}")

        # Calculate costs
        prompt_cost = (prompt_tokens / 1000) * input_cost_per_1k
        completion_cost = (completion_tokens / 1000) * output_cost_per_1k

        print(f"[DEBUG] Prompt cost: {prompt_cost}")
        print(f"[DEBUG] Completion cost: {completion_cost}")

        total_cost = prompt_cost + completion_cost
        print(f"[DEBUG] Total cost before rounding and multiplier: {total_cost}")

        # Multiply total cost by 3
        total_cost *= 3
        print(f"[DEBUG] Total cost after multiplying by 3: {total_cost}")

        total_cost_rounded = round(total_cost, 6)
        print(f"[DEBUG] Total cost after rounding: {total_cost_rounded}")

        return total_cost_rounded

    except KeyError as e:
        print(f"[ERROR] Model key not found in MODEL_COST: {e}")
        return 0.0

    except TypeError as e:
        print(f"[ERROR] Invalid type for token values: {e}")
        return 0.0

    except Exception as e:
        print(f"[ERROR] Unexpected error in calculate_cost: {e}")
        return 0.0


def sanitize_filename(filename: str) -> str:
    # Trim spaces
    filename = filename.strip()
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    # Remove any characters except word chars, dots, hyphens, and underscores
    filename = re.sub(r'[^\w\.-]', '', filename)
    return filename
def get_db_aws():
    db = SessionAWS()
    try:
        yield db
    finally:
        db.close()

def is_relevant_to_css_preparation(message: str, threshold: int = 85) -> bool:
    """
    Returns True if the message is likely about CSS preparation,
    using fuzzy matching to handle minor typos.
    """
    css_keywords = [
        "css exam", "web site" ,"competitive exam", "fpsc", "optional subjects", "compulsory subjects",
        "essay", "precis", "past papers", "syllabus", "interview", "psychological test",
        "strategy", "notes", "study plan", "guidance", "tips", "mcqs", "current affairs",
        "pak affairs", "islamiat", "english", "mentor", "coaching", "shah rukh",
        "video lectures", "subscription", "resources", "join", "how to start", "mentorship",
        "prepare", "roadmap", "study schedule", "preparation", "course", "material",
        "feedback", "evaluation", "essay checker", "review", "marking", "exam strategy"
    ]

    message_lower = message.lower()
    return any(fuzz.partial_ratio(keyword, message_lower) >= threshold for keyword in css_keywords)
    
def is_relevant_to_programming(message: str) -> bool:
    programming_keywords = [
        "python", "odoo", "data", "automation", "excel", "backend", "frontend",
        "web scraping", "api", "openai", "fastapi", "react", "database", "freelance",
        "deployment", "clinic","sajjad", "management system","contact","information", "project","he", "code", "script"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in programming_keywords)

def separate_sentences(text):
    # Split by newline characters and filter out any empty strings
    sentences = [sentence for sentence in text.split("\n") if sentence.strip()]
    return sentences


#meant to clean the relevant context that was used by the model to answer the user query
def clean(text):
    # This pattern finds a lowercase letter or digit followed by a space and then a capital letter
    # It inserts a period before the capital letter
    cleaned_text = re.sub(r'([a-z0-9])\s+([A-Z])', r'\1. \2', text)
    return cleaned_text

def delete_previous_pdf_in_aws():
    try:
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        # List objects in 'upload/' folder
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="upload/")
        if "Contents" not in response:
            print("No files found in 'upload/' folder.")
            return

        pdf_keys = [
            obj["Key"] for obj in response["Contents"]
            if obj["Key"].lower().endswith(".pdf") and not obj["Key"].endswith("/")
        ]

        if not pdf_keys:
            print("No PDFs to delete in 'upload/' folder.")
            return

        # Batch delete
        delete_payload = {
            "Objects": [{"Key": key} for key in pdf_keys]
        }
        delete_response = s3.delete_objects(Bucket=BUCKET_NAME, Delete=delete_payload)

        deleted = delete_response.get("Deleted", [])
        print(f"Deleted {len(deleted)} PDF(s):")
        for item in deleted:
            print(f" - {item['Key']}")

    except ClientError as e:
        print(f"AWS Client Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def extract_relevant_context(relevant_text, query, num_sentences=2):
    """
    Refines the relevant context by selecting the top N sentences
    that are most similar to the query.
    """
    query_embedding = get_embedding(query, model="text-embedding-ada-002")

    # Split text into sentences based on multiple delimiters
    sentences = re.split(r'[.?!]', relevant_text)

    cleaned_sentences = []
    sentence_embeddings = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Ignore empty sentences
            # Remove non-ASCII characters
            sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
            cleaned_sentences.append(sentence)

            # Generate embedding for the sentence
            try:
                embedding = get_embedding(sentence, model="text-embedding-ada-002")
                sentence_embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for sentence: {sentence}. Skipping.")
                continue  # Skip problematic sentences

    if not sentence_embeddings:
        return "No relevant context could be extracted."

    # Convert list to NumPy array for efficient similarity computation
    sentence_embeddings = np.array(sentence_embeddings)

    # Compute cosine similarity scores
    similarity_scores = cosine_similarity([query_embedding], sentence_embeddings)[0]

    # Pair sentences with their similarity scores
    scored_sentences = list(zip(cleaned_sentences, similarity_scores))

    # Sort sentences by similarity score in descending order
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Select the top N most relevant sentences
    top_relevant_sentences = [sentence for sentence, _ in scored_sentences[:num_sentences]]

    # Format final output
    final_context = '\n'.join([f"{sentence}" for sentence in top_relevant_sentences])


    return final_context


def create_qa_chain(vectorstore, question, db: Session, username, openai_api_key: str, top_n=5, fetch_k=10):
    client = openai.OpenAI(api_key=openai_api_key)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant. Use ONLY the following retrieved context to answer the user's query.
Do NOT use any information outside this context.
If the answer is not contained within the provided context, respond honestly by saying you don't know.
Additionally, suggest that the user try asking a more general-purpose language model like ChatGPT for questions beyond this context.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_n, "fetch_k": fetch_k}
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    total_tokens = 0
    document_texts = []
    document_metadata = []
    unique_pdf_names = set()

    try:
        search_results = retriever.get_relevant_documents(question)
        if not search_results:
            raise ValueError("No documents found for the given query.")

        for doc in search_results:
            text = doc.page_content
            metadata = doc.metadata
            pdf_name = metadata.get("pdf_name", "Unknown PDF")
            page_number = metadata.get("page_number", "Unknown Page")
            document_texts.append(text)
            document_metadata.append((pdf_name, page_number))
            unique_pdf_names.add(pdf_name)

        document_embeddings = []
        for text in document_texts:
            try:
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                document_embeddings.append(embedding)

                usage = getattr(response, "usage", None)
                if usage:
                    total_tokens += usage.total_tokens
            except openai.OpenAI.OpenAIError as e:
                print(f"Error generating embedding for document: {e}")
                document_embeddings.append(None)

        # Query embedding
        try:
            response = client.embeddings.create(
                input=question,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding
            usage = getattr(response, "usage", None)
            if usage:
                total_tokens += usage.total_tokens
        except openai.OpenAI.OpenAIError as e:
            print(f"Error generating embedding for query: {e}")
            query_embedding = None

        if query_embedding is None or any(e is None for e in document_embeddings):
            raise ValueError("Error in generating embeddings.")

        similarities = cosine_similarity([query_embedding], document_embeddings)[0]
        sorted_indices = similarities.argsort()[::-1]

        relevant_texts = [
            extract_relevant_context(document_texts[sorted_indices[0]], question)
        ]
        relevant_metadata = [document_metadata[sorted_indices[0]]]
        merged_relevant_text = "\n\n".join(relevant_texts)

        # Store cost
        cost = calculate_cost("text-embedding-ada-002", total_tokens, 0)
        cost_record = CostPerInteraction(
            username=username,
            model="text-embedding-ada-002",
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
            cost_usd=cost,
            created_at=datetime.utcnow()
        )
        try:
            db.add(cost_record)
            db.commit()
            print("[INFO] Cost record stored.")
        except SQLAlchemyError as e:
            db.rollback()
            print(f"[ERROR] Failed to store cost record: {e}")

        # QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

        return qa_chain, merged_relevant_text, relevant_metadata

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None, None
        
"""
old function : 
# def create_qa_chain(vectorstore, question, top_n=5):
#     client = openai.OpenAI(api_key=openai_api_key)
#
#     # Define a prompt template
#     prompt = PromptTemplate(
#         template="""
#     You are a helpful assistant. Use ONLY the following retrieved context to answer the user's query.
#     Do NOT use any information outside this context.
#     If the answer is not contained within the provided context, respond honestly by saying you don't know.
#     Additionally, suggest that the user try asking a more general-purpose language model like ChatGPT for questions beyond this context.
#
#     Context:
#     {context}
#
#     Question:
#     {question}
#
#     Answer:
#     """,
#         input_variables=["context", "question"]
#     )
#
#     # Set up the retrieval-augmented QA chain
#     retriever = vectorstore.as_retriever(search_kwargs={"k": top_n})
#     llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
#
#     try:
#         # Retrieve the top N relevant documents
#         search_results = retriever.get_relevant_documents(question)
#
#         if not search_results:
#             raise ValueError("No documents found for the given query.")
#
#         # Extract document texts, filenames, and page numbers
#         document_texts = []
#         document_metadata = []
#         unique_pdf_names = set()
#
#         for doc in search_results:
#             text = doc.page_content
#             metadata = doc.metadata
#             pdf_name = metadata.get("pdf_name", "Unknown PDF")
#             page_number = metadata.get("page_number", "Unknown Page")
#
#             document_texts.append(text)
#             document_metadata.append((pdf_name, page_number))
#             unique_pdf_names.add(pdf_name)
#
#         # Determine the number of unique PDFs to extract context from
#         num_relevant_docs = len(unique_pdf_names)
#
#         # Generate embeddings for the documents
#         document_embeddings = []
#         for text in document_texts:
#             try:
#                 response = client.embeddings.create(
#                     input=text,
#                     model="text-embedding-ada-002"
#                 )
#                 embedding = response.data[0].embedding
#                 document_embeddings.append(embedding)
#             except openai.OpenAI.OpenAIError as e:
#                 print(f"Error generating embedding for document: {e}")
#                 document_embeddings.append(None)
#
#         # Generate the embedding for the question
#         try:
#             response = client.embeddings.create(
#                 input=question,
#                 model="text-embedding-ada-002"
#             )
#             query_embedding = response.data[0].embedding
#         except openai.OpenAI.OpenAIError as e:
#             print(f"Error generating embedding for query: {e}")
#             query_embedding = None
#
#         # Ensure embeddings were successfully generated
#         if query_embedding is None or any(embedding is None for embedding in document_embeddings):
#             raise ValueError("Error in generating embeddings. Please check the API responses.")
#
#         # Calculate cosine similarity
#         similarities = cosine_similarity([query_embedding], document_embeddings)[0]
#
#         # Find indices of the top `num_relevant_docs` documents
#         #sorted_indices = similarities.argsort()[::-1]
#
#         sorted_indices = similarities.argsort()[::-1]
#
#         # Extract relevant text and metadata from multiple PDFs
#         #relevant_texts = [extract_relevant_context(document_texts[i], question) for i in sorted_indices]
#         relevant_texts = [extract_relevant_context(document_texts[sorted_indices[0]], question)]
#
#         most_relevant_pdf = sorted_indices[0]
#         relevant_metadata = [document_metadata[most_relevant_pdf]]
#
#         # Merge the contexts from multiple PDFs
#         merged_relevant_text = "\n\n".join(relevant_texts)
#
#         # Create QA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff",
#             chain_type_kwargs={"prompt": prompt},
#         )
#
#         return qa_chain, merged_relevant_text, relevant_metadata
#
#     except ValueError as ve:
#         print(f"ValueError: {ve}")
#         return None, None, None
#
#     except OpenAI.OpenAIError as e:
#         print(f"OpenAI API Error: {e}")
#         return None, None, None
#
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None, None, None

def create_or_load_vectorstore(
    pdf_text,
    username,
    openai_api_key,
    s3_client,
    bucket_name,
    db: Session,    
    vectorstore_key="vectorstore.faiss",
    embeddings_key="embeddings.pkl"
):
    
    try:
        print("[INFO] Starting creation of new vector store and embeddings...")
        global vectorstore

        print("[INFO] Initializing RecursiveCharacterTextSplitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # larger chunks to keep instructions together
            chunk_overlap=100,   # overlap to prevent cutting important lines
            separators=["\n\n", "\n", ".", " "]
        )
        print("[INFO] Text splitter initialized.")

        documents_with_page_info = []
        print("[INFO] Splitting text into chunks and creating Document objects...")
        for pdf_name, page_content in pdf_text.items():
            for page_number, text in page_content.items():
                if text.strip():
                    chunks = text_splitter.split_text(text)
                    print(f"[DEBUG] PDF '{pdf_name}', Page {page_number}: Split into {len(chunks)} chunks.")
                    for chunk in chunks:
                        document_with_page_info = Document(
                            page_content=chunk,
                            metadata={"pdf_name": pdf_name, "page_number": page_number}
                        )
                        documents_with_page_info.append(document_with_page_info)
        print(f"[INFO] Created {len(documents_with_page_info)} document chunks in total.")

        print("[INFO] Initializing OpenAI client...")
        client = OpenAI(api_key=openai_api_key)
        print("[INFO] OpenAI client initialized.")

        # Extract texts for embeddings
        texts = [doc.page_content for doc in documents_with_page_info]

        print("[INFO] Requesting embeddings from OpenAI API...")
        openai_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        print("[INFO] Embeddings received from OpenAI API.")

        # Extract embeddings vectors
        embeddings_list = [data.embedding for data in openai_response.data]

        # Wrap embeddings in a LangChain-compatible Embeddings class
        # Here, we create a simple class that returns our precomputed embeddings
        class PrecomputedEmbeddings(Embeddings):
            def __init__(self, precomputed):
                self.precomputed = precomputed
                self.index = 0  # to track which embedding to return

            def embed_documents(self, texts):
                result = self.precomputed[self.index:self.index + len(texts)]
                self.index += len(texts)
                return result

            def embed_query(self, text):
                resp = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return resp.data[0].embedding
        embeddings = PrecomputedEmbeddings(precomputed=embeddings_list)

        print("[INFO] Creating FAISS vector store from documents and embeddings...")
        vectorstore = FAISS.from_documents(documents_with_page_info, embeddings)
        print("[INFO] Vector store created.")

        print(f"[INFO] Saving vector store locally to '{vectorstore_key}'...")
        vectorstore.save_local(vectorstore_key)
        print("[INFO] Vector store saved locally.")

        embeddings_params = {openai_api_key: openai_api_key}
        print(f"[INFO] Saving embeddings parameters locally to '{embeddings_key}'...")
        joblib.dump(embeddings_params, embeddings_key)
        print("[INFO] Embeddings parameters saved.")

        # --- Estimate and store cost per interaction ---
        usage = getattr(openai_response, "usage", None)
        usage = getattr(openai_response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

            cost = calculate_cost("text-embedding-ada-002", prompt_tokens, completion_tokens)

            cost_record = CostPerInteraction(
                username=username,
                model="text-embedding-ada-002",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                created_at=datetime.utcnow()
            )

            try:
                db.add(cost_record)
                db.commit()
                print("[INFO] Cost record stored in database.")
            except SQLAlchemyError as e:
                db.rollback()
                print(f"[ERROR] Failed to store cost info: {e}")
        else:
            print("[WARN] Could not retrieve token usage info from OpenAI API.")

        print("[SUCCESS] Vector store and embeddings creation complete.")
        return vectorstore, embeddings

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
def insert_explicit_multiplication(expr_str):
    # Insert * between digit and variable separated by space: '3 xy' -> '3*xy'
    expr_str = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', expr_str)
    # Insert * between variable and variable separated by space: 'x y' -> 'x*y'
    expr_str = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1*\2', expr_str)
    return expr_str

def fix_implicit_multiplication(expr_str: str) -> str:
    # Insert * between number and variable (e.g. 2x -> 2*x)
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    # Insert * between variable and variable (e.g. xy -> x*y)
    expr_str = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr_str)
    # Insert * between number or variable and '(' (e.g. 2(x+1) -> 2*(x+1), x(x+1) -> x*(x+1))
    expr_str = re.sub(r'(\d|\w)\(', r'\1*(', expr_str)
    return expr_str

def clean_extracted_text(text):
    if text is None:
        return ""

    # Replace newline characters with space
    cleaned_text = text.replace('\n', ' ')

    # Insert space between lowercase and uppercase transitions (e.g., "smallText" → "small Text")
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)

    # Normalize fancy quotes
    cleaned_text = cleaned_text.replace('“', '"').replace('”', '"')

    # Add separator after colons before capitalized list items
    cleaned_text = re.sub(r': ([A-Z])', r':. \1', cleaned_text)

    # Remove common unwanted unicode characters
    unwanted_chars = ['\uf0b7', '\u00a0', '\ufeff']
    for char in unwanted_chars:
        cleaned_text = cleaned_text.replace(char, ' ')

    # Remove known repeated noise (site names, page numbers)
    cleaned_text = re.sub(r'www\.\S+\.com', '', cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r'Page\s*\d+', '', cleaned_text)     # Remove 'Page 4', 'Page 10', etc.

    # Collapse multiple spaces into one
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def extract_text_from_pdf(pdf_input, start_page=1, end_page=1, target_page=None):
    pdf_text = {}

    try:
        # Determine if input is a file path (string) or a BytesIO stream
        if isinstance(pdf_input, str):
            pdf_name = pdf_input.split('/')[-1]
            doc = fitz.open(pdf_input)
        else:
            # It's a file-like object (e.g., BytesIO), used when reading from S3
            pdf_name = "from_stream.pdf"
            doc = fitz.open(stream=pdf_input.read(), filetype="pdf")

        total_pages = len(doc)
        pdf_text = {pdf_name: {}}

        if target_page:
            if 1 <= target_page <= total_pages:
                page = doc.load_page(target_page - 1)
                text = clean_extracted_text(page.get_text())
                pdf_text[pdf_name][target_page] = text
            else:
                print(f"Page {target_page} is out of range. This PDF has {total_pages} pages.")
        else:
            # Fallback to last page if end_page is None or out of range
            if end_page is None or end_page > total_pages:
                end_page = total_pages
            if start_page < 1:
                start_page = 1

            for page_number in range(start_page - 1, end_page):  # 0-based index
                try:
                    page = doc.load_page(page_number)
                    page_text = clean_extracted_text(page.get_text())
                    print(page_text)
                    pdf_text[pdf_name][page_number + 1] = page_text  # Keep keys 1-based
                except IndexError:
                    print(f"IndexError: Page {page_number + 1} not found.")
                    break
                except Exception as e:
                    print(f"Error reading page {page_number + 1}: {e}")
                    continue
        
        return pdf_text

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return {}
        
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    text = text.replace("\n", " ")  # Clean text
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding
    
def is_relevant_to_rafis_kitchen(message: str) -> bool:
    keywords = [
        "rafi", "rafis kitchen", "restaurant", "olean", "wayne street", "800 wayne", "location", "address",
        "contact", "phone", "call", "opening hours", "timing", "open", "close", "menu", "food", "cuisine",
        "order", "takeout", "delivery", "pickup", "reservation", "book", "dish", "meal", "special", "chef",
        "owner", "amir", "drinks", "vegetarian", "non-veg", "halal", "dessert", "starter", "appetizer",
        "lunch", "dinner", "breakfast", "cost", "price", "payment", "card", "cash", "service", "facilities"
    ]

    message = message.lower()
    message = re.sub(r"[^\w\s]", " ", message)  # remove punctuation

    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, message):
            return True
    return False

#create menu end point


# SMTP configuration (from your scheduling script)
SMTP_HOST       = 'smtp.gmail.com'
SMTP_PORT       = 587
SMTP_USER       = 'proactive1.san@gmail.com'      # from_email
SMTP_PASS       = 'vsjv dmem twvz avhf'           # from_password
MANAGEMENT_EMAIL = 'proactive1@live.com'     # where we send reservations

#retrieving common mistake patterns from the database for css essay checker
# Token auth dependency with debugging
def get_token(request: Request, authorization: Optional[str] = Header(None)):
    print("=== get_token called ===")
    print("Request method:", request.method)
    print("Authorization header:", authorization)

    # Skip auth for CORS preflight
    if request.method == "OPTIONS":
        print("CORS preflight request, skipping token check")
        return None

    if not authorization or not authorization.startswith("Bearer "):
        print("Authorization missing or invalid")
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ")[1]
    print("Extracted token:", token)
    return token

@app.get("/css-common-mistakes", response_model=List[CommonMistakeSchema])
def get_common_mistakes(
    username: str = Query(..., description="Doctor username to filter results"),  # frontend sends doctor name here
    token: Optional[str] = Depends(get_token),
    db: Session = Depends(get_db)
):
    print("=== get_common_mistakes called ===")
    print("Token received in endpoint:", token)
    print("Username received:", username)

    try:
        mistakes = (
            db.query(CommonMistake)
            .filter(CommonMistake.session_id == username)  # filter by session_id
            .order_by(CommonMistake.created_at.desc())
            .all()
        )
        print(f"Fetched {len(mistakes)} mistakes for username '{username}'")
        return mistakes
    except Exception as e:
        print("Error fetching common mistakes:", e)
        raise HTTPException(status_code=500, detail="Internal server error")

"""
previous css mistake
@app.get("/css-common-mistakes", response_model=List[CommonMistakeSchema])
def get_common_mistakes(
    token: Optional[str] = Depends(get_token),
    db: Session = Depends(get_db)
):
    print("=== get_common_mistakes called ===")
    print("Token received in endpoint:", token)
    try:
        mistakes = db.query(CommonMistake).order_by(CommonMistake.created_at.desc()).all()
        print(f"Fetched {len(mistakes)} mistakes from database")
        return mistakes
    except Exception as e:
        print("Error fetching common mistakes:", e)
        raise HTTPException(status_code=500, detail="Internal server error")    
        """
#start
# In-memory session store
sessions = {}

@app.post("/process-assignment")
async def process_assignment(file: UploadFile = File(...)):
    print("=== /process-assignment called ===")
    
    try:
        # Validate file type
        if not file.filename.endswith(".docx"):
            print("Invalid file type")
            return JSONResponse(content={"error": "Please upload a .docx file"}, status_code=400)

        # Read file contents
        contents = await file.read()
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as f:
            f.write(contents)
        print(f"File saved as {temp_filename}, size: {len(contents)} bytes")

        # Extract text using docx2txt
        combined_text = docx2txt.process(temp_filename)
        print("Extracted text (first 300 chars):")
        print(combined_text[:300] + "..." if len(combined_text) > 300 else combined_text)

        # Generate a session ID for the student
        session_id = str(uuid.uuid4())
        print(f"Generated session_id: {session_id}")

        # Create prompt for OpenAI
        assignment_prompt = f"""
You are a teacher-assistant bot interacting with a student about their MPhil linguistics assignment. 
Your goal is to help the student reflect on their work, understand it deeply, and estimate whether their responses suggest genuine authorship.

Guidelines:
1. Begin the conversation with a friendly greeting and an initial question about the assignment.
2. Ask follow-up questions based on the student's previous answers to assess comprehension.
3. Encourage the student to explain concepts, reasoning, and choices in their work. Guide them to think critically about their assignment.
4. Randomize question phrasing and sentence structure so the student cannot simply copy-paste AI responses.
5. Detect potential copying by noting if the student's answers are generic, overly polished, or unrelated to the assignment content—but do NOT accuse directly.
6. Adjust difficulty or specificity of questions based on the student's responses to probe deeper understanding.
7. Ask a maximum of 5 questions. After the 5th answer, provide a concise evaluation of:
    - The student's understanding of the assignment
    - Indicators of authorship or originality
   Stop asking questions afterward. Do not respond to additional messages.
8. Maintain a polite, academic, conversational, and encouraging tone.
9. Do not give full evaluation before the 5 questions are asked. Focus on interaction and comprehension assessment.

Assignment text:
<<< BEGIN TEXT >>>
{combined_text.strip()}
<<< END TEXT >>>
"""




        print("Sending prompt to OpenAI...")
        sessions[session_id] = [
            {"role": "system", "content": "You are an assistant helping a student discuss their assignment with a teacher."},
            {"role": "user", "content": assignment_prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant helping a student discuss their assignment with a teacher."},
                {"role": "user", "content": assignment_prompt}
            ],
            temperature=0.7
        )

        bot_message = response.choices[0].message.content.strip()
        print("Received response from OpenAI (first 300 chars):")
        print(bot_message[:300] + "..." if len(bot_message) > 300 else bot_message)

        return {"initialMessage": bot_message, "session_id": session_id}

    except Exception as e:
        print("Error processing file or calling OpenAI:", e)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

@app.post("/continue-chat")
async def continue_chat(session_id: str = Form(...), message: str = Form(...)):
    print("=== /continue-chat called ===")
    print(f"Session ID: {session_id}")
    print(f"Student message: {message[:200]}{'...' if len(message) > 200 else ''}")

    try:
        # Validate session
        if session_id not in sessions:
            print("[ERROR] Invalid session ID")
            return JSONResponse(content={"error": "Invalid session ID"}, status_code=400)

        # Append student's message to session history
        sessions[session_id].append({"role": "user", "content": message})

        # Create system prompt for anti-cheating & dynamic question strategy
        system_prompt = """
You are a teacher-assistant bot interacting with a student about their assignment. 

Guidelines:
1. Ask follow-up questions based on previous answers to assess comprehension.
2. Randomize question phrasing to prevent copy-pasting AI responses.
3. Judge whether the student's answers reflect genuine understanding.
4. If the student explicitly requests an evaluation (e.g., types 'evaluate' or 'give me evaluation'), generate a final evaluation of comprehension and potential originality based on all previous responses. Do NOT ask further questions after providing the evaluation.
5. Maintain a polite, academic, and conversational tone throughout.
"""
        # Combine system prompt with session messages
        all_messages = [{"role": "system", "content": system_prompt}] + sessions[session_id]

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=all_messages,
            temperature=0.7
        )

        bot_message = response.choices[0].message.content.strip()
        print(f"Bot response: {bot_message[:300]}{'...' if len(bot_message) > 300 else ''}")

        # Append bot's reply to session history
        sessions[session_id].append({"role": "assistant", "content": bot_message})

        return {"botMessage": bot_message}

    except Exception as e:
        print("[ERROR] Exception in /continue-chat")
        traceback.print_exc()
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)
#end
def download_vectorstore_from_gcs(bucket_name: str, folder_in_bucket: str):
    print("[DEBUG] Starting download of vector store from GCS...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    temp_dir = tempfile.mkdtemp()
    try:
        for filename in ["index.faiss", "index.pkl"]:
            blob = bucket.blob(f"{folder_in_bucket}/{filename}")
            local_path = os.path.join(temp_dir, filename)
            blob.download_to_filename(local_path)
            print(f"[DEBUG] Successfully downloaded {filename} to {local_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download vector store from GCS: {e}")
        traceback.print_exc()
    print("[DEBUG] Vector store download complete")
    return temp_dir

async def evaluate_student_response_from_images(
    images: List[UploadFile],
    question_text: str,
    total_marks: int,
    qa_chain,  # RetrievalQA
    minimum_word_count: int = 80
):
    """
    Extract text from images, retrieve relevant instructions from vector store,
    and evaluate the student's response consistently.
    """
    try:
        # --- Step 1: OCR ---
        print("[DEBUG] Starting OCR on uploaded images...")
        combined_text = ""
        ocr_texts = []

        for idx, image in enumerate(images, start=1):
            image_bytes = await image.read()
            if not image_bytes:
                print(f"[WARNING] Image {idx} is empty, skipping...")
                continue

            ocr_result = client_google_vision_api.document_text_detection(
                vision.Image(content=image_bytes)
            )
            extracted_text = ocr_result.full_text_annotation.text if ocr_result.full_text_annotation else ""
            ocr_texts.append(extracted_text)
            combined_text += extracted_text + "\n\n"
            print(f"[DEBUG][OCR] Extracted text from image {idx}:")
            print(extracted_text)
            print("-" * 40)

        if not combined_text.strip():
            print("[ERROR] No text extracted from images")
            return {"status": "error", "detail": "No text extracted from images"}

        print("[DEBUG][OCR] Combined extracted text from all images:")
        print(combined_text)
        print("=" * 80)

        student_response = combined_text.strip()

        # Optional: Clean up common OCR artifacts (multiple newlines, spaces)
        cleaned_response = ' '.join(student_response.split())
        print("[DEBUG][OCR] Cleaned-up student response (spaces normalized):")
        print(cleaned_response)
        print("=" * 80)

        
        print(f"[DEBUG] Total extracted student response length: {len(student_response)} characters")

        # --- Step 2: Retrieve instructions with dynamic k (two-pass, context-aware) ---
        print("[DEBUG] Retrieving relevant instructions from vector store...")
        retrieval_query = (
           f"Provide all instructions, features, and marking rules relevant for answering: "
           f"{question_text}"
        )       
 
        retriever = qa_chain.retriever
        print("[DEBUG] Retrieval query:", retrieval_query)
        retrieved_docs = retriever.get_relevant_documents(retrieval_query)  # increase k from default 3 to 10
        
        if not retrieved_docs:
            print("[WARNING] No instructions retrieved from vector store")
            retrieved_context = "No instructions retrieved. Model should give 0 marks for all features."
        else:
            retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
            print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents, total length: {len(retrieved_context)} chars")
        # ✅ Print the actual retrieved context for debugging
        print(f"[DEBUG] Retrieved context:\n{retrieved_context}\n{'-'*80}")

        # --- Step 3: Construct strict evaluation prompt ---
        evaluation_prompt = f"""
You are an expert sociology examiner and a supportive teacher.
Use ONLY the retrieved instructions below to evaluate the student's response.
Treat the instructions as authoritative. Do NOT use any outside knowledge.

--- Retrieved Instructions ---
{retrieved_context}
---------------------------

Question:
{question_text}

Student Response:
{student_response}

Task:

1. **Improved Response:**
   - Rewrite the student response into the strongest possible version that would receive maximum marks STRICTLY based on the retrieved instructions.
   - Include ONLY points, features, or examples explicitly mentioned in the instructions.
   - Keep the response concise but complete.
   - Ensure the response meets the minimum word count of {minimum_word_count} words.

2. **Detailed Marking and Feedback (STRICT Scheme Compliance):**
   - Identify all attempted features or points in the response.
   - For each attempted feature, present it in the following format:
     - **Attempted Feature:** <text>
     - **Closest matching phrase:** <text from instructions>
     - **Marks awarded:** <marks>
   - Assign marks strictly based on the retrieved instructions.
   - Do NOT reward more features than allowed by the instructions.
   - Features cannot be double-counted.
   - Ensure total marks do not exceed {total_marks}.
   - Provide optional notes (e.g., spelling, grammar, minor clarity issues).

3. **Overall Assessment:**
   - Summarize how well the response meets the retrieved instructions.
   - Confirm whether the minimum word count was achieved.
   - Provide practical advice strictly tied to instructions **only if the response did not achieve the maximum marks**.
   - State the final mark in the format: **Overall Mark: <score/{total_marks}>**.

Format your answer as clear, structured text using bullet points if helpful. Do NOT use JSON or any structured data format. Make sure to wrap the main headings and subheadings in `**` so that they render as bold on the front end.
"""



        print("[DEBUG][PROMPT] Evaluation prompt sent to LLM:")
        print(evaluation_prompt)
        print("=" * 80)

        # --- Step 4: Run evaluation ---
        print("[DEBUG] Sending evaluation prompt to QA chain...")
        evaluation_result = qa_chain.run(evaluation_prompt)
        print("[DEBUG] Received evaluation result from QA chain (raw):")
        print(evaluation_result)
        print("=" * 80)
        
        # No JSON parsing attempt, just return the text
        return {
            "status": "success",
            "evaluation_text": evaluation_result,   # The full free-form feedback and assessment
            "total_marks": total_marks,
            "minimum_word_count": minimum_word_count,
            "student_response": student_response
        }
    except Exception as e:
        print(f"[ERROR] Exception occurred during evaluation: {e}")
        return {"status": "error", "detail": str(e)}
        
# -----------------------------
# Initialize QA chain
# -----------------------------
def initialize_qa_chain_anz_way(bucket_name: str, folder_in_bucket: str):
    global qa_chain_anz_way

    print("[DEBUG] Initializing QA Chain (anz way)...")

    try:
        # Step 1: Download vector store from GCS
        temp_dir = download_vectorstore_from_gcs(bucket_name, folder_in_bucket)

        # Step 2: Recreate embeddings (must match original used in training)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )

        # Step 3: Load FAISS index with embeddings
        vectorstore = FAISS.load_local(
            temp_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Step 4: Create retriever and QA chain
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        qa_chain_anz_way = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key),
            retriever=retriever,
            chain_type="stuff"
        )

        print("[DEBUG] QA Chain (anz way) initialized successfully.")
        return qa_chain_anz_way


    except Exception as e:
        print(f"[ERROR] Failed to initialize QA Chain (anz way): {e}")
        traceback.print_exc()

#to send the student exam prepartion data to anz way@app.post("/student_report")
@app.post("/student_report", response_model=List[StudentReflectionSchema])
def get_student_report(req: StudentReportRequest, db: Session = Depends(get_db)):
    print("=== get_student_report called ===")
    print(f"Student ID: {req.student_id}, From: {req.from_date}, To: {req.to_date}")

    try:
        # Convert dates from string to datetime objects
        from_dt = datetime.strptime(req.from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(req.to_date, "%Y-%m-%d")
        to_dt = datetime.combine(to_dt, time.max)  # include the entire "to" day

        # Query reflections for this student in date range
        reflections = (
            db.query(StudentReflection)
            .filter(
                StudentReflection.student_id == req.student_id,
                StudentReflection.created_at >= from_dt,
                StudentReflection.created_at <= to_dt
            )
            .order_by(StudentReflection.created_at.desc())
            .all()
        )

        print(f"Fetched {len(reflections)} reflections for student {req.student_id}")
        return reflections

    except Exception as e:
        print("Error fetching student report:", e)
        raise HTTPException(status_code=500, detail="Internal server error")
        

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the approximate cost of an OpenAI API call in USD.
    Adjust the rates according to OpenAI's pricing for each model.
    """
    rates = {
        "gpt-4o-mini": {"prompt": 0.0015, "completion": 0.002},  # $ per 1k tokens
        "text-embedding-3-small": {"prompt": 0.0004, "completion": 0},
    }

    model_rates = rates.get(model_name)
    if not model_rates:
        print(f"[WARNING] No cost info for model '{model_name}', defaulting to 0")
        return 0.0

    cost = (prompt_tokens / 1000) * model_rates["prompt"] + \
           (completion_tokens / 1000) * model_rates["completion"]

    print(f"[DEBUG] Calculated cost for model '{model_name}': "
          f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, cost_usd={cost:.6f}")

    return round(cost, 6)


def initialize_qa_chain_with_cost(bucket_name: str, folder_in_bucket: str, username: str):
    global qa_chain_anz_way

    print(f"[DEBUG] Initializing QA Chain for bucket='{bucket_name}', folder='{folder_in_bucket}', username='{username}'")

    try:
        # Step 1: Download vector store
        print("[DEBUG] Downloading vector store from GCS...")
        temp_dir = download_vectorstore_from_gcs(bucket_name, folder_in_bucket)
        print(f"[DEBUG] Vector store downloaded to temporary directory: {temp_dir}")

        # Step 2: Recreate embeddings
        print("[DEBUG] Creating embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        print("[DEBUG] Embeddings created successfully.")

        # Step 3: Load FAISS index
        print("[DEBUG] Loading FAISS vector store...")
        vectorstore = FAISS.load_local(
            temp_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[DEBUG] FAISS vector store loaded successfully.")

        # Step 4: Create retriever
        print("[DEBUG] Creating retriever...")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        print("[DEBUG] Retriever created successfully.")

        # Step 5: Create QA chain (same as before)
        print("[DEBUG] Creating RetrievalQA chain...")
        qa_chain_anz_way = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key),
            retriever=retriever,
            chain_type="stuff"
        )
        print("[DEBUG] QA chain created successfully.")

        

        print("[DEBUG] QA Chain (anz way) initialized successfully.")
        return qa_chain_anz_way

    except Exception as e:
        print(f"[ERROR] Failed to initialize QA Chain (anz way): {e}")
        traceback.print_exc()
        return None


def log_to_db(
    db: Session,
    username: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    model_name: str = "",
    is_embedding: bool = False
):
    """Logs token usage + cost into the DB using SQLAlchemy ORM session."""

    cost_info = MODEL_COSTS.get(model_name, {})

    if is_embedding:
        # Embedding cost = total_tokens × per-token embedding price
        embedding_cost = cost_info.get("embedding", 0)
        cost_usd = total_tokens * embedding_cost
    else:
        # Chat/Completion cost
        cost_usd = (
            (prompt_tokens * cost_info.get("prompt", 0)) +
            (completion_tokens * cost_info.get("completion", 0))
        )

    new_entry = CostPerInteraction(
        username=username,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=Decimal(str(cost_usd))  # ensures precision
    )

    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    print(f"[DEBUG] Token usage logged: user={username}, model={model_name}, tokens={total_tokens}, cost=${cost_usd:.6f}")

@app.post("/student_total_usage")
def get_student_total_usage(req: StudentUsageRequest, db: Session = Depends(get_db)):
    try:
        # Sum total cost for the given student
        total_usage = db.query(func.sum(CostPerInteraction.cost_usd)) \
                        .filter(CostPerInteraction.username == req.student_name) \
                        .scalar() or 0.0

        return {"total_usage_usd": float(total_usage)}

    except Exception as e:
        print(f"[ERROR] Failed to get student usage: {e}")
        return {"total_usage_usd": 0.0}
        

@app.get("/check-user-access")
def check_user_access(username: str, db: Session = Depends(get_db)):
    print("[DEBUG] --- check_user_access called ---")
    print(f"[DEBUG] Username received: {username}")

    # Query total cost
    total_cost = db.query(func.sum(CostPerInteraction.cost_usd)).filter(
        CostPerInteraction.username == username
    ).scalar() or 0.0

    print(f"[DEBUG] Total accumulated cost for user '{username}': {total_cost} USD")

    access_allowed = total_cost < MAX_USER_COST
    print(f"[DEBUG] Access allowed: total_cost < MAX_USER_COST ({total_cost} < {MAX_USER_COST})")

    response_content = {
        "username": username,
        "total_cost": float(total_cost),  # <-- convert Decimal to float
        "access_allowed": access_allowed,
        "max_allowed_cost": MAX_USER_COST
    }

    print(f"[DEBUG] Response content prepared: {response_content}")

    return JSONResponse(content=response_content)
    
    

@app.get("/users_total_usage")
def users_total_usage(
    min_usage: float = Query(0, description="Minimum usage filter"),
    max_usage: float = Query(1000, description="Maximum usage filter"),
    db: Session = Depends(get_db)
):
    # Query sum of tokens and cost per user
    users = db.query(
        CostPerInteraction.username,
        func.sum(CostPerInteraction.total_tokens).label("total_tokens"),
        func.sum(CostPerInteraction.prompt_tokens).label("total_prompt_tokens"),
        func.sum(CostPerInteraction.completion_tokens).label("total_completion_tokens")
    ).group_by(CostPerInteraction.username).all()

    response = []
    for u in users:
        # Compute USD using latest per-token rates
        prompt_cost = MODEL_COST_PER_TOKEN["gpt-4o-mini"]["prompt"]
        completion_cost = MODEL_COST_PER_TOKEN["gpt-4o-mini"]["completion"]
        total_usd = (u.total_prompt_tokens * prompt_cost) + (u.total_completion_tokens * completion_cost)

        # Apply filter
        if min_usage <= total_usd <= max_usage:
            response.append({
                "username": u.username,
                "total_tokens": int(u.total_tokens),
                "total_usage_usd": float(total_usd)
            })

    return JSONResponse(content=response)
    
#when start conversation is pressed
@app.post("/chat_anz_way_model_evaluation")
async def chat_with_ai(
    req: StartConversationRequest,
    db: Session = Depends(get_db)   # <-- injects DB session
):
    try:
        subject = req.subject
        question_text = req.question_text
        marks = req.marks
        username = req.username 

        print("\n[DEBUG] --- /chat_anz_way_model_evaluation called ---")
        print(f"[DEBUG] Received request: subject='{subject}', marks={marks}, "
        f"question_text='{question_text[:100]}...', username='{req.username}'")


        # --- Step 1: Initialize QA chain ---
        print(f"[DEBUG] Initializing QA chain for subject '{subject}'...")
        try:
            folder_in_bucket = f"{subject}_instructions.faiss"
            qa_chain_anz_way = initialize_qa_chain_with_cost(
                bucket_name="sociology_anz_way",
                folder_in_bucket=f"{subject}_instructions.faiss",
                username=req.username
            )

            print(f"[DEBUG] QA chain initialized successfully for subject: {subject}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize QA chain for subject {subject}: {str(e)}")
            return JSONResponse(
                content={"status": "error", "detail": str(e)}
            )

        # --- Step 2: Retrieve relevant instructions/context ---
        retrieval_query = f"Provide all instructions, features, and marking rules relevant for answering: {question_text}"
        print(f"[DEBUG] Retrieval query: {retrieval_query[:100]}...")

        retriever = qa_chain_anz_way.retriever
        retrieved_docs = retriever.get_relevant_documents(retrieval_query)
        if not retrieved_docs:
            print("[WARNING] No instructions retrieved from vector store")
            retrieved_context = "No instructions retrieved. Model should give 0 marks for all features."
        else:
            retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
            print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents, total length: {len(retrieved_context)} chars")

        # --- Step 3: Generate session ID ---
        session_id = str(uuid.uuid4())
        sessions[session_id] = []
        print(f"[DEBUG] New session created: session_id={session_id}")

        # --- Step 4: Compose AI prompt ---
        prompt = f"""
        You are an expert exam tutor guiding a student in {subject}. The student has asked the following question:

        Question: {question_text}
        Marks: {marks}
        
        Context/Instructions/Marking scheme from syllabus/vector store:
        {retrieved_context}
        
        Your instructions:
        
        1. **Start your first response by explicitly presenting the marking scheme** for this question as extracted from the retrieved context. Make it clear which points or features are expected and how marks are allocated.
        2. After presenting the marking scheme, adopt the tone of a **friendly, encouraging teacher**. Guide the student step by step on how to approach and answer the question.
        3. Ask interactive questions to gauge the student’s understanding.
        4. For each student response, reference the marking scheme to indicate how well they would score.
        5. Encourage learning: highlight strengths, gently point out areas to improve, and provide actionable tips to help them score better.
        6. Stay strictly on-topic; do not answer unrelated questions.
        7. Keep explanations clear, structured, and concise so the student can follow easily.
        
        Example behavior:
        
        - Begin: “Here is the marking scheme for this question: …”
        - Then ask: “Can you explain your approach to the first part of the question?”
        - After student response: “Based on your answer, you are likely to score X out of {marks}. To improve, consider Y.”
        
        Remember: Your goal is to simulate a patient, interactive tutor who helps the student understand the question, improve their answer, and maximize their marks.

        """

        print(f"[DEBUG] AI prompt composed, length={len(prompt)} chars")

        # --- Step 5: Call OpenAI API ---
        print("[DEBUG] Sending prompt to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that tutors students using provided instructions and marking scheme."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        ai_reply = response.choices[0].message.content.strip()
        print(f"[DEBUG] Received AI reply (truncated 200 chars): {ai_reply[:200]}")

        usage = response.usage
        log_to_db(db, username, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, "gpt-4o-mini")

        # --- Step 6: Store session messages ---
        sessions[session_id].append({"role": "user", "content": question_text})
        sessions[session_id].append({"role": "assistant", "content": ai_reply})
        print(f"[DEBUG] Session messages stored. Total messages: {len(sessions[session_id])}")

        print("[DEBUG] Returning response to frontend")
        return JSONResponse(
            content={"reply": ai_reply, "session_id": session_id}
        )

    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            content={"status": "error", "detail": str(e)}
        )


MAX_EXCHANGES = 5  # 5 exchanges = 10 messages (user+assistant)
@app.post("/send_message_anz_way_model_evaluation")
async def send_message(
    req: SendMessageRequest,
    db: Session = Depends(get_db)
):    
    global sessions

    try:
        print("\n[DEBUG] Received message request:")
        print(f"  session_id: {req.session_id}")
        print(f"  user_message: {req.message[:100]}")  # truncate for readability

        # Validate session
        if req.session_id not in sessions:
            print("[WARNING] Invalid session_id received.")
            return JSONResponse(
                status_code=400,
                content={"reply": "⚠️ Invalid session_id. Start a new conversation first."}
            )

        session_history = sessions[req.session_id]
        print(f"[DEBUG] Current session length: {len(session_history)} messages")

        # Check if max exchanges reached
        if len(session_history) >= MAX_EXCHANGES * 2:
            print(f"[INFO] Max exchanges reached for session {req.session_id}")
        
            # --- Evaluate preparedness before ending session ---
            try:
                print("[DEBUG] Preparing final AI evaluation for DB save")
            
                # Construct prompt for AI to extract question and preparedness
                final_prompt = f"""
                You are an AI tutor evaluating a student's exam preparation.
                
                Instructions:
                1. From the conversation below, extract the main exam question that the student was preparing for. Use the **most recent relevant question**.
                2. Assess the student's preparedness using **exactly one of these labels**: "Well prepared", "Needs improvement", "Not prepared".
                3. Return the result **strictly in pure JSON only** — do not include backticks, markdown, explanations, or any extra text.
                
                Conversation:
                {session_history}
                
                Output JSON format (strictly):
                {{
                    "question_text": "...",
                    "preparedness_level": "..."
                }}
                """

                print(f"[DEBUG] Sending final evaluation prompt to OpenAI (truncated 200 chars): {final_prompt[:200]}...")
            
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": final_prompt}]
                )
            
                reply_content = response.choices[0].message.content
                print(f"[DEBUG] Received final AI evaluation reply (truncated 200 chars): {reply_content[:200]}")
            
                # Parse JSON response
                summary = json.loads(reply_content)
                extracted_question = summary.get("question_text", "N/A")
                ai_label = summary.get("preparedness_level", "Not prepared")
                print(f"[DEBUG] Extracted question (truncated 100 chars): {extracted_question[:100]}")
                print(f"[DEBUG] AI reported preparedness level: {ai_label}")
            
                # Convert AI string to Enum
                ai_enum_value = ai_label.lower().replace(" ", "_")
                try:
                    preparedness_enum = PreparednessLevel(ai_enum_value)
                except ValueError:
                    print(f"[WARNING] AI returned unexpected preparedness value '{ai_label}', defaulting to 'needs_improvement'")
                    preparedness_enum = PreparednessLevel.needs_improvement
            
                # Save evaluation to database
                new_reflection = StudentReflection(
                    student_id=req.id,
                    question_text=extracted_question,
                    preparedness_level=preparedness_enum
                )
                db.add(new_reflection)
                db.commit()
                db.refresh(new_reflection)
                print(f"[DB] Saved student reflection with ID {new_reflection.id}")
            
            except Exception as e:
                db.rollback()
                print("[ERROR] Failed to save student reflection:", e)
            
        
            # Return response to frontend
            return JSONResponse(
                status_code=200,
                content={
                    "reply": "✅ This conversation has reached the 5-exchange limit. Please start a new session for further evaluation.",
                    "session_id": req.session_id,
                    "conversation_ended": True
                }
            ) 
        # Append user message to session
        session_history.append({"role": "user", "content": req.message})
        print(f"[DEBUG] Appended user message. Session length now: {len(session_history)}")

        # System prompt
        system_prompt = (
            "You are a helpful AI tutor for sociology exam preparation. "
            "Always stay focused on the specific exam question being attempted. "
            "Give step-by-step feedback strictly using the marking scheme. "
            "Keep answers short, clear, and easy to read — avoid long paragraphs. "
            "Encourage the student with guiding questions, but let them do most of the thinking. "
            "❌ Do not drift into general sociology or unrelated examples. "
            "If the student goes off-topic, politely redirect them back to the current question."
        )


        print("[DEBUG] System prompt prepared.")

        # Combine system prompt + session history
        messages = [{"role": "system", "content": system_prompt}] + session_history
        print("[DEBUG] Combined messages to send to OpenAI API:")
        for idx, m in enumerate(messages):
            preview = m['content'][:80].replace("\n", " ")
            print(f"  {idx+1}. {m['role']}: {preview}...")

        # Call OpenAI API
        print("[DEBUG] Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        reply = response.choices[0].message.content
        print(f"[DEBUG] Received AI reply (truncated 150 chars): {reply[:150]}")

        # Save AI reply
        session_history.append({"role": "assistant", "content": reply})
        print(f"[DEBUG] Appended AI reply. Session length now: {len(session_history)}")

        usage = response.usage
        log_to_db(
            db,
            username=req.username,            # Use the username from the request
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model_name="gpt-4o-mini"
        )

        

        return JSONResponse(
            content={
                "reply": reply,
                "session_id": req.session_id,
                "conversation_ended": False
            }
        )

    except Exception as e:
        print("[ERROR] Exception while processing message:")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "reply": "⚠️ Server error. Please try again later.",
                "detail": str(e)
            }
        )

# a new evaluate your essay so that anser could include diagrams
async def evaluate_student_response_from_images_new(
    db: Session,                # <-- database session for logging
    images: List[UploadFile],
    question_text: str,
    total_marks: int,
    qa_chain,                   # RetrievalQA
    minimum_word_count: int = 80,
    student_response: str = None,
    username: str = None        # <-- added username for logging
):
    """
    Evaluate a student's response (text + optional diagrams) against a question.
    Works with OCR-only or pre-combined OCR+Vision input.
    Returns free-form structured text suitable for front-end display.
    Logs token usage roughly based on word count if username is provided.
    """
    print("\n[DEBUG] === Student Response Evaluation Started ===")

    # Step 1: Prepare student response
    if student_response:
        print("[DEBUG] Using pre-combined student response (OCR + Vision AI).")
    else:
        print("[DEBUG] No pre-combined response provided. Falling back to OCR-only.")
        extracted_texts = []

        for idx, image in enumerate(images):
            print(f"[DEBUG] Processing image {idx + 1}/{len(images)}: {image.filename}")
            contents = await image.read()
            try:
                ocr_result = ocr_client.document_text_detection(image={"content": contents})
                extracted_text = (
                    ocr_result.full_text_annotation.text.strip()
                    if ocr_result.full_text_annotation
                    else ""
                )
                print(f"[DEBUG] OCR extracted {len(extracted_text.split())} words.")
                extracted_texts.append(extracted_text)
            except Exception as e:
                print(f"[ERROR] OCR failed on image {image.filename}: {e}")
                extracted_texts.append("")

        student_response = "\n".join(extracted_texts).strip()
        print("[DEBUG] Final OCR-only extracted response length:",
              len(student_response.split()), "words")

    if not student_response.strip():
        print("[DEBUG] No student response extracted. Returning 0 marks.")
        return {
            "score": 0,
            "total": total_marks,
            "feedback": "No valid response detected in the submission."
        }

    # Step 2: Retrieve authoritative instructions
    retrieval_query = (
        f"Provide all instructions, features, and marking rules relevant for answering: "
        f"{question_text}"
    )

    retriever = qa_chain.retriever
    retrieved_docs = retriever.get_relevant_documents(retrieval_query)
    if not retrieved_docs:
        print("[WARNING] No instructions retrieved from vector store")
        retrieved_context = "No instructions retrieved. Model should give 0 marks for all features."
    else:
        retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
        print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents, total length: {len(retrieved_context)} chars")

    # Step 3: Construct evaluation prompt
    evaluation_prompt = f"""
You are an expert sociology examiner and a supportive teacher.
Use ONLY the retrieved instructions below to evaluate the student's response.
Treat the instructions as authoritative. Do NOT use any outside knowledge.

--- Retrieved Instructions ---
{retrieved_context}
---------------------------

Question:
{question_text}

Student Response:
{student_response}

Task:

1. **Improved Response:**
   - Rewrite the student response into the strongest possible version that would receive maximum marks STRICTLY based on the retrieved instructions.
   - Include ONLY points, features, or examples explicitly mentioned in the instructions.
   - Keep the response concise but complete.
   - Ensure the response meets the minimum word count of {minimum_word_count} words.

2. **Detailed Marking and Feedback (STRICT Scheme Compliance):**
   - Identify all attempted features or points in the response.
   - For each attempted feature, present it in the following format:
     - **Attempted Feature:** <text>
     - **Closest matching phrase:** <text from instructions>
     - **Marks awarded:** <marks>
   - Assign marks strictly based on the retrieved instructions.
   - Do NOT reward more features than allowed by the instructions.
   - Features cannot be double-counted.
   - Ensure total marks do not exceed {total_marks}.
   - Provide optional notes (e.g., spelling, grammar, minor clarity issues).

3. **Overall Assessment:**
   - Summarize how well the response meets the retrieved instructions.
   - Confirm whether the minimum word count was achieved.
   - Provide practical advice strictly tied to instructions **only if the response did not achieve the maximum marks**.
   - State the final mark in the format: **Overall Mark: <score/{total_marks}>**.

Format your answer as clear, structured text using headings, bullet points, and bold formatting for front-end display.
Do NOT return JSON or any structured data format.
"""

    print("[DEBUG][PROMPT] Evaluation prompt sent to QA chain:")
    print(evaluation_prompt)
    print("=" * 80)

    # Step 4: Run evaluation
    try:
        evaluation_result = qa_chain.run(evaluation_prompt)
        print("[DEBUG] Received evaluation result from QA chain (raw):")
        print(evaluation_result)
        print("=" * 80)

        # Step 5: Rough cost logging
        if username:
            # Estimate tokens roughly based on word count
            def estimate_tokens(text: str) -> int:
                return int(len(text.split()) / 0.75)  # ~1 token ≈ 0.75 words

            prompt_tokens = estimate_tokens(evaluation_prompt)
            completion_tokens = estimate_tokens(evaluation_result)
            total_tokens = prompt_tokens + completion_tokens

            log_to_db(
                db=db,
                username=username,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model_name="gpt-4o-mini"
            )
            print("[DEBUG] Logged token usage to DB.")

        return {
            "status": "success",
            "evaluation_text": evaluation_result,
            "total_marks": total_marks,
            "minimum_word_count": minimum_word_count,
            "student_response": student_response
        }

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return {
            "status": "error",
            "detail": str(e)
        }




def run_ocr_on_image(image_file) -> str:
    """
    Run OCR on an uploaded image using Google Vision API.
    
    Args:
        image_file (UploadFile): FastAPI UploadFile
    
    Returns:
        str: Extracted text from the image (empty string if none)
    """
    try:
        client = vision.ImageAnnotatorClient()

        # Read image into memory
        content = image_file.file.read()
        image = vision.Image(content=content)

        # Run OCR
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            print(f"[OCR ERROR] {response.error.message}")
            return ""

        if not texts:
            print("[OCR DEBUG] No text detected in image.")
            return ""

        # First annotation is the full text block
        extracted_text = texts[0].description.strip()
        print(f"[OCR DEBUG] Extracted text (length {len(extracted_text)}): {extracted_text[:200]}...")

        return extracted_text

    except Exception as e:
        print(f"[OCR EXCEPTION] Failed to run OCR: {str(e)}")
        return ""
    finally:
        # Reset file pointer so Vision AI (for diagrams) can also use it later
        image_file.file.seek(0)

async def run_vision_on_image(image_file):
    """
    Analyze an image with Gemini 1.5 Flash to extract diagram / visual notes.
    """
    try:
        # Read image bytes from the uploaded file
        image_bytes = await image_file.read()

        # Convert to Vertex AI Part
        image_part = Part.from_data(mime_type="image/png", data=image_bytes)

        # Prepare the prompt
        prompt = """
        You are an assistant that analyzes diagrams in student essays.
        Summarize what this diagram shows, list key points, and note 
        any observations that would help a teacher evaluate it.
        """

        # Run Gemini inference
        response = vision_model.generate_content(
            [prompt, image_part]
        )

        diagram_notes = response.text.strip() if response and response.text else ""
        return diagram_notes or "[No diagram analysis produced]"
    
    except Exception as e:
        print(f"[ERROR] Vision AI analysis failed for '{image_file.filename}': {e}")
        return "[Error during Vision AI analysis]"


#a new train-on-images so that answers could include diagrams
@app.post("/train-on-images-anz-way-new")
async def train_on_images_anz_way(
    db: Session = Depends(get_db),
    images: List[UploadFile] = File(...),
    subject: str = Form(...),
    question_text: str = Form(...),
    total_marks: int = Form(...),
    minimum_word_count: int = Form(...),
    username: str = Form(...),
    request: Request = None
):
    origin = request.headers.get("origin") if request else "*"
    cors_headers = {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true"
    }

    global qa_chain_anz_way
    print("\n[DEBUG] New request received")
    print(f"[DEBUG] Question: {question_text}")
    print(f"[DEBUG] Total Marks: {total_marks}")
    print(f"[DEBUG] Number of images uploaded: {len(images)}")

    # Initialize QA chain if needed
    if qa_chain_anz_way is None:
        try:
            folder_in_bucket = f"{subject}_instructions.faiss"
            qa_chain_anz_way = initialize_qa_chain_anz_way(
                bucket_name="sociology_anz_way",
                folder_in_bucket=folder_in_bucket
            )
            print(f"[DEBUG] qa_chain_anz_way initialized successfully for subject: {subject}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize QA chain: {str(e)}")
            return JSONResponse(content={"status": "error", "detail": str(e)}, headers=cors_headers)

    # Word count mapping
    word_count_map = {4: 50, 6: 80, 8: 120, 10: 150, 26: 450}
    if total_marks not in word_count_map:
        return JSONResponse(
            content={"status": "error", "detail": f"Invalid total_marks {total_marks}. Allowed: {list(word_count_map.keys())}"},
            headers=cors_headers
        )
    minimum_word_count = word_count_map[total_marks]

    # Process images: OCR + Vision AI
    combined_essay_text = ""
    combined_diagram_notes = ""
    for idx, image_file in enumerate(images, start=1):
        print(f"[DEBUG] Processing image #{idx}: {image_file.filename}")

        # OCR
        ocr_text = run_ocr_on_image(image_file)
        combined_essay_text += "\n" + ocr_text

        # Vision AI
        diagram_notes = await run_vision_on_image(image_file)
        combined_diagram_notes += "\n" + diagram_notes

    print(f"[DEBUG] Combined diagram notes: {combined_diagram_notes}")
    diagram_section = combined_diagram_notes.strip()
    if not diagram_section or diagram_section == "[No diagram analysis produced]":
        student_response = f"Extracted Essay Text:\n{combined_essay_text.strip()}"
    else:
        student_response = f"""Extracted Essay Text:
{combined_essay_text.strip()}

Diagram Interpretation:
{diagram_section}"""

    print("[DEBUG] Combined student response prepared.")

    # Run evaluation
    print("[DEBUG] Sending student response for evaluation...")
    eval_result = await evaluate_student_response_from_images_new(
        db=db,  # <-- pass database session for logging
        images=images,
        question_text=question_text,
        total_marks=total_marks,
        qa_chain=qa_chain_anz_way,
        minimum_word_count=minimum_word_count,
        student_response=student_response,
        username=username  # <-- pass username for logging
    )

    # Prepare frontend response
    response_payload = {
        "status": "success",
        "evaluation_text": eval_result.get("evaluation_text", "[No feedback returned]"),
        "total_marks": total_marks,
        "minimum_word_count": minimum_word_count,
        "student_response": student_response
    }

    print("[DEBUG] Returning response to frontend.")
    return JSONResponse(content=response_payload, headers=cors_headers)







@app.post("/train-on-images-anz-way")
async def train_on_images_anz_way(
    images: List[UploadFile] = File(...),
    question_text: str = Form(...),
    total_marks: int = Form(...),  # from frontend
    request: Request = None
):
    origin = request.headers.get("origin") if request else "*"
    cors_headers = {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true"
    }

    global qa_chain_anz_way

    # ✅ Lazy fallback: initialize if not ready
    if qa_chain_anz_way is None:
        try:
            
            qa_chain_anz_way = initialize_qa_chain_anz_way(
                bucket_name="sociology_anz_way",
                folder_in_bucket=folder_in_bucket
            )
        except Exception as e:
            return JSONResponse(
                content={
                    "status": "error",
                    "detail": f"Failed to initialize QA chain: {str(e)}"
                },
                headers=cors_headers
            )

    # ✅ Double-check if still None
    if qa_chain_anz_way is None:
        return JSONResponse(
            content={"status": "error", "detail": "QA chain still not initialized"},
            headers=cors_headers
        )

    # ✅ Define mapping dictionary
    word_count_map = {
    4: 100,    # full short-answer question (e.g., 2 reasons with explanations)
    6: 150,    # multi-part short answer / detailed paragraph
    8: 200,    # longer answer / mini-essay
    10: 250,   # extended answer / mini-essay with examples
    26: 450    # long essay-style answer / evaluation question
    }

    # ✅ Validate total_marks
    if total_marks not in word_count_map:
        return JSONResponse(
            content={
                "status": "error",
                "detail": f"Invalid total_marks: {total_marks}. "
                          f"Allowed values are {list(word_count_map.keys())}"
            },
            headers=cors_headers
        )

    minimum_word_count = word_count_map[total_marks]

    # ✅ Run evaluation
    result = await evaluate_student_response_from_images(
        images=images,
        question_text=question_text,
        total_marks=total_marks,
        qa_chain=qa_chain_anz_way,
        minimum_word_count=minimum_word_count
    )

    return JSONResponse(content=result, headers=cors_headers)





@app.options("/train-on-images")
async def options_train_on_images():
    """Handle CORS preflight requests explicitly."""
    return JSONResponse(
        content={"message": "CORS preflight OK"},
        headers={
            "Access-Control-Allow-Origin": "*",  # ⚠️ Replace with exact domain in production
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
        status_code=200,
    )


async def extract_text_helper(image: UploadFile) -> dict:
    # Read and prepare image for Vision API
    image_bytes = await image.read()
    image_content = vision.Image(content=image_bytes)

    # Run OCR
    response = client_google_vision_api.document_text_detection(image=image_content)
    ocr_text = response.full_text_annotation.text if response.full_text_annotation else ""

    if not ocr_text.strip():
        return {"text": ""}

    # Use OpenAI to clean up OCR text
    prompt = f"""
    The following text was extracted by an OCR system and may have mistakes or odd formatting. 
    Please rewrite it so that it is clear, but ONLY fix OCR mistakes (e.g., word splits, misread letters, misplaced line breaks). 
    Do NOT paraphrase or change meaning.

    Text:
    {ocr_text}

    Cleaned Text:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that ONLY fixes OCR errors, "
                    "without paraphrasing or altering meaning."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    cleaned_text = response.choices[0].message.content.strip()
    return {"text": cleaned_text}


@app.post("/train-on-images")
async def train_on_images(
    images: List[UploadFile] = File(...),
    doctorData: Optional[str] = Form(None),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """Endpoint: Upload essay images -> OCR -> Correction -> Improvement -> Mistake analysis -> Save to DB"""
    
    origin = request.headers.get("origin") if request else None
    cors_headers = {
        "Access-Control-Allow-Origin": origin if origin else "*",
        "Access-Control-Allow-Credentials": "true",
    }

    if not images:
        return JSONResponse(
            content={"detail": "No images uploaded"},
            status_code=400,
            headers=cors_headers,
        )

    # Parse doctorData JSON
    doctor = {}
    if doctorData:
        try:
            doctor = json.loads(doctorData)
        except json.JSONDecodeError:
            return JSONResponse(
                content={"detail": "Invalid doctorData JSON"},
                status_code=400,
                headers=cors_headers,
            )

    global username_for_interactive_session
    username_for_interactive_session = doctor.get("name") if doctor else None

    

    try: 
                
                # ✅ Inside /train-on-images you now do this:
        # STEP 1 & 2: Use your existing extract_text_helper() for OCR + cleanup
        # ------------------------------------------------
        combined_text = ""
        
        for image in images:
            if not image:
                continue
        
            # Call helper (dict guaranteed)
            result = await extract_text_helper(image)
            extracted_text = result.get("text", "") if isinstance(result, dict) else ""
            if extracted_text:
                combined_text += extracted_text.strip() + "\n\n"
        
        if not combined_text.strip():
            return JSONResponse(
                content={"detail": "No text extracted from images"},
                status_code=400,
                headers=cors_headers,
            )
        
        # This replaces the previous Step 2 output
        corrected_text = combined_text.strip()    
        # ------------------------------------------------
        # STEP 3: Improve essay quality with feedback
        # ------------------------------------------------
        improvement_prompt = f"""
    You are an expert creative writing tutor. Your goal is to help a student improve their writing skills.
    
    1. Rewrite the following essay with:
       - Better overall structure and flow
       - Clear grammar, punctuation, and sentence construction
       - Richer and more precise vocabulary
       - Logical organization and smooth transitions
       - Formal academic style appropriate for A-level essays
    
    2. Keep the original meaning intact. Do not add new ideas.
    
    3. Wrap **only the words, phrases, or sentences that are changed or improved** in double asterisks to highlight the actual improvements. Do NOT wrap the parts that remain unchanged.
    
    4. After the essay, provide a short note (2–3 sentences) summarizing key improvements.
    
    Original OCR-corrected essay:
    <<< BEGIN TEXT >>>
    {corrected_text}
    <<< END TEXT >>>
    """
        improvement_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative writing tutor helping a student improve their essay.",
                },
                {"role": "user", "content": improvement_prompt},
            ],
            temperature=0.3,
        )
        improved_text = improvement_response.choices[0].message.content.strip()


        # ------------------------------------------------
        # STEP 4: Merge original + improved essay
        # ------------------------------------------------
        final_output = f"""
        Original Text:
        <<< BEGIN ORIGINAL TEXT >>>
        {combined_text.strip()}
        <<< END ORIGINAL TEXT >>>

        Improved Text:
        <<< BEGIN IMPROVED TEXT >>>
        {improved_text}
        <<< END IMPROVED TEXT >>>
        """

        # ------------------------------------------------
        # STEP 5: Mistake analysis (AI -> JSON format)
        # ------------------------------------------------
        analysis_prompt = f"""
        You are an expert essay analysis assistant.

        Compare the original and improved essay.
        For each correction, create an object with:
        - "original_text": the exact original fragment
        - "corrected_text": the improved fragment
        - "category": one of ["Grammar", "Punctuation", "Vocabulary", "Sentence structure / flow", "Redundancy / conciseness"]
        - "explanation": a concise explanation of why the change was made

        Produce ONLY a JSON array of objects.
        If there are no corrections, return [].

        Original Essay:
        <<< BEGIN ORIGINAL >>>
        {combined_text.strip()}
        <<< END ORIGINAL >>>

        Improved Essay:
        <<< BEGIN IMPROVED >>>
        {improved_text}
        <<< END IMPROVED >>>
        """

        analysis_response = None
        try:
            analysis_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You output only clean, parsable JSON for essay mistake patterns."},
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0,
            )
        except Exception as e:
            print(f"[ERROR] Failed to get essay analysis: {e}")

        # Parse JSON safely
        mistake_patterns_data = []
        if analysis_response:
            raw_content = analysis_response.choices[0].message.content.strip()
            try:
                mistake_patterns_data = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse AI JSON: {e}")
                mistake_patterns_data = []

        # ------------------------------------------------
        # STEP 6: Save mistakes into DB
        # ------------------------------------------------
        saved_count = 0
        for idx, mistake in enumerate(mistake_patterns_data, start=1):
            try:
                mistake_record = CommonMistake(
                    session_id=username_for_interactive_session,  # Using username instead of session_id
                    original_text=mistake.get("original_text", ""),
                    corrected_text=mistake.get("corrected_text", ""),
                    category=mistake.get("category", ""),
                    explanation=mistake.get("explanation", ""),
                    created_at=datetime.utcnow(),
                )
                db.add(mistake_record)
                saved_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to create mistake record {idx}: {e}")

        try:
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Failed to commit mistake patterns: {e}")

        # ------------------------------------------------
        # STEP 7: Save session + Return response
        # ------------------------------------------------
        session_id = str(uuid4())
        session_texts[session_id] = {
            "text": final_output,
            "doctorData": doctor,
            "mistake_patterns": mistake_patterns_data,
        }

        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "images_processed": len(images),
                "total_text_length": len(final_output),
                "corrected_text": final_output,
                "mistake_patterns": mistake_patterns_data,
            },
            headers=cors_headers,
        )

    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(
            content={"detail": f"Unexpected server error: {str(e)}\n{tb_str}"},
            status_code=500,
            headers=cors_headers,
        )


# @app.post("/train-on-images")
#previous working code for CSS_Academy1
# async def train_on_images(
#     images: List[UploadFile] = File(...),
#     doctorData: Optional[str] = Form(None),
#     request: Request = None,
# ):
#     origin = request.headers.get("origin") if request else None
#     cors_headers = {
#         "Access-Control-Allow-Origin": origin if origin else "*",
#         "Access-Control-Allow-Credentials": "true",
#     }

#     if not images:
#         print("[train-on-images] No images uploaded")
#         return JSONResponse(
#             content={"detail": "No images uploaded"},
#             status_code=400,
#             headers=cors_headers,
#         )

#     # Parse doctorData JSON string to Python dict
#     doctor = {}
#     if doctorData:
#         try:
#             doctor = json.loads(doctorData)
#             print(f"[train-on-images] Received doctorData: {doctor}")
#         except json.JSONDecodeError:
#             print("[train-on-images] Invalid doctorData JSON")
#             return JSONResponse(
#                 content={"detail": "Invalid doctorData JSON"},
#                 status_code=400,
#                 headers=cors_headers,
#             )
#     else:
#         print("[train-on-images] No doctorData provided")

#     global username_for_interactive_session
#     username_for_interactive_session = doctor.get("name") if doctor else None

#     combined_text = ""

#     try:
#         print(f"[train-on-images] Received {len(images)} images")

#         for idx, image in enumerate(images, start=1):
#             print(f"[train-on-images] Reading image {idx}: filename={image.filename}")
#             image_bytes = await image.read()
#             print(f"[train-on-images] Read {len(image_bytes)} bytes from image {idx}")

#             if not image_bytes:
#                 print(f"[train-on-images] Warning: Image {idx} has zero bytes, skipping")
#                 continue

#             image_content = vision.Image(content=image_bytes)

#             print(f"[train-on-images] Sending image {idx} to Google Vision API")
#             response = client_google_vision_api.document_text_detection(image=image_content)

#             if response.error.message:
#                 print(f"[train-on-images] Google Vision API error for image {idx}: {response.error.message}")
#                 return JSONResponse(
#                     content={"detail": f"Google Vision API error: {response.error.message}"},
#                     status_code=500,
#                     headers=cors_headers,
#                 )

#             ocr_text = response.full_text_annotation.text if response.full_text_annotation else ""
#             print(f"[train-on-images] OCR text length for image {idx}: {len(ocr_text)} characters")

#             combined_text += ocr_text + "\n\n"

#         if not combined_text.strip():
#             print("[train-on-images] No text extracted from any images")
#             return JSONResponse(
#                 content={"detail": "No text extracted from images"},
#                 status_code=400,
#                 headers=cors_headers,
#             )

#         # Step 1: Correct OCR errors ONLY (no formatting, just corrected text)
#         correction_prompt = f"""
#         The following text is extracted using OCR and contains errors such as missing spaces, broken words, or misrecognized characters.
        
#         Your task is to correct only clear OCR errors, for example:
#         - Missing or extra spaces
#         - Broken or merged words
#         - Confused characters (e.g., '0' instead of 'O', '1' instead of 'I')
        
#         Do NOT paraphrase or reword sentences or change sentence structure. Only fix errors caused by OCR.
        
#         Wrap every corrected word or phrase in double asterisks (`**`) so corrections are visible.
        
#         Example:
#         Original: "Thesis Statement: Ensuring the provision f in human rights has become an illusion"
#         Corrected: "Thesis Statement: Ensuring the provision **of** human rights has become an illusion"
        
#         Text to correct:
#         <<< BEGIN TEXT >>>
#         {combined_text.strip()}
#         <<< END TEXT >>>
#         """

        
#         correction_response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are an assistant dedicated to correcting only obvious OCR errors. "
#                         "Fix broken words, missing spaces, and misrecognized characters. "
#                         "Do not paraphrase or change sentence structure. "
#                         "Wrap all corrections in double asterisks (`**`)."
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": correction_prompt
#                 }
#             ],
#             temperature=0  # deterministic corrections
#         )
#         corrected_text = correction_response.choices[0].message.content.strip()        
        
#         # Step 2: Produce final formatted output with Original and Improved Text
#         formatting_prompt = f"""
#         You are given the original OCR text and a corrected version of it.
        
#         Please produce the response exactly in this format:
        
#         Original Text:
#         <<< BEGIN ORIGINAL TEXT >>>
#         {combined_text.strip()}
#         <<< END ORIGINAL TEXT >>>
        
#         Improved Text:
#         <<< BEGIN IMPROVED TEXT >>>
#         [Copy the corrected version here, wrapping every correction in double asterisks (`**`)]
#         <<< END IMPROVED TEXT >>>
#         """
        
#         formatting_response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are an assistant that highlights corrections by wrapping changed words or phrases in `**`."
#                 },
#                 {
#                     "role": "user",
#                     "content": formatting_prompt.replace("[Copy the corrected version here, wrapping every correction in double asterisks (`**`)]", corrected_text)
#                 }
#             ],
#             temperature=0.2
#         )
        
#         final_output = formatting_response.choices[0].message.content.strip()
#         print(f"[train-on-images] Correction completed, final output length: {len(final_output)}")

#         # Generate a unique session ID and store data
#         session_id = str(uuid4())
#         session_texts[session_id] = {
#             "text": final_output,
#             "doctorData": doctor,
#         }
#         print(f"[train-on-images] Session {session_id} created with final output length {len(final_output)}")
        
#         return JSONResponse(
#             content={
#                 "status": "success",
#                 "session_id": session_id,
#                 "images_processed": len(images),
#                 "total_text_length": len(final_output),
#                 "corrected_text": final_output,  # send the formatted text with original + improved
#             },
#             headers=cors_headers,
#         )

#     except Exception as e:
#         tb_str = traceback.format_exc()
#         print(f"[train-on-images] Exception occurred: {str(e)}\nTraceback:\n{tb_str}")
#         return JSONResponse(
#             content={"detail": f"Unexpected server error: {str(e)}"},
#             status_code=500,
#             headers=cors_headers,
#         )



@app.post("/chat_interactive_tutor", response_model=ChatResponse)
async def chat_interactive_tutor(
    request: ChatRequest_CSS,
    db: Session = Depends(get_db)
):
    try:
        print("[DEBUG] Received request:", request)
        session_id = request.session_id.strip()
        user_message = request.message.strip()
        print(f"[DEBUG] session_id: {session_id}")
        print(f"[DEBUG] user_message: {user_message}")

        if session_id not in session_texts:
            print(f"[ERROR] Session ID {session_id} not found in session_texts")
            raise HTTPException(status_code=404, detail="Session ID not found")

        full_text = session_texts[session_id]
        print(f"[DEBUG] Retrieved full_text of length {len(full_text)}")

        if request.first_message:
            print("[DEBUG] Processing first message of the session")

            system_message = {
                "role": "system",
                "content": (
                    "You are a concise and insightful creative writing tutor.\n"
                    "You are helping a student improve their writing, which may be a short passage, several paragraphs, or a full essay.\n"
                    "Only respond to user queries that are directly relevant to the essay shared. Do not entertain or answer unrelated questions.\n"
                    "After each suggestion or feedback point, insert two newline characters (\\n\\n) to create a blank line for better readability.\n"
                    "Keep your responses under 150 words unless the student explicitly requests a full rewrite. Do not exceed this limit in regular responses."
                ),
            }

            user_intro_message = {
                "role": "user",
                "content": (
                    f"Here is the student's writing passage:\n\n{full_text}\n\n"
                    "Please keep your feedback concise (within 150 words) unless asked for a full essay rewrite and use line breaks after each point."
                    "Keep this in mind for the session. Now, here is my question:"
                )
            }

            user_current_message = {
                "role": "user",
                "content": (
                    f"{user_message}\n\n"
                    "Please insert two newline characters (\\n\\n) after each suggestion to improve readability. "
                    "Keep the response under 150 words. "
                    "Do not respond to queries unrelated to the essay."
                )
            }

            messages = [system_message, user_intro_message, user_current_message]
            session_histories[session_id] = messages.copy()
            print(f"[DEBUG] Initialized session_histories[{session_id}] with messages")

        else:
            if session_id not in session_histories:
                print(f"[WARN] Session history missing for session ID {session_id}. Initializing with intro + full_text.")
            
                system_message = {
                    "role": "system",
                    "content": (
                        "You are a concise and insightful creative writing tutor.\n"
                        "You are helping a student improve their writing...\n"
                        "Only respond to user queries that are directly relevant to the essay shared...\n"
                    )
                }
            
                user_intro_message = {
                    "role": "user",
                    "content": (
                        f"Here is the student's writing passage:\n\n{session_texts[session_id]}\n\n"
                        "Please keep your feedback concise (within 150 words) unless asked for a full rewrite..."
                    )
                }
            
                session_histories[session_id] = [system_message, user_intro_message]

            messages = session_histories[session_id]
            print(f"[DEBUG] Current session message count: {len(messages)}")
            messages.append({"role": "user", "content": user_message})
            print(f"[DEBUG] Appended user message to session history")

        # --- Call OpenAI ---
        model_name = "gpt-3.5-turbo"
        print(f"[DEBUG] Sending request to OpenAI model {model_name} with {len(messages)} messages")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
        )

        reply = response.choices[0].message.content.strip()
        usage = response.usage
        print(f"[DEBUG] OpenAI response received, reply length: {len(reply)}")
        print(f"[DEBUG] Token usage: {usage}")

        # --- Store Cost Info ---
        cost = calculate_cost(model_name, usage.prompt_tokens, usage.completion_tokens)
        print(f"[DEBUG] Calculated cost: ${cost:.6f}")

        cost_record = CostPerInteraction(
            username=username_for_interactive_session,
            model=model_name,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=cost,
            created_at=datetime.utcnow()
        )

        try:
            db.add(cost_record)
            db.commit()
            print("[DEBUG] Cost record saved to database")
        except SQLAlchemyError as e:
            db.rollback()
            print(f"[ERROR] Failed to save cost_record to database: {e}")

        # --- Update session history and return ---
        session_histories[session_id].append({"role": "assistant", "content": reply})
        print("[DEBUG] Appended assistant reply to session history")
        return ChatResponse(reply=reply)

    except HTTPException:
        # Re-raise HTTPExceptions so FastAPI can handle them normally
        raise

    except Exception as e:
        print(f"[ERROR] Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
        

@app.post("/api/pdf_chatbot")
async def chat(
    request: ChatRequest_interactive_pdf,
    db: Session = Depends(get_db)  # Replace with your auth dependency
):
    user_message = request.message
    username_for_interactive_session = request.user_name
    # Assuming vectorStore is always initialized elsewhere
    #qa_chain, relevant_texts, document_metadata = create_qa_chain(vectorstore, user_message)  
    qa_chain, relevant_texts, document_metadata = create_qa_chain(vectorstore, user_message, db=db, username=username_for_interactive_session, openai_api_key=openai_api_key)
    # Merge relevant texts into a single string for the prompt
   

    # Run the QA chain with all required inputs
    answer = qa_chain.run(user_message)  

    S3_BASE_URL = "https://pdfquerybucket.s3.amazonaws.com"
    UPLOADS_FOLDER = "upload"

    context_data = []
    for metadata in document_metadata:
        pdf_s3_key = metadata[0]  # e.g., "folder/subfolder/file.pdf"
        pdf_page = metadata[1]

        safe_pdf_key = quote(pdf_s3_key)  # URL encode to handle spaces/special chars

        pdf_url = f"{S3_BASE_URL}/{UPLOADS_FOLDER}/{safe_pdf_key}"

        context_data.append({
            "page_number": pdf_page,
            "pdf_url": pdf_url,
        })
        context_data.append({
            "page_number": pdf_page,
            "pdf_url": pdf_url,
        })

    # Process relevant texts for search strings
    search_strings = separate_sentences(relevant_texts)
    bot_reply = f"ChatBot: {answer}"
    relevant_texts=clean(relevant_texts)
    response = {
        "reply": bot_reply,
        "context": context_data,
        "search_strings": search_strings,
        "relevant_texts": relevant_texts  # Add this line
    }

    print(response)
    

    return JSONResponse(content=response)

@app.post("/api/train_model")
async def train_model(pages: PageRange, db: Session = Depends(get_db)):
    
    try:
        # Assuming BASELINE_COST and USAGE_LIMIT_INCREASE are floats, convert them to Decimal
        baseline_cost = Decimal(str(BASELINE_COST))
        usage_limit_increase = Decimal(str(USAGE_LIMIT_INCREASE))
        
        total_cost = db.query(func.sum(CostPerInteraction.cost_usd)).scalar() or Decimal('0.0')
        print(f"[DEBUG] Total cost so far: {total_cost}")
        print(f"[DEBUG] Baseline cost: {baseline_cost}")
        print(f"[DEBUG] Usage limit increase threshold: {usage_limit_increase}")
        
        if total_cost - baseline_cost >= usage_limit_increase:
            print(f"[DEBUG] Usage limit exceeded. Total cost increase: {total_cost - baseline_cost}")
            raise HTTPException(status_code=403, detail="The usage limit has exceeded.")
        else:
            print(f"[DEBUG] Usage limit not exceeded. Proceeding with training.")
    
        start_page = pages.start_page
        end_page = pages.end_page
        username_for_interactive_session = pages.user_name

        print("=" * 50)
        print(f"[INFO] Received request to train model")
        print(f"[INFO] Page range: {start_page} to {end_page}")
        print(f"[INFO] Username: {username_for_interactive_session}")

        combined_text = {}
        total_pdf = 0

        print("[INFO] Listing S3 objects in 'upload/'")
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="upload/")

        if "Contents" not in response:
            print("[WARNING] No contents found in S3 under 'upload/'")
            return JSONResponse(
                status_code=404,
                content={"error": "No PDF files found in the S3 'upload/' folder."}
            )

        for obj in response["Contents"]:
            key = obj["Key"]
            if key.lower().endswith(".pdf") and not key.endswith("/"):
                try:
                    print(f"[INFO] Processing file: {key}")
                    total_pdf += 1

                    s3_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                    file_stream = BytesIO(s3_obj["Body"].read())

                    print(f"[INFO] Extracting text from PDF '{key}' between pages {start_page} and {end_page}")
                    pdf_data = extract_text_from_pdf(file_stream, start_page, end_page)

                    combined_text.update(pdf_data)

                except Exception as e:
                    print(f"[ERROR] Failed to process {key}")
                    traceback.print_exc()
                    continue

        if not combined_text:
            print("[WARNING] No text extracted from any PDF.")
            return JSONResponse(
                status_code=400,
                content={"error": "No valid text extracted from any PDF."}
            )

        print("[INFO] Creating/loading vector store")
        vectorstore, embeddings = create_or_load_vectorstore(
            pdf_text=combined_text,
            username=username_for_interactive_session,
            openai_api_key=openai_api_key,
            s3_client=s3,
            bucket_name=BUCKET_NAME,
            db=db
        )
    

        print(f"[SUCCESS] Vectorstore trained using {total_pdf} PDFs.")
        return JSONResponse(
            status_code=200,
            content={"message": f"Model trained successfully from {total_pdf} PDFs!"}
        )

    except ClientError as e:
        print(f"[S3 ERROR] Client error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to access S3 bucket.", "details": str(e)}
        )

    except Exception as e:
        print(f"[ERROR] Unexpected error during training: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Model could not be trained!", "details": str(e)}
        )

#start here 

@app.post("/extract_text_pdfEssayChecker_mywebsite")
async def extract_text_pdfEssayChecker_mywebsite(pdf_file: UploadFile = File(...)):
    try:
        print(f"\n📥 [INFO] Received file: {pdf_file.filename}")

        # Step 1: Read PDF bytes
        try:
            pdf_bytes = await pdf_file.read()
            print(f"[DEBUG] PDF file read successful. Size: {len(pdf_bytes)} bytes.")
        except Exception as read_err:
            print("[❌ ERROR] Failed to read PDF file.")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail="Unable to read uploaded PDF.")

        # Step 2: Convert PDF to images
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = len(doc)
            print(f"[INFO] PDF loaded. Number of pages: {num_pages}")
            if num_pages == 0:
                print("[⚠️ WARNING] PDF contains no pages.")
                raise HTTPException(status_code=400, detail="Uploaded PDF has no pages.")
        except Exception as open_err:
            print("[❌ ERROR] Failed to open or parse PDF with fitz.")
            traceback.print_exc()
            raise HTTPException(status_code=422, detail="Invalid or corrupt PDF.")

        full_ocr_text = ""
        for i, page in enumerate(doc):
            try:
                print(f"\n📝 [INFO] Processing page {i+1}/{num_pages}")
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                image = vision.Image(content=img_bytes)

                # OCR with Google Vision
                print("[INFO] Calling Google Vision OCR API...")
                response = client_google_vision_api.document_text_detection(image=image)
                page_text = response.full_text_annotation.text.strip()

                if not page_text:
                    print(f"[⚠️ WARNING] Page {i+1} returned no OCR text.")
                else:
                    print(f"[DEBUG] OCR text length for page {i+1}: {len(page_text)} characters.")

                full_ocr_text += page_text + "\n"

            except Exception as ocr_err:
                print(f"[❌ ERROR] OCR failed on page {i+1}.")
                traceback.print_exc()
                continue  # Process remaining pages

        if not full_ocr_text.strip():
            print("[❌ ERROR] OCR process completed but no text was extracted.")
            raise HTTPException(status_code=422, detail="OCR failed to extract any text.")

        print(f"[INFO] Total extracted OCR text: {len(full_ocr_text)} characters.")

        # Step 3: GPT OCR Correction
        correction_prompt = f"""
You are given an essay extracted via OCR. Your task is to fix only clear OCR-related errors such as:
- Missing or extra spaces between words
- Broken or joined words
- Stray or incorrect characters (like '1' instead of 'I')

⚠️ Do NOT paraphrase, rewrite, or change the sentence structure or grammar unless it's clearly an OCR error.

Essay:
<<<
{full_ocr_text}
>>>
"""
        print("[🧠 GPT] Sending OCR correction prompt to GPT...")
        try:
            correction_response = client.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant that only corrects obvious OCR errors. Do not change any sentence structure or reword anything. "
                        "Only fix things like broken words, missing spaces, and random characters caused by OCR. "
                        "If you're unsure whether something is an OCR error, leave it unchanged."
                    )},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.2,
            )
            cleaned_text = correction_response.choices[0].message.content.strip()
            print(f"[INFO] Cleaned OCR text length: {len(cleaned_text)} characters.")
        except Exception as gpt_ocr_err:
            print("[❌ ERROR] GPT OCR correction failed.")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed during GPT OCR correction.")

        assessment_prompt = f"""
The following is a CSS English Essay written by a candidate.

You are a **strict but constructive CSS examiner** following official CSS English Essay evaluation criteria:
- Content & Relevance
- Organization & Coherence
- Grammar & Mechanics
- Vocabulary & Expression

Your tasks:
1. **Score each category (1–10)** and give a one-sentence justification.
2. **Provide detailed improvement feedback** in bullet points (at least 1 point per category).
3. **Rewrite the first 100 words** of the essay, marking all changes in **bold**.
4. **Briefly justify** your rewrites.

Essay:
{cleaned_text}
"""

        print("[🧠 GPT] Sending essay assessment prompt to GPT...")
        try:
            assessment_response = client.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a CSS essay examiner providing feedback and ideal writing samples."},
                    {"role": "user", "content": assessment_prompt}
                ],
                temperature=0.3,
            )
            assessment = assessment_response.choices[0].message.content.strip()
            print(f"[INFO] Assessment received. Length: {len(assessment)} characters.")
        except Exception as gpt_assess_err:
            print("[❌ ERROR] GPT assessment failed.")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed during GPT essay evaluation.")

        # Step 5: Extract Model Rewrite
        model_rewrite_match = re.search(
            r"\*\*Ideal Rewrite \(First 100 Words\):\*\*\s*(.+)", 
            assessment, 
            re.DOTALL
        )
        model_rewrite = model_rewrite_match.group(1).strip() if model_rewrite_match else ""
        print(f"[INFO] Extracted model rewrite: {len(model_rewrite)} characters.")

        return {
            "text1": cleaned_text,
            "text2": model_rewrite,
            "assessment": assessment
        }

    except Exception as e:
        print("[❌ FINAL ERROR] An exception occurred during full processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

#end here
# for extracting text from the image for my portfolio website
@app.post("/extract_text")
async def extract_text(image: UploadFile = File(...)):
    # Read and prepare image for Vision API
    image_bytes = await image.read()
    image_content = vision.Image(content=image_bytes)

    # Get raw OCR text (use document_text_detection or text_detection)
    response = client_google_vision_api.document_text_detection(image=image_content)
    ocr_text = response.full_text_annotation.text

    if not ocr_text:
        return {"text": ""}

    # Use OpenAI to clean up OCR text
    prompt = f"""
    The following text was extracted by an OCR system and may have mistakes or odd formatting. Please rewrite it so that it is clear, grammatically correct, and preserves the original meaning.

    Text:
    {ocr_text}

    Cleaned Text:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that fixes OCR errors such as incorrect word splits, "
                    "misread letters, and misplaced line breaks, while preserving the original grammar, "
                    "wording, and tone exactly as intended. Do not paraphrase, summarize, or correct grammar. "
                    "Only fix errors caused by OCR."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    cleaned_text = response.choices[0].message.content.strip()
    return {"text": cleaned_text}
@app.post("/extract_text_essayChecker_mywebsite")
async def extract_text(image: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await image.read()
        image_content = vision.Image(content=image_bytes)

        # Extract text using Vision API
        response = client_google_vision_api.document_text_detection(image=image_content)
        ocr_text = response.full_text_annotation.text if response.full_text_annotation else ""

        if not ocr_text:
            return {"text1": "", "text2": "", "assessment": "No text found in image."}

        # First clean OCR errors using GPT
        correction_prompt = f"""
        You are given an essay extracted via OCR. Your task is to fix only clear OCR-related errors such as:
        - Missing or extra spaces between words
        - Broken or joined words
        - Stray or incorrect characters (like '1' instead of 'I')
        
        ⚠️ Do NOT paraphrase, rewrite, or change the sentence structure or grammar unless it's clearly an OCR error.
        
        Essay:
        <<<
        {ocr_text}
        >>>
        """
        
        correction_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                                        "You are an assistant that only corrects obvious OCR errors. Do not change any sentence structure or reword anything." 
                                        "Only fix things like broken words, missing spaces, and random characters caused by OCR. "
                                        "If you're unsure whether something is an OCR error, leave it unchanged. Preserve the original meaning and order of all words and sentences."

                                    )},
                {"role": "user", "content": correction_prompt}
            ],
            temperature=0.2,
        )
        cleaned_text = correction_response.choices[0].message.content.strip()

        

        # Now send cleaned essay to essay examiner
        assessment_prompt = f"""
        The following is a CSS essay written by a candidate.
        
        You are a strict but constructive CSS examiner. Your tasks are:
        
        1. Assign a score (1–10) based on official CSS English Essay evaluation criteria: clarity, coherence, argumentation, grammar, structure, and style.
        
        2. Provide detailed and didactic feedback. Clearly explain what was weak in the original essay, **why** it was weak, and **how** it negatively affected the score. Be specific when discussing grammar, sentence construction, transitions, vocabulary use, structure, or logical flow.
        
        3. Rewrite the **first 100 words** of the essay to reflect a version that would score 10/10.
           - **Stay as close as possible** to the student’s original **wording, phrasing, and sentence structure**.
           - Only make changes that are **strictly necessary** to improve grammar, punctuation, clarity, coherence, and flow.
           - Do **not** substitute the student’s expressions with completely different vocabulary or ideas unless **absolutely necessary** for clarity or correctness.
           - **Highlight all improvements by wrapping the changed or corrected parts in double asterisks (i.e., bold)** so the student can easily compare the revision with their original writing.
        
        4. After the rewrite, **justify your changes** in a clear, teaching tone. Focus on:
           - Why each change was necessary,
           - How it improves clarity, grammar, or coherence,
           - And how it **preserves the student’s voice and intent** while enhancing the overall quality.
        
        Respond using the following format:
        
        **Score:** <your score here>/10
        
        **Feedback:**
        <Explain what is weak, why it is weak, and how it impacted the score. Focus on specifics such as clarity, logic, transitions, grammar, or vocabulary.>
        
        **Ideal Rewrite (First 100 Words):**
        <Your revised version of the first 100 words of the essay, with all changes clearly highlighted using bold (surround changes with **double asterisks**).>
        
        **Justification of Changes:**
        <Explain, like a teacher, the reasons behind your edits. Emphasize clarity, tone, structure, and logic. Show the student how the revision improves the essay without overwriting their personal style or meaning.>
        
        Essay:
        {cleaned_text}
        """

        assessment_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a CSS essay examiner providing feedback and ideal writing samples."},
                {"role": "user", "content": assessment_prompt}
            ],
            temperature=0.3,
        )
        assessment = assessment_response.choices[0].message.content.strip()
        
        # Extract model rewrite section (first 100 words)
        model_rewrite_match = re.search(
            r"\*\*Ideal Rewrite \(First 100 Words\):\*\*\s*(.+)", 
            assessment, 
            re.DOTALL
        )
        model_rewrite = model_rewrite_match.group(1).strip() if model_rewrite_match else ""


        # Highlight differences
        #text1, text2 = highlight_sentence_differences(cleaned_text, model_rewrite)
        
        print("")

        return {
            "text1": cleaned_text,         # cleaned essay with red highlights
            "text2": model_rewrite,         # improved essay with green highlights
            "assessment": assessment  # feedback and score
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/evaluate-assignment")
async def evaluate_assignment(
    session_id: str = Form(...),
    recipient_email: str = Form(...)
):
    try:
        print("\n[DEBUG] ---------------- New Evaluation Request ----------------")
        print(f"[DEBUG] Session ID received: {session_id}")
        print(f"[DEBUG] Recipient email received: {recipient_email}")

        # Validate session
        if session_id not in sessions:
            print(f"[ERROR] Invalid session ID: {session_id}")
            return JSONResponse({"error": "Invalid session"}, status_code=400)

        # --- Normalize session data (supports both list or dict storage) ---
        raw = sessions[session_id]
        if isinstance(raw, dict):
            chat_history = raw.get("messages", [])
            print("[DEBUG] Session storage detected: dict with 'messages' key")
        elif isinstance(raw, list):
            chat_history = raw
            print("[DEBUG] Session storage detected: plain list of messages")
        else:
            print(f"[ERROR] Unexpected session data type: {type(raw)}")
            return JSONResponse({"error": "Corrupt session data"}, status_code=500)

        # --- Extract the latest USER message as the assignment text ---
        assignment_text = ""
        for msg in reversed(chat_history):
            try:
                if msg.get("role") == "user":
                    assignment_text = msg.get("content", "")
                    if assignment_text:
                        break
            except AttributeError:
                # In case a non-dict item sneaks into the list
                continue

        print(f"[DEBUG] Chat history messages count: {len(chat_history)}")
        print(f"[DEBUG] Assignment text length (last user msg): {len(assignment_text)} characters")

        if not assignment_text.strip():
            print("[ERROR] No user assignment text found in session")
            return JSONResponse({"error": "No student assignment found in session"}, status_code=400)

        # --- Construct authorship-only evaluation prompt ---
        evaluation_prompt = f"""
        You are an academic examiner. Your job is to award marks strictly according to the 2×2 mark scheme below.
        
        Marking scheme:
        - Up to 2 marks per feature of laboratory experiments.
        - 1 mark for correctly identifying a feature.
        - +1 mark (total 2) if the student also clearly describes/explains that feature.
        - Maximum of 2 distinct features may be credited.
        - Maximum total = 4 marks.
        
        Instructions:
        1. Extract the student’s points from their answer.
        2. Match each point against the mark scheme.
        3. Award marks mechanically: identify = 1, identify+describe = 2.
        4. Do not give more than 2 marks per feature or more than 4 overall.
        5. Be strict: if description is vague or absent, do not award the second mark.
        
        Assignment (student answer):
        <<< BEGIN TEXT >>>
        {assignment_text}
        <<< END TEXT >>>
        
        Now output in this exact format:
        
        Extracted Features:
        - [Feature 1: …] → [Marks awarded: X/2]
        - [Feature 2: …] → [Marks awarded: X/2]
        
        Total Mark: [X/4]
        Reasoning: [short explanation of why marks were given or withheld]
        """

        print("[DEBUG] Evaluation prompt constructed successfully")

        # --- Call OpenAI (authorship-only, low temperature) ---
        print("[DEBUG] Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant providing a concise authorship evaluation only."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.2
        )
        print("[DEBUG] Response received from OpenAI API")

        # --- Extract assessment safely ---
        try:
            assessment = response.choices[0].message.content.strip()
        except Exception:
            print("[ERROR] Failed to extract assessment from OpenAI response")
            print("[DEBUG] Raw OpenAI response:", response)
            return JSONResponse({"error": "LLM response parsing error"}, status_code=500)

        print("[DEBUG] Authorship assessment extracted successfully")
        print(f"[DEBUG] Assessment preview: {assessment[:200]}{'...' if len(assessment) > 200 else ''}")

        # --- Compose minimal email (authorship verdict ONLY) ---
        subject = "Assignment Authorship Evaluation"
        body = f"""
        <h2>Assignment Authorship Evaluation</h2>
        <p><strong>Verdict & Rationale:</strong></p>
        <pre style="white-space: pre-wrap; font-family: inherit;">{assessment}</pre>
        <p style="color:#666; font-size: 12px; margin-top: 16px;">
          Note: This is an AI-assisted authorship signal and should be used alongside teacher judgment and any formal plagiarism tools.
        </p>
        """.strip()
        print("[DEBUG] Email body prepared successfully")

        # --- Send email ---
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        print(f"[DEBUG] Email message constructed: Subject='{subject}', From='{SMTP_USER}', To='{recipient_email}'")

        print(f"[DEBUG] Connecting to SMTP server {SMTP_HOST}:{SMTP_PORT}...")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            print("[DEBUG] SMTP connection secured with STARTTLS")
            server.login(SMTP_USER, SMTP_PASS)
            print("[DEBUG] SMTP login successful")
            server.send_message(msg)
        print(f"[DEBUG] Email sent successfully to {recipient_email}")

        print("[DEBUG] ---------------- Evaluation Request Completed ----------------\n")
        return {"reportMessage": "✅ Evaluation report has been sent to the teacher."}

    except Exception as e:
        print("[ERROR] Exception occurred during evaluation process")
        traceback.print_exc()
        return JSONResponse({"error": "Internal server error"}, status_code=500)
        

@app.post("/extract_text_essayChecker")
async def extract_text_essay_checker(
    images: List[UploadFile] = File(...),
    user_message: str = Form("")   # ← add this to receive the tutor’s note
):
    print("[DEBUG] Received request to /extract_text_essayChecker")
    print(f"[DEBUG] Number of images received: {len(images)}")
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    try:
        # 1. OCR each image and accumulate text
        all_pages_text = []
        for idx, image_file in enumerate(images, start=1):
            try:
                print(f"[DEBUG] Processing image #{idx}: filename='{image_file.filename}'")
                image_bytes = await image_file.read()
                print(f"[DEBUG]   → Read {len(image_bytes)} bytes from '{image_file.filename}'")

                vision_image = vision.Image(content=image_bytes)
                ocr_response = client_google_vision_api.document_text_detection(image=vision_image)
                print(f"[DEBUG]   → Google Vision OCR call completed for '{image_file.filename}'")

                page_text = ""
                if ocr_response is not None and ocr_response.full_text_annotation:
                    page_text = ocr_response.full_text_annotation.text
                    print(f"[DEBUG]   → Extracted {len(page_text)} characters of text from '{image_file.filename}'")
                else:
                    print(f"[WARN]   → No text found in OCR for '{image_file.filename}'")

                all_pages_text.append(page_text)
            except Exception as ocr_err:
                print(f"[ERROR] OCR error on image #{idx} ('{image_file.filename}'): {ocr_err}")
                traceback.print_exc()
                all_pages_text.append("")  # Append empty so index alignment remains

        # Combine pages into one essay string
        combined_essay_text = "\n\n".join(all_pages_text).strip()
        print(f"[DEBUG] Combined essay text length: {len(combined_essay_text)} characters")

        # First clean OCR errors using GPT
        correction_prompt = f"""
        You are given text extracted from images using OCR. Your task is to fix only clear OCR-related errors such as:
        
        - Missing spaces between words  
        - Words that were incorrectly split or joined  
        - Obvious character recognition mistakes (e.g., '1' instead of 'I', or '0' instead of 'O')  
        
        ⚠️ Do NOT paraphrase, reword, or change the sentence structure.  
        ⚠️ Preserve the original wording and order exactly, unless a change is clearly required to fix an OCR issue.
        
        <<< BEGIN ESSAY TEXT >>>
        {combined_essay_text}
        <<< END ESSAY TEXT >>>
        """
        
        correction_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                                        "You are an assistant that only corrects obvious OCR errors. Do not change any sentence structure or reword anything. "
                                        "Only fix things like broken words, missing spaces, and random characters caused by OCR. "
                                        "If you're unsure whether something is an OCR error, leave it unchanged. Preserve the original meaning and order of all words and sentences."

                                    )},
                {"role": "user", "content": correction_prompt}
            ],
            temperature=0.2,
        )
        
        # Extract the cleaned essay text
        combined_essay_text = correction_response.choices[0].message.content.strip()
        
        print(f"[DEBUG] combined essay text length: {len(combined_essay_text)} characters")

        if not combined_essay_text:
            print("[ERROR] No OCR text found across all images.")
            return {
                "success": False,
                "message": "No OCR text found in any of the uploaded images."
            }

        # 2. Build the prompt for OpenAI
        prompt = f"""
        The following text is a combined CSS essay (across multiple pages) from a candidate.
        
        You are a strict but constructive CSS English essay examiner. Your tasks are:
        
        1. Assign a score from 1 to 10 based on official CSS essay evaluation criteria: clarity, coherence, grammar, structure, argumentation, and style.
        
        2. Rewrite the essay to a version that would receive a perfect 10/10 score.
           - Stay as close as possible to the student’s original wording, phrasing, tone, and structure.
           - Only make changes that are strictly necessary to improve grammar, punctuation, sentence clarity, paragraph coherence, logical flow, or eliminate repetition.
           - Do **not** replace the student’s ideas, voice, or expressions with different vocabulary or examples unless absolutely necessary for clarity or logic.
           - Do **not** rewrite just to sound more polished — only change what genuinely needs improvement.
           - **Wrap every changed word or phrase in double asterisks (`**`) so the student can clearly see the improvements.**
        
        3. Then, provide:
           - The **complete revised version** of the essay with all changes bolded using `**` syntax.
           - The original essay’s **score out of 10**.
           - A **formal explanation** justifying the score, using academic language to cite specific strengths and weaknesses.
        
        Essay text (all pages combined):
        {combined_essay_text}
        
        Your detailed assessment:
        """
        print("[DEBUG] Prompt built for OpenAI (first 300 chars):")
        print(prompt[:300] + ("..." if len(prompt) > 300 else ""))

        # 3. Send to OpenAI for evaluation
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledgeable and strict CSS essay examiner who evaluates essays "
                            "according to the CSS exam standards of Pakistan. You provide detailed feedback, "
                            "scoring, and constructive advice for improvement."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            print("[DEBUG] Received OpenAI response")
        except Exception as openai_err:
            print(f"[ERROR] OpenAI API call failed: {openai_err}")
            traceback.print_exc()
            return {
                "success": False,
                "message": "OpenAI API call failed. See server logs for details."
            }

        try:
            assessment = response.choices[0].message.content.strip()
            print(f"[DEBUG] Assessment length: {len(assessment)} characters")
        except Exception as parse_err:
            print(f"[ERROR] Failed to parse OpenAI response: {parse_err}")
            traceback.print_exc()
            return {
                "success": False,
                "message": "Failed to parse OpenAI response."
            }

        # 4. Email the assessment
        recipient_email = "mshahrukhhaider@gmail.com"
        #recipient_email = "proactive1@hotmail.com"
        subject = "Your Combined CSS Essay Assessment"
        body = f"""
            <h2>Your CSS Essay Assessment</h2>
            <p>Dear Candidate,</p>
        
            <p><strong>Note from Student/Tutor:</strong><br/>
            {user_message}
            </p>
        
            <p><strong>Combined Essay Text (from all uploaded pages):</strong></p>
            <pre style="white-space: pre-wrap;">{combined_essay_text}</pre>
        
            <p><strong>Detailed Assessment:</strong></p>
            <pre style="white-space: pre-wrap;">{assessment}</pre>
        
            <p>Regards,<br/>CSS Essay Checker Bot</p>
        """


        try:
            print(f"[DEBUG] Preparing to send email to {recipient_email}")
            msg = MIMEMultipart()
            msg['From'] = SMTP_USER
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                print(f"[DEBUG] Connecting to SMTP server {SMTP_HOST}:{SMTP_PORT}")
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
            print(f"[DEBUG] Email sent successfully to {recipient_email}")
        except Exception as email_err:
            print(f"[ERROR] Failed to send email: {email_err}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Assessment generated but failed to send email: {email_err}"
            }

        # All steps succeeded
        return {
            "success": True,
            "message": "Assessment completed and emailed successfully.",
            "email_sent_to": recipient_email
        }

    except Exception as e:
        # Catch any unforeseen errors
        print(f"[ERROR] Unhandled exception in /extract_text_essayChecker: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")
@app.post("/generate-campaign", response_model=CampaignResponse)
def generate_campaign(request: CampaignRequest, db: Session = Depends(get_db)):
    if not request.campaignName or not request.goal:
        raise HTTPException(status_code=400, detail="Missing required fields")

    print(f"[DEBUG] Generating campaign for: {request.campaignName}")
    print(f"[DEBUG] Goal: {request.goal}, Tone: {request.tone}")
    print(f"[DEBUG] Doctor Info: id={request.doctorData.id}, name={request.doctorData.name}")

    # 1. Call OpenAI to generate suggestions
    try:
        prompt = f"""
You are an expert AI marketing assistant. Generate exactly 5 social media post suggestions for a campaign.

Campaign Name: {request.campaignName}
Goal: {request.goal}
Tone: {request.tone}
Doctor Info: {request.doctorData.dict()}

Important:
- Return the output as a valid JSON array of strings ONLY.
- Example: ["Post idea 1", "Post idea 2", "Post idea 3", "Post idea 4", "Post idea 5"]
- Do NOT include any extra text outside the JSON array.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI marketing assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        suggestions_text = response.choices[0].message.content.strip()
        print(f"[DEBUG] Raw suggestions from OpenAI: {suggestions_text}")
        suggestions = json.loads(suggestions_text)

        if not isinstance(suggestions, list):
            raise ValueError("Invalid response format from OpenAI.")
    except Exception as e:
        print(f"[ERROR] OpenAI generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {e}")

    # 2. Save campaign first
    try:
        campaign = Campaign(
            campaign_name=request.campaignName,
            goal=request.goal,
            tone=request.tone,
            doctor_id=request.doctorData.id,
            doctor_name=request.doctorData.name,
            suggestions=json.dumps(suggestions)
        )
        db.add(campaign)
        db.commit()
        db.refresh(campaign)
        print(f"[DEBUG] Campaign saved with ID: {campaign.id}")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to save campaign: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save campaign: {e}")

    # 3. Save suggestions
    try:
        for s in suggestions:
            suggestion_entry = CampaignSuggestion2(
                campaign_id=campaign.id,
                content=s,
                status="pending",
                user_id=request.doctorData.id 
            )
            db.add(suggestion_entry)
        db.commit()
        print(f"[DEBUG] Suggestions saved for campaign {campaign.id}")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to save suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save suggestions: {e}")

    # 4. Return a valid response
    return {
        "campaignId": campaign.id,   # use campaign, not new_campaign
        "message": "Campaign created. Suggestions pending approval."
    }

@app.get("/queue/posts")
def get_all_pending_suggestions(user_id: int, db: Session = Depends(get_db)):
    suggestions = (
        db.query(CampaignSuggestion2)
        .filter_by(status="pending", user_id=user_id)
        .all()
    )
    result = [
        {
            "id": s.id,
            "content": s.content,
            "campaignId": s.campaign_id,
            "userId": s.user_id
        }
        for s in suggestions
    ]
    print(f"DEBUG: Approvals endpoint result for user {user_id}:", result)  # 👈 debug print
    return result
@app.get("/campaigns/{campaign_id}/suggestions/pending")
def get_pending_suggestions(campaign_id: int, db: Session = Depends(get_db)):
    suggestions = db.query(CampaignSuggestion2).filter_by(
        campaign_id=campaign_id,
        status="pending"
    ).all()
    return [{"id": s.id, "content": s.content} for s in suggestions]

@app.post("/suggestions/{suggestion_id}/approve")
def approve_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    suggestion = db.get(CampaignSuggestion2, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    suggestion.status = "approved"
    db.commit()
    return {"success": True}

@app.post("/suggestions/{suggestion_id}/reject")
def reject_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    suggestion = db.get(CampaignSuggestion2, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    suggestion.status = "rejected"
    db.commit()
    return {"success": True}
    
    
@app.post("/solve_math_problem")
async def solve_math_problem(image: UploadFile = File(...)):
    # Validate file type
    if not image.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"detail": "Invalid file type. Please upload an image."})

    try:
        image_bytes = await image.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"detail": "Uploaded file is empty."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Could not read uploaded file. Error: {str(e)}"})

    # Send to Google Vision API
    try:
        image_content = vision.Image(content=image_bytes)
        response = client_google_vision_api.text_detection(image=image_content)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Google Vision API error: {str(e)}"})

    if response.error.message:
        return JSONResponse(status_code=500, content={"detail": f"Vision API error: {response.error.message}"})

    text_annotations = response.text_annotations
    if not text_annotations:
        return {"problem": "", "solution": "No text found in image."}

    raw_text = text_annotations[0].description
    print(f"[DEBUG] OCR Extracted:\n{raw_text}")

    # Clean OCR text lines
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    print(f"[DEBUG] OCR Lines: {lines}")

    # Handle fraction if exactly two lines (numerator / denominator)
    if len(lines) == 2:
        numerator = lines[0]
        denominator = lines[1]
        cleaned_expr = f"({numerator})/({denominator})"
    else:
        # Otherwise join all lines without spaces (or adjust as needed)
        cleaned_expr = ''.join(lines)

    print(f"[DEBUG] Expression after combining numerator and denominator: {cleaned_expr}")

    # Fix implicit multiplication like '2xy' -> '2*x*y'
    fixed_expr = fix_implicit_multiplication(cleaned_expr)
    print(f"[DEBUG] Fixed expression after implicit multiplication fix: {fixed_expr}")

    try:
        expr = sympy.sympify(fixed_expr)
        simplified = sympy.simplify(expr)
        solution = str(simplified)
    except Exception as e:
        return {
            "problem": raw_text.strip(),
            "solution": f"❌ Error simplifying expression: {e}"
        }

    return {
        "problem": raw_text.strip(),
        "solution": f"Simplified result: {solution}"
    }

@app.post("/api/upload")
async def upload_pdfs(pdfs: Union[UploadFile, List[UploadFile]] = File(...)):
    # Normalize to list even if a single file is uploaded
    if not isinstance(pdfs, list):
        pdfs = [pdfs]

    uploaded_files = []
    delete_previous_pdf_in_aws()

    for pdf in pdfs:
        if not pdf.filename.strip().lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{pdf.filename} is not a PDF")

        try:
            safe_filename = sanitize_filename(pdf.filename)
            key = f"upload/{safe_filename}"
            s3.upload_fileobj(pdf.file, BUCKET_NAME, key)
            file_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
            uploaded_files.append(file_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed for {pdf.filename}: {str(e)}")

    return JSONResponse(content={"message": "Files uploaded successfully", "files": uploaded_files})

@app.post("/api/get-sales-report", response_model=List[SalesResponse])
async def get_sales_report(
    request: SalesRequest,
    db: Session = Depends(get_db)
):
    try:
        # Validate dates
        if request.start_date > request.end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date.")

        # Query database
        sales = (
            db.query(Sale)
            .filter(Sale.date >= request.start_date, Sale.date <= request.end_date)
            .order_by(Sale.date)
            .all()
        )

        return [
            SalesResponse(
                bill_number=sale.bill_number,
                date=sale.date,
                total=sale.total
            )
            for sale in sales
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/send-email-ShahRukh")
async def send_email(
    name: str = Form(...),
    email: str = Form(...),
    message: str = Form(...)
):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'New Contact Form Submission'
        msg['From'] = SMTP_USER
        msg['To'] = 'mshahrukhhaider@gmail.com'
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


@app.post("/api/reservation")
async def make_reservation(reservation: ReservationBase, db: Session = Depends(get_db_aws)):
    try:
        new_reservation = ReservationModel_new(
            table_id=reservation.table_id,
            customer_name=reservation.customer_name,
            customer_contact=reservation.customer_contact,
            date=reservation.date,
            time_slot=reservation.time_slot,
            status=reservation.status,
        )
        db.add(new_reservation)
        db.commit()
        db.refresh(new_reservation)

        prompt = (
            f"A new reservation record was received with these details:\n"
            f"  • Customer Name: {reservation.customer_name}\n"
            f"  • Contact: {reservation.customer_contact}\n"
            f"  • Date: {reservation.date}\n"
            f"  • Time: {reservation.time_slot}\n"
            f"  • Table Number: {reservation.table_id}\n"
            f"  • Status: {reservation.status}\n\n"
            "Generate **five** question-and-answer pairs that reference the customer by name. "
            "Use exactly this phrasing pattern for at least these questions:\n"
            "  1. Did [Customer] book a reservation?\n"
            "  2. At what time did [Customer] book a reservation?\n"
            "  3. What is the status of [Customer]’s reservation?\n"
            "  4. What are [Customer]’s contact details?\n"
            "  5. If any detail is missing (e.g., party size), note how to obtain it.\n\n"
            "**If the answer to any question is not available in the context above, reply with 'I do not know.'**\n\n"
            "Respond in this exact format, one pair per line:\n"
            "Q1: <question text>\n"
            "A1: <answer text>\n"
            "Q2: <question text>\n"
            "A2: <answer text>\n"
            "…\n"
            "Q5: <question text>\n"
            "A5: <answer text>"
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that turns reservation data into management FAQs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        faq_text = response.choices[0].message.content.strip()

        pairs = re.findall(r'(Q\d:\s*.+?)\n(A\d:\s*.+?)(?=\nQ\d:|\Z)', faq_text, re.S)

        for q_label, a_label in pairs:
            question = q_label.split(":", 1)[1].strip()
            answer = a_label.split(":", 1)[1].strip()

            sentences = re.split(r'(?<=[.!?])\s+', answer)
            if len(sentences) <= 2:
                sentences = [answer]

            for sentence in sentences:
                if not sentence.strip():
                    continue

                embedding = client.embeddings.create(
                    input=sentence,
                    model="text-embedding-ada-002"
                ).data[0].embedding

                new_faq = FAQModel(
                    question=question,
                    answer=sentence,
                    embedding=embedding
                )
                db.add(new_faq)

        db.commit()
        return {"message": "Reservation details added successfully!", "reservation_id": new_reservation.id}

    except Exception as e:
        db.rollback()
        print(f"Error adding reservation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add reservation: {e}")
    
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
#meant to be used for all pizza restaurants
@app.post("/api/sendorder_pizzapoint")
async def send_order_pizzapoint(
    order: OrderDataPizzaPoint,
    db: Session = Depends(get_db),
):
    # --- 1) Build and send WhatsApp message ---
    try:
        print("Building message body...")

        lines = [
            "*🍕 New Pizza Point Order*",
            f"🏬 Restaurant: {order.restaurant_name}",
            f"📞 Customer Phone: {order.phone}",
            f"🕒 Timestamp: {order.timestamp}",
            "",
            "*🛒 Order Items:*"
        ]

        item_lines = []
        for item in order.items:
            print(f"Processing item: {item.name}, quantity: {item.quantity}, price: {item.price}")
            line = f"- {item.name} — {item.quantity} × Rs.{item.price:.0f} = Rs.{item.quantity * item.price:.0f}"
            lines.append(line)
            item_lines.append(f"{item.name} x{item.quantity} @Rs.{item.price}")

        lines.append("")
        lines.append(f"*💰 Total: Rs.{order.total:.0f}*")

        message_body = "\n".join(lines)
        print(f"Message body: {message_body}")

        print(f"Twilio number: {twilio_number}")
        print(f"Recipient number: {pizzapoint_number}")

        message = client_twilio.messages.create(
            body=message_body,
            from_=twilio_number,
            to=pizzapoint_number
        )

        print(f"Message SID: {message.sid}")

    except Exception as e:
        print(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending WhatsApp message: {str(e)}")

    # --- 2) Log usage in railway_usage ---
    today = date.today()
    
    vendor = order.restaurant_name
    
    
    print(f"Logging usage for {vendor} on {today}")

    usage = (
        db.query(RailwayUsage)
        .filter(RailwayUsage.date == today, RailwayUsage.vendor_name == vendor)
        .first()
    )

    if usage:
        usage.api_calls += 1
        print(f"Updated usage for {vendor}, API calls: {usage.api_calls}")
    else:
        usage = RailwayUsage(
            date=today,
            vendor_name=vendor,
            api_calls=1
        )
        db.add(usage)
        print(f"New usage entry created for {vendor}")

    # --- 3) Save Order to DB ---
    
    order_record = PizzaOrder(
    restaurant_name=order.restaurant_name,
    phone=order.phone,
    timestamp=order.timestamp,
    items="; ".join(item_lines),
    total=order.total,
    image_url=order.items[0].image_url  # ✅ use first item
)
    db.add(order_record)
    print("Pizza order saved to DB.")
    order_timestamp = datetime.strptime(order.timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")


    # Save to SalePizzaPoint (sales reporting)
    sale_record = SalePizzaPoint(
        bill_number=f"BILL-{int(datetime.now().timestamp())}",  # or however you generate bill numbers
        date=order_timestamp.date(),  # Ensure it's a `date` object
        total=order.total
    )
    db.add(sale_record)
    print("Sales record saved to DB.")
    
    db.commit()
    print("Database commit complete.")

    return {
        "success": True,
        "message": "Order sent via WhatsApp",
        "sid": message.sid,
        "usage": {
            "date": today.isoformat(),
            "vendor_name": vendor,
            "api_calls": usage.api_calls
        }
    }

@app.get("/view-sales", response_model=List[SalesResponse])
async def view_sales(
    start_date: date = Query(..., description="The start date of the sales report"),
    end_date: date = Query(..., description="The end date of the sales report"),
    db: Session = Depends(get_db),
):
    try:
        sales = db.query(SalePizzaPoint).filter(
            SalePizzaPoint.date >= start_date,
            SalePizzaPoint.date <= end_date
        ).all()

        if not sales:
            raise HTTPException(status_code=404, detail="No sales found for the given date range")

        return sales

    except HTTPException:
        raise  # Allow FastAPI to handle known HTTP errors

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
#this end point is for orders that is given by using the items selected through multilistbox(hajvery milk)
@app.post("/api/sendorderText")
async def send_order_hajvery(
    order: OrderDataHajvery,
    db: Session = Depends(get_db),
):
    # --- 1) Build and send WhatsApp message ---
    try:
        lines = [
            "*📦 New Hajvery Milk Shop Order*",
            f"🏠 Address / Vehicle: {order.customerInfo}",
            "",
            "*🛒 Cart Items:*"
        ]
        for item in order.cart:
            lines.append(
                f"- {item.name} — {item.quantity} × {item.price:.2f} = "
                f"{(item.quantity * item.price):.2f}"
            )
        lines.append("")
        lines.append(f"*💰 Total: {order.totalAmount:.2f}*")
        message_body = "\n".join(lines)

        message = client_twilio.messages.create(
            body=message_body,
            from_=twilio_number,
            to=hajvery_number
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WhatsApp send failed: {e}")

    # --- 2) Log usage in railway_usage ---
    today = date.today()
    vendor = order.vendorName

    # Try to find existing record for today & this vendor
    usage = (
        db.query(RailwayUsage)
          .filter(RailwayUsage.date == today, RailwayUsage.vendor_name == vendor)
          .first()
    )
    if usage:
        usage.api_calls += 1
    else:
        usage = RailwayUsage(
            date=today,
            vendor_name=vendor,
            api_calls=1
        )
        db.add(usage)

    db.commit()

    return {
        "success": True,
        "message": "Order sent via WhatsApp",
        "sid": message.sid,
        "usage": {
            "date": today.isoformat(),
            "vendor_name": vendor,
            "api_calls": usage.api_calls
        }
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

#meant to display orders for rafis kitchen only.
@app.get("/orders", response_model=List[OrderResponse])
def get_orders(restaurant_name: Optional[str] = Query(None), db: Session = Depends(get_db)):
    query = db.query(OrderModel).options(joinedload(OrderModel.items))
    
    # Filter by restaurant_name if provided
    if restaurant_name:
        query = query.filter(OrderModel.restaurant_name == restaurant_name)

    # Execute the query and fetch all orders
    orders = query.all()

    return orders
#meant to display orders for all pizza point type restaurants


@app.get("/orders_pizza_point", response_model=List[OrderResponsePizzaOrder])
def get_orders_pizza_point(
    restaurant_name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(PizzaOrder)
    
    if restaurant_name:
        query = query.filter(PizzaOrder.restaurant_name == restaurant_name)

    try:
        orders = query.all()
    except Exception as e:
        print("Error during query:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    parsed_orders = []
    for order in orders:
        parsed_items = []

        # Split by ';' and parse each item
        for item_str in order.items.split(';'):
            item_str = item_str.strip()
            if item_str:
                try:
                    name_part, price_part = item_str.split(' x')
                    quantity, price = price_part.split(' @Rs.')

                    # Use the same image_url for all items in the order
                    image_url = order.image_url

                    parsed_items.append({
                        "name": name_part.strip(),
                        "quantity": int(quantity.strip()),
                        "price": float(price.strip()),
                        "image_url": image_url  # Add the image_url here
                    })
                except Exception as parse_error:
                    print(f"Failed to parse item string: '{item_str}', error: {parse_error}")
                    continue

        # Create the OrderResponsePizzaOrder object with the parsed data
        parsed_orders.append(OrderResponsePizzaOrder(
            id=order.id,
            restaurant_name=order.restaurant_name,
            phone=order.phone,
            timestamp=order.timestamp,
            items=parsed_items,
            total=order.total
        ))

    return parsed_orders

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

#meant to be used for RAfis kitchen
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
#meant to delete orders from Pizza Point type restaurants
@app.delete("/delete-pizza-order/{order_id}")
def delete_pizza_order(order_id: int, db: Session = Depends(get_db)):
    order = db.query(PizzaOrder).filter(PizzaOrder.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Pizza order not found")

    db.delete(order)
    db.commit()

    return {"message": f"Pizza order {order_id} deleted successfully"}
# — OpenAI Chat Endpoints —
@app.post("/api/chatRK")
async def chat_rk(msg: Message):
    try:
        if not is_relevant_to_rafis_kitchen(msg.message):
            return {
                "reply": (
                    "I'm sorry, but I can only assist with questions related to Rafi's Kitchen at 800 Wayne Street, Olean, NY, "
                    "including menu, location, and services provided by Amir, the owner."
                )
            }

        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant for Rafi's Kitchen, a family-friendly seasonal restaurant located at 800 Wayne Street, Olean, NY 14760. "
                        "Rafi's Kitchen offers Mediterranean, Italian, Lebanese, and Pakistani cuisine, and is open from May to December. "
                        "The restaurant is owned by Amir. Your role is to answer questions strictly related to Rafi's Kitchen — including its menu, hours, services, amenities, and location. "
                        "Do not respond to questions unrelated to the restaurant or based on general knowledge outside of Rafi's Kitchen. "
                        "Answer accurately based on the restaurant’s official information. If the question is outside your scope, politely inform the user that you can only answer questions about Rafi's Kitchen."
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


@app.get("/api/faqs", response_model=list[FAQOut])
def get_faqs(db: Session = Depends(get_db_aws)):
    # Query only the needed columns
    faqs = db.query(FAQModel.id, FAQModel.question, FAQModel.answer).all()

    # Map tuples to dicts (or to FAQOut if FAQOut supports ORM mode)
    result = [{"id": id, "question": question, "answer": answer} for id, question, answer in faqs]

    return result
@app.post("/api/chat_database", response_model=ChatResponse)
async def chat_with_database(request: ChatRequest, db: Session = Depends(get_db_aws)):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    # 1. Generate embedding for user query
    query_embedding = get_embedding(user_message)  # e.g., returns List[float]

    # 2. Query top 3 FAQs by vector similarity with explicit cast to vector
    faqs = db.execute(
        text("""
            SELECT id, question, answer, embedding <=> CAST(:embedding AS vector) AS distance
            FROM faqs
            ORDER BY distance
            LIMIT 3
        """),
        {"embedding": query_embedding}
    ).fetchall()
    # 3. Prepare context from top FAQs
    context_text = "\n\n".join([f"Q: {faq.question}\nA: {faq.answer}" for faq in faqs])
    print("Context passed to the model:\n", context_text)

    # 4. Compose prompt with context + user question
    prompt = f"Use the following FAQ knowledge base to answer the user question.\n\n{context_text}\n\nUser Question: {user_message}\nAnswer:"

    # 5. Call OpenAI chat completion
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers user questions using the provided FAQ knowledge base. If the answer is not explicitly stated, try to infer from the given information but do not make up unsupported facts."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()

    return ChatResponse(reply=answer)



class AIRequest(BaseModel):
    message: str
@app.get("/api/get-ai-sentence")
async def get_ai_sentence():
    print("Received request for AI sentence...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a teacher for O-Level students. "
                        "Your task is to generate exactly ONE grammatically correct sentence suitable for O-Level English students to practice paraphrasing. "
                        "The sentence must contain only one sentence—do NOT generate multiple sentences or examples. "
                        "Make it slightly complex by including at least one subordinate clause or descriptive element. "
                        "It should be 2-3 lines long, but still understandable by O-Level students. "
                        "DO NOT include numbering, bullet points, or extra sentences. Strictly one sentence only."
                    )
                },
                {
                    "role": "user",
                    "content": "Please generate a sentence to paraphrase."
                }
            ],
            temperature=0.3
        )

        sentence = response.choices[0].message.content.strip()
        print("Generated sentence:", sentence)
        return {"sentence": sentence}

    except Exception as e:
        print("Error generating sentence:", e)
        return {"sentence": "Failed to generate sentence."}

@app.post("/api/evaluate-paraphrase")
async def evaluate_paraphrase(request: ParaphraseRequest):
    print("Received paraphrase evaluation request:", request)

    try:
        # Construct the system message for GPT
        system_prompt = (
            f"You are an English teacher for {request.level} students. "
            "Evaluate the student's paraphrase based on clarity, grammar, and faithfulness to the original sentence. "
            "Provide concise feedback and suggest improvements if necessary. "
            "Additionally, generate 3 alternative paraphrased versions of the original sentence that the student could use as examples."
        )
        print("System prompt sent to GPT:", system_prompt)

        # Call GPT to evaluate the paraphrase
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original sentence: {request.original}\nStudent paraphrase: {request.paraphrase}\nPlease provide evaluation and suggestions."}
            ],
            temperature=0.2
        )

        feedback = response.choices[0].message.content.strip()
        print("GPT feedback:", feedback)

        return {"feedback": feedback}

    except Exception as e:
        print("Error in evaluate_paraphrase:", e)
        raise HTTPException(status_code=500, detail="Failed to evaluate paraphrase.")




@app.post("/api/chatwebsite_ShahRukh")
async def chat_website(msg: Message):
    try:
        if not is_relevant_to_css_preparation(msg.message):
            return {
                "reply": (
                    "I'm sorry, but my training dictates that this is an irrelevant question. if you want your query answered. you can talk directly "
                    "with Mr Shah Rukh at his whats app number:+923214353162 or email him at mshahrukhhaider@gmail.com . i am sure he will be happy to answer that for you.."
                    "do you have another query i can help you with? "                    
                )
            }

        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.2,
            messages = [
                {
                    "role": "system",
                    "content": """
            You are a professional virtual assistant representing Shah Rukh, a successful CSS qualifier who now mentors aspiring candidates. 
            Shah Rukh has launched a website to share the proven strategies, resources, and personal mentorship that helped him crack the CSS exam. 
            The platform offers a range of services including AI-powered guidance through a chatbot, essay checking with personalized feedback, exclusive video lectures, study plans, and subscription-based access to premium content. 
            Your role is to engage users respectfully and help them understand how Shah Rukh's mentorship can improve their chances of CSS success. 
            Highlight the features of the website such as the AI chatbot for 24/7 support, the essay review system, flexible subscription plans, and expert advice directly from Shah Rukh. 
            Keep the conversation focused on CSS preparation, mentorship, and how the platform adds value. 
            If the user shows interest in joining, direct them to the subscription or contact page. 
            Avoid discussing topics unrelated to CSS exam preparation or website offerings. 
            Your goal is to guide students effectively and encourage them to benefit from Shah Rukh’s experience and resources.
            Format your responses with line breaks between sections or steps. Use markdown for bullet points and headings where helpful.

            """
                },
                {
                    "role": "user",
                    "content": msg.message,
                }
            ]

        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        logging.error(f"chatwebsite error: {e}")
        raise HTTPException(status_code=500, detail="Sorry, something went wrong.")

@app.post("/api/chatwebsite")
async def chat_website(msg: Message):
    try:
        if not is_relevant_to_programming(msg.message):
            return {
                "reply": (
                    "I'm sorry, but my training dictates that this is an irrelevant question. if you want your query answered. you can talk directly "
                    "with Mr Sajjad at his whats app number:+923004112884 or email him at proactive1.san@gmail.com . i am sure he will be happy to answer that for you.."
                    "do you have another query i can help you with? "                    
                )
            }

        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.2,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are my virtual assistant, responsible for engaging professionally with potential clients on my behalf. "
                        "I am Sajjad Ali Noor, a full stack developer specializing in Python for backend development and React for frontend applications. "
                        "I have extensive experience building and deploying production-ready software, including real-time apps, SaaS dashboards, and AI-integrated tools. "
                        "On the backend, I use Python (Flask/FastAPI/Django), create custom modules, and work with databases efficiently. "
                        "On the frontend, I build polished, responsive UIs using React. "
                        "I deploy backends on Railway and frontends on Vercel for fast, scalable, and reliable delivery. "
                        "I am skilled in automating workflows, integrating APIs, web scraping, and building systems that save time and increase business efficiency. "
                        "I also have experience with coding interview preparation, algorithmic problem solving, and AI-assisted solutions. "
                        "When responding to clients, emphasize my technical depth, ability to solve real-world problems, and focus on delivering measurable value. "
                        "Only accept projects that fit my expertise to ensure excellent results. "
                        "If the client wants to proceed, share my portfolio website (https://sajjadalinoor.vercel.app/) and my email: proactive1.san@gmail.com."
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
        logging.error(f"chatwebsite error: {e}")
        raise HTTPException(status_code=500, detail="Sorry, something went wrong.")

@app.post("/improve_expresion")
async def improve_expresion(req: TextRequest):
    try:
        print("Received text:", req.text.strip())

        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior communication coach helping software engineers prepare for coding interviews.\n\n"
                        "The user has written a 'think before the code' explanation — a description of their approach before writing code.\n\n"
                        "Your job is to:\n"
                        "1. Assess whether the explanation is 'interview-ready' — respond with Yes or No, and give a short reason.\n"
                        "2. Show exactly what the user wrote, in full.\n"
                        "3. Provide a line-by-line comparison:\n"
                        "    - Quote each sentence or step from the user's original explanation.\n"
                        "    - Then rewrite it to be more clear, professional, and technically sound.\n"
                        "    - If any part of the logic is incorrect, incomplete, or inefficient, improve it — explain what you changed and why.\n"
                        "4. Keep the revisions concise and realistic for an interview setting.\n"
                        "5. At the end, briefly summarize how your improvements help make the explanation more effective and interview-ready.\n\n"
                        "Your goal is to help the user sound both clear and competent — like someone who understands the problem and can explain their solution approach with confidence."
                    )
                },
                {
                    "role": "user",
                    "content": req.text.strip()
                }
            ]
        )

        improved_text = response.choices[0].message.content.strip()
        return {"improved_text": improved_text}

    except Exception as e:
        print(f"Error in /improve_expresion: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong while improving the text.")
    
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
    





















































































































































































































































