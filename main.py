import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
import random
import sqlite3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import time
import hashlib

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
WHATSAPP_TOKEN = os.getenv('WHATSAPP_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WEBHOOK_VERIFY_TOKEN = os.getenv('WEBHOOK_VERIFY_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', None)
HUMAN_CONTACT = os.getenv('HUMAN_CONTACT', '03328923360')

# HARDCODED PROPERTIES DATABASE with Multiple Images
PROPERTIES_FOR_SALE = [
    {
        "type": "House",
        "size": "1 Kanal",
        "location": "DHA Phase 5",
        "price": "19 Cr",
        "details": "Brand new, 6 beds, full basement, theater, swimming pool, near Jalal Sons and park",
        "id": "H001",
        "listing_type": "sale",
        "image_urls": [
            "https://ibb.co/Jjs2NCpY",
            "https://ibb.co/fYyWymLs",
            "https://ibb.co/MDJsB1RT"  # Replace with actual URLs or local paths
        ]
    },
    {
        "type": "House",
        "size": "1 Kanal",
        "location": "DHA Phase 6",
        "price": "11.50 Cr",
        "details": "Brand new, 7 beds, full basement, home theater, near mosque",
        "id": "H002",
        "listing_type": "sale",
        "image_urls": [
            "https://ibb.co/x98LhPh",
            "https://ibb.co/ccRmn4nS",
            "https://ibb.co/Rx49W3w"
        ]
    },
    {
        "type": "House",
        "size": "1 Kanal",
        "location": "DHA Phase 6",
        "price": "18 Cr",
        "details": "Brand new, 6 beds, full basement, furnished, near Dolmen Mall",
        "id": "H003",
        "listing_type": "sale",
        "image_urls": [
            "https://ibb.co/YFMjN5rn",
            "https://ibb.co/ycRnZLRP",
            "https://ibb.co/7JfrMm4M"
        ]
    },
    {
        "type": "House",
        "size": "1 Kanal",
        "location": "DHA Phase 6",
        "price": "20 Cr",
        "details": "Brand new, 6 beds, full basement, furnished, attached bathrooms",
        "id": "H004",
        "listing_type": "sale",
        "image_urls": [
            "https://ibb.co/LDRMfCQ9",
            "https://ibb.co/DfW9zh4z",
            "https://ibb.co/NdN6z8mw"
        ]
    },
    {
        "type": "House",
        "size": "1 Kanal",
        "location": "DHA Phase 7, T Block",
        "price": "8 Cr",
        "details": "Brand new, 5 beds, attached bathrooms, tiled flooring",
        "id": "H005",
        "listing_type": "sale",
        "image_urls": [
            "https://ibb.co/F4JsFQLf",
            "https://ibb.co/4ndZ6k5f",
            "https://ibb.co/1f5Zv8Zh"
        ]
    },
    {
        "type": "Plot",
        "size": "10 Marla",
        "location": "Central Park Housing Society, Block G, Plot #91",
        "price": "87 Lac",
        "details": "Possession plot, ready for construction",
        "id": "P001",
        "listing_type": "sale",
        "image_urls": [
            ""
        ]
    },
    {
        "type": "Plot",
        "size": "5 Marla",
        "location": "DHA 9 Town, A 1525",
        "price": "95 Lac",
        "details": "Facing park, DB pole clear, prime location",
        "id": "P002",
        "listing_type": "sale",
        "image_urls": [
            ""
        ]
    },
    {
        "type": "Plot",
        "size": "5 Marla",
        "location": "DHA 9 Town, E 1548",
        "price": "120 Lac",
        "details": "Facing park, DB pole clear, excellent investment",
        "id": "P003",
        "listing_type": "sale",
        "image_urls": [
            ""
        ]
    }
]

PROPERTIES_FOR_RENT = [
    # Add rent properties with multiple image URLs if needed
    # Example:
    # {
    #     "type": "House",
    #     "size": "10 Marla",
    #     "location": "DHA Phase 8",
    #     "rent": "2 Lac",
    #     "details": "Fully furnished, 4 beds, near market",
    #     "id": "R001",
    #     "listing_type": "rent",
    #     "image_urls": [
    #         "https://example.com/images/r001_1.jpg",
    #         "https://example.com/images/r001_2.jpg"
    #     ]
    # }
]

# Combined database for compatibility
PROPERTIES_DATABASE = PROPERTIES_FOR_SALE + PROPERTIES_FOR_RENT

# Message deduplication tracking - prevents duplicate processing
processed_message_ids = set()
MESSAGE_EXPIRY_MINUTES = 60  # Messages expire after 1 hour


def format_properties_for_context(properties_list, listing_type="all"):
    """Format properties into a string for AI context"""
    if not properties_list:
        return "No properties available in our current database."

    if listing_type == "sale":
        formatted = "PROPERTIES FOR SALE IN OUR DATABASE:\n\n"
    elif listing_type == "rent":
        formatted = "PROPERTIES FOR RENT IN OUR DATABASE:\n\n"
    else:
        formatted = "AVAILABLE PROPERTIES IN OUR DATABASE:\n\n"

    for i, prop in enumerate(properties_list, 1):
        property_emoji = "🏠" if prop['type'] == 'House' else "🏗️" if prop['type'] == 'Plot' else "🏢"
        formatted += f"{i}. {property_emoji} {prop['type']} - {prop['size']}\n"
        formatted += f"   📍 Location: {prop['location']}\n"

        if prop['listing_type'] == 'sale':
            formatted += f"   💰 Sale Price: {prop['price']}\n"
        else:
            formatted += f"   💵 Monthly Rent: {prop['rent']}\n"

        formatted += f"   ✨ Details: {prop['details']}\n"
        formatted += f"   🆔 Property ID: {prop['id']}\n"
        formatted += f"   📷 Images: {', '.join(prop['image_urls'])}\n\n"

    return formatted


def filter_properties_by_query(message, properties_list=None):
    """Filter properties based on user query and intent (buy/rent)"""
    message_lower = message.lower()

    # Determine intent first
    buy_keywords = ['buy', 'purchase', 'sale', 'khareedna', 'خریدنا']
    rent_keywords = ['rent', 'rental', 'kiraya', 'کرایہ', 'lease']

    intent = None
    if any(keyword in message_lower for keyword in buy_keywords):
        intent = 'sale'
        properties_list = PROPERTIES_FOR_SALE
    elif any(keyword in message_lower for keyword in rent_keywords):
        intent = 'rent'
        properties_list = PROPERTIES_FOR_RENT
    else:
        # If no clear intent, use all properties
        properties_list = properties_list or PROPERTIES_DATABASE

    filtered_properties = []

    # Extract keywords from message
    house_keywords = ['house', 'home', 'ghar']
    plot_keywords = ['plot', 'zameen', 'land']
    apartment_keywords = ['apartment', 'flat']
    location_keywords = ['dha', 'phase', 'central park', 'town', 'gulberg']

    for prop in properties_list:
        match_score = 0

        # Type matching
        if any(keyword in message_lower for keyword in house_keywords) and prop['type'].lower() == 'house':
            match_score += 3
        elif any(keyword in message_lower for keyword in plot_keywords) and prop['type'].lower() == 'plot':
            match_score += 3
        elif any(keyword in message_lower for keyword in apartment_keywords) and prop['type'].lower() == 'apartment':
            match_score += 3

        # Location matching
        if any(keyword in message_lower for keyword in location_keywords):
            if any(keyword in prop['location'].lower() for keyword in location_keywords if keyword in message_lower):
                match_score += 2

        # Price/rent range matching (basic)
        if intent == 'sale':
            if 'crore' in message_lower or 'cr' in message_lower:
                if 'price' in prop and 'cr' in prop['price'].lower():
                    match_score += 1
            elif 'lac' in message_lower or 'lakh' in message_lower:
                if 'price' in prop and 'lac' in prop['price'].lower():
                    match_score += 1
        elif intent == 'rent':
            if 'lac' in message_lower or 'lakh' in message_lower:
                if 'rent' in prop and 'lac' in prop['rent'].lower():
                    match_score += 1
            elif 'thousand' in message_lower or 'k' in message_lower:
                if 'rent' in prop and 'k' in prop['rent'].lower():
                    match_score += 1

        if match_score > 0 or intent is None:
            filtered_properties.append((prop, match_score))

    # Sort by match score and return top properties
    if filtered_properties:
        filtered_properties.sort(key=lambda x: x[1], reverse=True)
        return [prop[0] for prop in filtered_properties[:5]], intent
    else:
        return properties_list[:5], intent


# Validate required environment variables
required_vars = ['WHATSAPP_TOKEN', 'WHATSAPP_PHONE_NUMBER_ID', 'WEBHOOK_VERIFY_TOKEN']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    logger.warning("⚠️ Neither GROQ_API_KEY nor GEMINI_API_KEY found, using offline AI only")


def validate_whatsapp_token():
    """Validate WhatsApp token by making a test API call"""
    try:
        session = create_session_with_retries()
        response = session.get(
            f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}",
            headers={'Authorization': f'Bearer {WHATSAPP_TOKEN}'},
            timeout=10
        )
        response.raise_for_status()
        logger.info("✅ WhatsApp token validated successfully")
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Failed to validate WhatsApp token: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 401:
            logger.error("⚠️ 401 Unauthorized: Verify WHATSAPP_TOKEN in .env matches Meta for Developers access token")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error validating WhatsApp token: {str(e)}")


def init_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (phone TEXT, role TEXT, content TEXT, timestamp TEXT)''')
    conn.commit()
    c.execute('''CREATE TABLE IF NOT EXISTS user_state
                 (phone TEXT, intent TEXT, type TEXT, price_range TEXT, area TEXT, size TEXT, timestamp TEXT)''')
    conn.commit()
    # Add table for message deduplication
    c.execute('''CREATE TABLE IF NOT EXISTS processed_messages
                 (message_id TEXT PRIMARY KEY, timestamp TEXT, phone TEXT)''')
    conn.commit()
    conn.close()


init_db()


def create_session_with_retries():
    """Create session with SAFE retry configuration to prevent duplicate messages"""
    session = requests.Session()
    # FIXED: Only retry on network errors, NOT on HTTP errors that might cause duplicates
    retries = Retry(
        total=2,  # Reduced from 3 to minimize retry attempts
        backoff_factor=2,  # Increased backoff to prevent rapid retries
        status_forcelist=[502, 503, 504],  # REMOVED 429 and 500 to prevent duplicate sends
        allowed_methods=["GET"],  # CRITICAL: Only retry GET requests, never POST
        raise_on_status=True  # Raise exceptions instead of retrying
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def generate_message_id(phone, message_content, timestamp):
    """Generate unique message ID for deduplication"""
    content_hash = hashlib.md5(f"{phone}:{message_content}:{timestamp}".encode()).hexdigest()
    return f"{phone}_{content_hash}"


def is_message_processed(message_id):
    """Check if message was already processed"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT message_id FROM processed_messages WHERE message_id = ?", (message_id,))
    result = c.fetchone()
    conn.close()
    return result is not None


def mark_message_processed(message_id, phone):
    """Mark message as processed to prevent duplicates"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    current_time = datetime.now().isoformat()
    c.execute("INSERT OR REPLACE INTO processed_messages (message_id, timestamp, phone) VALUES (?, ?, ?)",
              (message_id, current_time, phone))

    # Clean up old processed messages (older than MESSAGE_EXPIRY_MINUTES)
    expiry_time = (datetime.now() - timedelta(minutes=MESSAGE_EXPIRY_MINUTES)).isoformat()
    c.execute("DELETE FROM processed_messages WHERE timestamp < ?", (expiry_time,))

    conn.commit()
    conn.close()


def add_contact_message(message):
    """Add contact info message to every response"""
    contact_msg = f"\n\nIf you want to talk to a sales agent contact: {HUMAN_CONTACT}"
    if contact_msg in message:
        message = message.replace(contact_msg, '')
    return message + contact_msg


def get_ai_response_groq(message, conversation_history=None, properties_context=""):
    if not GROQ_API_KEY:
        logger.warning("⚠️ Groq API key missing, skipping Groq API")
        return None
    try:
        business_context = f"""You are an AI assistant for Syed Real Estate, a leading agency in Lahore.

CRITICAL PROPERTY RULES:
- You can ONLY mention properties from the provided database below
- NEVER invent, create, or hallucinate any properties not in this list
- NEVER mix up sale prices with rent prices - they are completely different
- For SALE properties, mention the SALE PRICE only
- For RENT properties, mention the MONTHLY RENT only
- If asked about properties not in the database, say "Our agent can help you further in this matter, please contact the number given below or wait till our team reaches out to you"
- Always reference properties by their exact details from the database
- Use emojis when listing properties: 🏠 for houses, 🏢 for apartments, 🏗️ for plots

{properties_context}

RESPONSE RULES: Keep responses 25-30 words, concise, professional. Relate to property services. English default, simple basic Urdu(dont include any hindi) for Urdu input (ghar, zameen). Include emoji. 
For queries (buy, rent, house, plot): Ask type, general price range, area (buy) or location, size, type (rent). List only properties from the database above if any 2 of the details given.
For location: Disclose 28F 1st Floor, Commercial Area Sector F, DHA Phase 1, Lahore, 54792, timings 10 AM-7:30 PM. 
For emergencies (urgent): Suggest agent contact. Offer market updates. If asked about AI, say: 'I am Syed Real Estate's AI assistant.' No human tone."""

        if conversation_history is None:
            conversation_history = []

        messages = [
            {"role": "system", "content": business_context},
            *conversation_history[-4:],
            {"role": "user", "content": message}
        ]

        session = create_session_with_retries()
        response = session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7
            },
            timeout=15
        )

        response.raise_for_status()
        result = response.json()
        ai_response = result['choices'][0]['message']['content'].strip()
        logger.info("✅ Successfully used Groq API with property context")
        return ai_response

    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ Groq API error: {str(e)}")
        return None


def get_ai_response_gemini(message, conversation_history=None, properties_context=""):
    if not GEMINI_API_KEY:
        logger.warning("⚠️ GEMINI_API_KEY missing, skipping Gemini API")
        return None
    try:
        business_context = f"""You are an AI assistant for Syed Real Estate, a leading agency in Lahore.

CRITICAL PROPERTY RULES:
- You can ONLY mention properties from the provided database below
- NEVER invent, create, or hallucinate any properties not in this list
- NEVER mix up sale prices with rent prices - they are completely different
- For SALE properties, mention the SALE PRICE only
- For RENT properties, mention the MONTHLY RENT only
- If asked about properties not in the database, say "Our agent can help you further in this matter, please contact the number given below or wait till our team reaches out to you"
- Always reference properties by their exact details from the database
- Use emojis when listing properties: 🏠 for houses, 🏢 for apartments, 🏗️ for plots

{properties_context}

RESPONSE RULES: Keep responses 25-30 words, concise, professional. Relate to property services. English default, simple basic Urdu(dont include any hindi) for Urdu input (ghar, zameen). Include emoji. 
For queries (buy, rent, house, plot): Ask type, general price range, area (buy) or location, size, type (rent). List only properties from the database above if any 2 of the details given.
For location: Disclose 28F 1st Floor, Commercial Area Sector F, DHA Phase 1, Lahore, 54792, timings 10 AM-7:30 PM. 
For emergencies (urgent): Suggest agent contact. Offer market updates. If asked about AI, say: 'I am Syed Real Estate's AI assistant.' No human tone."""

        if conversation_history is None:
            conversation_history = []

        messages = [
            {"role": "system", "content": business_context},
            *conversation_history[-4:],
            {"role": "user", "content": message}
        ]

        session = create_session_with_retries()
        response = session.post(
            "https://api.google.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gemini-2.5-flash",
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7
            },
            timeout=15
        )

        response.raise_for_status()
        result = response.json()
        ai_response = result['choices'][0]['message']['content'].strip()
        logger.info("✅ Successfully used Gemini API with property context")
        return ai_response

    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ Gemini API error: {str(e)}")
        return None


def get_smart_offline_response(message, conversation_history=None):
    """Advanced offline AI-like responses using keyword matching and context"""
    message_lower = message.lower()

    context_keywords = []
    if conversation_history:
        for msg in conversation_history[-3:]:
            if msg['role'] == 'user':
                context_keywords.extend(msg['content'].lower().split())

    business_keywords = {
        'property': ['house', 'plot', 'buy', 'rent', 'جائیداد', 'گھر', 'زمین', 'خرید', 'کرایہ'],
        'location': ['location', 'جگہ'],
        'services': ['services', 'سروسز'],
        'emergency': ['urgent', 'فوری'],
    }

    def extract_state(phone):
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        c.execute(
            "SELECT intent, type, price_range, area, size FROM user_state WHERE phone = ? ORDER BY timestamp DESC LIMIT 1",
            (phone,))
        state = c.fetchone()
        conn.close()
        return state if state else (None, None, None, None, None)

    def save_state(phone, intent, type_, price_range, area, size):
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO user_state (phone, intent, type, price_range, area, size, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (phone, intent, type_, price_range, area, size, datetime.now().isoformat()))
        c.execute(
            "DELETE FROM user_state WHERE phone = ? AND rowid NOT IN (SELECT rowid FROM user_state WHERE phone = ? ORDER BY timestamp DESC LIMIT 1)",
            (phone, phone))
        conn.commit()
        conn.close()

    phone = conversation_history[0]['content'].split(':')[0] if conversation_history else "unknown"
    intent, type_, price_range, area, size = extract_state(phone)

    # Check if this is a property query
    if any(keyword in message_lower for keyword_list in business_keywords.values() for keyword in keyword_list):
        # Filter properties based on the message and intent
        filtered_properties, detected_intent = filter_properties_by_query(message, PROPERTIES_DATABASE)

        if filtered_properties and detected_intent:
            property_list = []
            for i, prop in enumerate(filtered_properties[:3], 1):  # Limit to top 3
                property_emoji = "🏠" if prop['type'] == 'House' else "🏗️" if prop['type'] == 'Plot' else "🏢"

                if prop['listing_type'] == 'sale':
                    price_info = f"Sale Price: {prop['price']}"
                else:
                    price_info = f"Monthly Rent: {prop['rent']}"

                property_list.append(
                    f"{i}. {property_emoji} {prop['type']} ({prop['size']}) - 📍 {prop['location']} - 💰 {price_info} - ✨ {prop['details']}"
                )

            intent_text = "for sale" if detected_intent == 'sale' else "for rent" if detected_intent == 'rent' else ""
            return f"Available properties {intent_text}: {' '.join(property_list)} 🏠"
        else:
            return "Please specify your requirements (buy/rent, house/plot, location, budget) to see our available properties 🏠"

    return "How can I help you with your property needs today? 🏠"


def get_ai_response(message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    # Filter properties based on the user message for better context
    relevant_properties, intent = filter_properties_by_query(message, PROPERTIES_DATABASE)

    # Determine listing type for context formatting
    listing_type = "all"
    if intent == 'sale':
        listing_type = "sale"
    elif intent == 'rent':
        listing_type = "rent"

    properties_context = format_properties_for_context(relevant_properties, listing_type)

    # Try Groq first with property context
    response = get_ai_response_groq(message, conversation_history, properties_context)
    if response:
        return add_contact_message(response)

    # Fall back to Gemini if Groq fails and Gemini key is available
    if GEMINI_API_KEY:
        response = get_ai_response_gemini(message, conversation_history, properties_context)
        if response:
            return add_contact_message(response)

    logger.warning("⚠️ Both Groq and Gemini APIs failed, using offline AI")
    fallback_response = get_smart_offline_response(message, conversation_history)
    return add_contact_message(fallback_response)


def send_whatsapp_message(to, message, image_urls=None):
    """Send message via WhatsApp Business API with optional images - FIXED to prevent unsolicited messages"""
    # CRITICAL: Prevent empty or unsolicited messages
    if not message and not image_urls:
        logger.warning("⚠️ Attempted to send empty message - prevented")
        return None

    if not to or len(to.strip()) == 0:
        logger.warning("⚠️ Attempted to send message without recipient - prevented")
        return None

    # FIXED: Use session without retries for POST requests to prevent duplicates
    session = requests.Session()
    try:
        url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            'Authorization': f'Bearer {WHATSAPP_TOKEN}',
            'Content-Type': 'application/json'
        }

        # Send text message first
        if message:
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": message}
            }
            logger.info(f"📤 Sending WhatsApp text message to {to}")
            response = session.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"✅ Text message sent successfully to {to}")

        # Send images if provided
        if image_urls and isinstance(image_urls, list):
            for image_url in image_urls:
                payload = {
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "image",
                    "image": {
                        "link": image_url,
                        "caption": f"Image for property ID: {image_urls.index(image_url) + 1}"
                    }
                }
                logger.info(f"📤 Sending WhatsApp image message to {to}")
                response = session.post(url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                logger.info(f"✅ Image message sent successfully to {to}")
                time.sleep(1)  # Add delay to avoid rate limiting

        return response.json() if 'response' in locals() else None

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP error sending WhatsApp message to {to}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 401:
            logger.error("⚠️ 401 Unauthorized: Verify WHATSAPP_TOKEN in .env matches Meta for Developers access token")
        # CRITICAL: Do not retry on HTTP errors to prevent duplicate messages
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error sending WhatsApp message to {to}: {str(e)}")
        # CRITICAL: Do not retry on network errors to prevent duplicate messages
        return None


def send_welcome_message_with_buttons(to):
    """Send welcome message with interactive buttons only if triggered by an incoming message"""
    if not to:
        logger.warning("⚠️ Attempted to send welcome message without recipient - prevented")
        return None

    # FIXED: Use session without retries for POST requests
    session = requests.Session()
    try:
        url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            'Authorization': f'Bearer {WHATSAPP_TOKEN}',
            'Content-Type': 'application/json'
        }
        welcome_text = f"Welcome to Syed Real Estate Lahore. Visit 28F 1st Floor, Sector F, DHA Phase 1, 10 AM-7:30 PM. How can we assist? 🏠"
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": welcome_text},
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": "buy", "title": "Buy"}},
                        {"type": "reply", "reply": {"id": "rent", "title": "Rent"}},
                        {"type": "reply", "reply": {"id": "our_location", "title": "Our Location"}}
                    ]
                }
            }
        }
        logger.info(f"📤 Sending welcome message with buttons to {to}")
        response = session.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ Welcome message with buttons sent to {to}")
        return response.json()

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP error sending button message to {to}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 401:
            logger.error("⚠️ 401 Unauthorized: Verify WHATSAPP_TOKEN in .env matches Meta for Developers access token")
        logger.info(f"🔄 Attempting fallback text message to {to}")
        fallback_response = send_whatsapp_message(to,
                                                  f"Welcome to Syed Real Estate Lahore. Visit 28F 1st Floor, Sector F, DHA Phase 1, 10 AM-7:30 PM.\n\nIf you want to talk to a sales agent contact: {HUMAN_CONTACT}")
        return fallback_response
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error sending button message to {to}: {str(e)}")
        logger.info(f"🔄 Attempting fallback text message to {to}")
        fallback_response = send_whatsapp_message(to,
                                                  f"Welcome to Syed Real Estate Lahore. Visit 28F 1st Floor, Sector F, DHA Phase 1, 10 AM-7:30 PM.\n\nIf you want to talk to a sales agent contact: {HUMAN_CONTACT}")
        return fallback_response


def handle_button_response(button_id, phone_number):
    """Handle button click responses"""
    responses = {
        "buy": f"Please specify property type (plot, commercial, house), general price range, and area. We will suggest options. 🏠",
        "rent": f"Please provide general location, size, and type (house or apartment). We will find suitable rentals. 🏠",
        "our_location": f"Syed Real Estate is at 28F 1st Floor, Commercial Area Sector F, DHA Phase 1, Lahore, 54792, 10 AM-7:30 PM. Visit us. 📍"
    }
    response_text = responses.get(button_id,
                                  f"Thank you for your interest. How can we assist with your property needs? 🏠")
    send_whatsapp_message(phone_number,
                          response_text + f"\n\nIf you want to talk to a sales agent contact: {HUMAN_CONTACT}")


def get_conversation_history(phone):
    """Retrieve conversation history from SQLite"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT role, content FROM conversations WHERE phone = ? ORDER BY timestamp DESC LIMIT 10", (phone,))
    history = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return history


def save_conversation(phone, role, content):
    """Save conversation to SQLite"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (phone, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (phone, role, content, datetime.now().isoformat()))
    c.execute(
        "DELETE FROM conversations WHERE phone = ? AND rowid NOT IN (SELECT rowid FROM conversations WHERE phone = ? ORDER BY timestamp DESC LIMIT 10)",
        (phone, phone))
    conn.commit()
    conn.close()


@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Webhook verification endpoint"""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    logger.info(f"🔍 Verification attempt - Mode: {mode}, Token: {token}")
    if mode == 'subscribe' and token == WEBHOOK_VERIFY_TOKEN:
        logger.info('✅ Webhook verified successfully!')
        return challenge, 200
    logger.error('❌ Webhook verification failed')
    return 'Verification failed', 403


@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle incoming WhatsApp messages - FIXED to prevent duplicate processing and unsolicited messages"""
    try:
        body = request.get_json()
        logger.info(f"📨 Received payload: {json.dumps(body, indent=2)}")

        if body.get('object') != 'whatsapp_business_account':
            logger.info("ℹ️ Non-WhatsApp payload received, ignoring")
            return 'EVENT_RECEIVED', 200

        for entry in body.get('entry', []):
            for change in entry.get('changes', []):
                if change.get('field') != 'messages':
                    continue

                value = change.get('value', {})
                messages = value.get('messages', [])

                # CRITICAL: Process each message only once
                for message in messages:
                    phone_number = message.get('from')
                    message_id = message.get('id')  # WhatsApp message ID
                    message_timestamp = message.get('timestamp')
                    message_type = message.get('type')

                    # FIXED: Create unique message identifier for deduplication
                    unique_msg_id = f"{phone_number}_{message_id}_{message_timestamp}"

                    # CRITICAL: Skip if message already processed
                    if is_message_processed(unique_msg_id):
                        logger.info(f"⏭️ Message {unique_msg_id} already processed, skipping")
                        continue

                    # Mark message as processed immediately to prevent duplicates
                    mark_message_processed(unique_msg_id, phone_number)

                    logger.info(f"🔄 Processing message {unique_msg_id} from {phone_number}")

                    # Only send welcome message for first-time users with text/interactive messages
                    conversation_history = get_conversation_history(phone_number)
                    should_send_welcome = (not conversation_history and
                                           message_type in ['text', 'interactive'] and
                                           phone_number)  # Ensure phone number exists

                    if should_send_welcome:
                        logger.info(f"👋 Sending welcome message to new user {phone_number}")
                        result = send_welcome_message_with_buttons(phone_number)
                        if result is None:
                            logger.warning(f"⚠️ Failed to send welcome message to {phone_number}")

                    # Handle different message types
                    if message_type == 'text':
                        message_body = message.get('text', {}).get('body', '').strip()
                        if not message_body:  # Skip empty messages
                            logger.info(f"⏭️ Empty text message from {phone_number}, skipping")
                            continue

                        logger.info(f"📥 Text message from {phone_number}: {message_body}")

                        # Save user message and get AI response
                        save_conversation(phone_number, "user", message_body)
                        conversation_history = get_conversation_history(phone_number)
                        ai_response = get_ai_response(message_body, conversation_history)

                        if ai_response:
                            # Filter properties to send images
                            relevant_properties, _ = filter_properties_by_query(message_body, PROPERTIES_DATABASE)
                            image_urls_to_send = []
                            for prop in relevant_properties[:3]:  # Limit to top 3 properties
                                if 'image_urls' in prop and prop['image_urls']:
                                    image_urls_to_send.extend(prop['image_urls'])

                            save_conversation(phone_number, "assistant", ai_response)
                            result = send_whatsapp_message(phone_number, ai_response, image_urls_to_send)

                            if result is None:
                                logger.warning(f"⚠️ Failed to send AI response to {phone_number}")
                        else:
                            logger.warning(f"⚠️ No AI response generated for {phone_number}")

                    elif message_type == 'voice':
                        logger.info(f"🎤 Voice message received from {phone_number} - HUMAN TAKEOVER TRIGGERED")
                        # No automated response for voice messages - human takeover

                    elif message_type == 'interactive':
                        interactive = message.get('interactive', {})
                        if interactive.get('type') == 'button_reply':
                            button_id = interactive.get('button_reply', {}).get('id')
                            if button_id:  # Ensure button_id exists
                                logger.info(f"🔘 Button clicked: {button_id} by {phone_number}")
                                handle_button_response(button_id, phone_number)
                            else:
                                logger.warning(f"⚠️ Button interaction without button_id from {phone_number}")

                    elif message_type in ['image', 'document', 'audio', 'video']:
                        logger.info(f"📎 Media message ({message_type}) from {phone_number} - HUMAN TAKEOVER TRIGGERED")
                        # No automated response for media messages - human takeover

                    else:
                        logger.info(f"❓ Unknown message type '{message_type}' from {phone_number}")

        return 'EVENT_RECEIVED', 200

    except Exception as e:
        logger.error(f"❌ Webhook processing error: {str(e)}", exc_info=True)
        return 'EVENT_RECEIVED', 200  # Always return 200 to prevent webhook retries


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT phone) FROM conversations")
    active_conversations = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM processed_messages")
    processed_count = c.fetchone()[0]
    conn.close()

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_conversations': active_conversations,
        'processed_messages': processed_count,
        'groq_api': bool(GROQ_API_KEY),
        'gemini_api': bool(GEMINI_API_KEY),
        'properties_for_sale': len(PROPERTIES_FOR_SALE),
        'properties_for_rent': len(PROPERTIES_FOR_RENT),
        'total_properties': len(PROPERTIES_DATABASE)
    }), 200


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get conversation statistics"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT phone, COUNT(*), MAX(timestamp) FROM conversations GROUP BY phone")
    conversations = [
        {'phone': row[0], 'message_count': row[1], 'last_message': row[2]}
        for row in c.fetchall()
    ]
    conn.close()
    return jsonify({
        'total_conversations': len(conversations),
        'conversations': conversations
    }), 200


@app.route('/properties', methods=['GET'])
def get_properties():
    """Get all properties in the database"""
    return jsonify({
        'properties_for_sale': len(PROPERTIES_FOR_SALE),
        'properties_for_rent': len(PROPERTIES_FOR_RENT),
        'total_properties': len(PROPERTIES_DATABASE),
        'sale_properties': PROPERTIES_FOR_SALE,
        'rent_properties': PROPERTIES_FOR_RENT
    }), 200


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Dashboard to display logs in an HTML table"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT phone, role, content, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 50")
    logs = [{'phone': row[0], 'role': row[1], 'content': row[2], 'timestamp': row[3]} for row in c.fetchall()]

    c.execute("SELECT COUNT(*) FROM processed_messages")
    processed_count = c.fetchone()[0]
    conn.close()

    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>WhatsApp Bot Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .stats { background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; font-weight: bold; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f5f5f5; }
            .user { background-color: #e8f5e8; }
            .assistant { background-color: #e8e8f5; }
        </style>
    </head>
    <body>
        <h1>WhatsApp Bot Dashboard</h1>
        <div class="stats">
            <h3>📊 Statistics</h3>
            <p><strong>Properties for Sale:</strong> ''' + str(len(PROPERTIES_FOR_SALE)) + '''</p>
            <p><strong>Properties for Rent:</strong> ''' + str(len(PROPERTIES_FOR_RENT)) + '''</p>
            <p><strong>Total Properties:</strong> ''' + str(len(PROPERTIES_DATABASE)) + '''</p>
            <p><strong>Processed Messages:</strong> ''' + str(processed_count) + '''</p>
            <p><strong>Active Conversations:</strong> ''' + str(len(set(log['phone'] for log in logs))) + '''</p>
        </div>
        <h2>📱 Recent Conversations</h2>
        <table>
            <tr>
                <th>Phone</th>
                <th>Role</th>
                <th>Content</th>
                <th>Timestamp</th>
            </tr>
            ''' + ''.join(f'''
            <tr class="{log['role']}">
                <td>{log['phone']}</td>
                <td>{log['role']}</td>
                <td>{log['content'][:100]}{'...' if len(log['content']) > 100 else ''}</td>
                <td>{log['timestamp']}</td>
            </tr>
            ''' for log in logs) + '''
        </table>
    </body>
    </html>
    '''


@app.route('/test-ai', methods=['GET'])
def test_ai():
    """Test AI response endpoint"""
    test_message = request.args.get('message', 'Show me houses for rent in DHA')
    response = get_ai_response(test_message)
    return jsonify({
        'test_message': test_message,
        'ai_response': response,
        'properties_for_sale': len(PROPERTIES_FOR_SALE),
        'properties_for_rent': len(PROPERTIES_FOR_RENT),
        'total_properties': len(PROPERTIES_DATABASE),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/clear-processed', methods=['POST'])
def clear_processed_messages():
    """Clear processed messages cache - useful for testing"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("DELETE FROM processed_messages")
    conn.commit()
    conn.close()
    return jsonify({'message': 'Processed messages cache cleared'}), 200


if __name__ == '__main__':
    logger.info("🚀 Starting WhatsApp AI Chatbot...")
    logger.info("✅ All required environment variables are set")
    logger.info(f"🏠 Loaded {len(PROPERTIES_FOR_SALE)} properties for sale")
    logger.info(f"🏡 Loaded {len(PROPERTIES_FOR_RENT)} properties for rent")
    logger.info(f"📊 Total {len(PROPERTIES_DATABASE)} properties in database")

    # Validate token at startup
    validate_whatsapp_token()

    logger.info("🎯 Key fixes applied:")
    logger.info("   ✅ Message deduplication system")
    logger.info("   ✅ Safe retry configuration (GET only)")
    logger.info("   ✅ Enhanced error handling")
    logger.info("   ✅ Prevented unsolicited message sending")
    logger.info("   ✅ Added processed message tracking")

    logger.info("📡 Available endpoints:")
    logger.info("   - /webhook (GET/POST)")
    logger.info("   - /health (GET)")
    logger.info("   - /stats (GET)")
    logger.info("   - /dashboard (GET)")
    logger.info("   - /test-ai (GET)")
    logger.info("   - /properties (GET)")
    logger.info("   - /clear-processed (POST)")

   port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    if os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('RENDER') or os.getenv('HEROKU'):
        logger.info(f"🌐 Starting in production mode on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.info(f"🔧 Starting in development mode on port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)




