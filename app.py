from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
import os
import uuid
import json
import re
import logging
import html
from datetime import datetime, timedelta
import sqlite3
from contextlib import contextmanager
from dotenv import load_dotenv
import requests as http_requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Production-ready CORS - restrict in production
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    if os.getenv('FLASK_ENV') == 'production':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# JWT Configuration - MUST be set in production
JWT_SECRET = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET:
    if os.getenv('FLASK_ENV') == 'production':
        raise ValueError("JWT_SECRET_KEY must be set in production!")
    logger.warning("⚠️  Using default JWT secret - NOT SAFE FOR PRODUCTION!")
    JWT_SECRET = 'dev-secret-change-in-production'

app.config['JWT_SECRET_KEY'] = JWT_SECRET
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

# API Base URL for production (used in PDF download URLs)
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')

# Lenco Payment Gateway Configuration
LENCO_SECRET_KEY = os.getenv('LENCO_SECRET_KEY', '')
LENCO_API_URL = 'https://api.lenco.co/access/v2'

# Legacy Payment Gateway Configuration (Tumeny - deprecated)
PAYMENT_GATEWAY_URL = os.getenv('PAYMENT_GATEWAY_URL', '')
PAYMENT_GATEWAY_ENABLED = os.getenv('PAYMENT_GATEWAY_ENABLED', 'false').lower() == 'true'

if LENCO_SECRET_KEY:
    logger.info("💳 Lenco payment gateway configured (instant mobile money)")
elif PAYMENT_GATEWAY_ENABLED:
    logger.info(f"💳 Legacy payment gateway enabled: {PAYMENT_GATEWAY_URL}")
else:
    logger.info("💰 Running in payment simulation mode")

# Configure Gemini AI
# Get free API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    logger.info("✅ Gemini AI key configured (using REST API)")
else:
    logger.warning("⚠️  GEMINI_API_KEY not set. AI features will not work.")

def call_gemini_api(prompt):
    """Call Gemini API using REST instead of deprecated SDK"""
    if not GEMINI_API_KEY:
        raise Exception("AI service not configured")
    
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 40,
            "maxOutputTokens": 2048
        }
    }
    
    response = http_requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result['candidates'][0]['content']['parts'][0]['text']

# Directory to store generated PDFs
PDF_DIR = 'generated_cvs'
os.makedirs(PDF_DIR, exist_ok=True)

# Database setup
DB_PATH = 'cv_generator.db'
DB_CONNECTION_POOL = []
MAX_POOL_SIZE = 10

@contextmanager
def get_db():
    """Context manager for database connections with automatic cleanup"""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        conn.close()

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Admin users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Customers/Transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE NOT NULL,
            phone_number TEXT,
            payment_method TEXT,
            amount REAL,
            status TEXT,
            cv_data TEXT,
            pdf_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_id ON transactions(transaction_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON transactions(phone_number)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON transactions(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON transactions(created_at)')
    
    # Check if admin exists, if not create default admin
    cursor.execute('SELECT COUNT(*) FROM admins WHERE username = ?', ('admin',))
    if cursor.fetchone()[0] == 0:
        # Default password: admin123 (CHANGE THIS IN PRODUCTION!)
        default_password = generate_password_hash('admin123')
        cursor.execute('INSERT INTO admins (username, password_hash) VALUES (?, ?)', 
                      ('admin', default_password))
        logger.warning("⚠️  Default admin created - username: admin, password: admin123")
        logger.warning("⚠️  CHANGE THIS PASSWORD IMMEDIATELY!")
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# Memory management - limit cache sizes with LRU
MAX_CACHE_SIZE = 200  # Increased for better hit rate
ai_cache = {}
cache_access_count = {}  # Track access for LRU

def add_to_cache(cache_dict, key, value):
    """Add to cache with LRU eviction to prevent memory leaks"""
    if len(cache_dict) >= MAX_CACHE_SIZE:
        # Remove least recently used entry (LRU)
        if cache_access_count:
            lru_key = min(cache_access_count, key=cache_access_count.get)
            del cache_dict[lru_key]
            del cache_access_count[lru_key]
        else:
            # Fallback to FIFO if access count empty
            oldest_key = next(iter(cache_dict))
            del cache_dict[oldest_key]
    
    cache_dict[key] = value
    cache_access_count[key] = 1  # Initialize access count

def get_from_cache(cache_dict, key):
    """Get from cache and update access count for LRU"""
    if key in cache_dict:
        cache_access_count[key] = cache_access_count.get(key, 0) + 1
        return cache_dict[key]
    return None

def validate_phone_number(phone):
    """Validate Zambian phone number format"""
    # Accept formats: +260XXXXXXXXX, 260XXXXXXXXX, 0XXXXXXXXX
    pattern = r'^(\+?260|0)?[0-9]{9,10}$'
    return re.match(pattern, phone.replace(' ', '')) is not None

def validate_transaction_id(tid):
    """Validate transaction ID is a valid UUID"""
    try:
        uuid.UUID(tid)
        return True
    except (ValueError, AttributeError):
        return False

def sanitize_string(text):
    """Sanitize user input to prevent XSS"""
    if not text or not isinstance(text, str):
        return text
    # Escape HTML special characters
    return html.escape(text.strip())

def sanitize_cv_data(cv_data):
    """Recursively sanitize CV data to prevent XSS attacks"""
    if isinstance(cv_data, dict):
        return {key: sanitize_cv_data(value) for key, value in cv_data.items()}
    elif isinstance(cv_data, list):
        return [sanitize_cv_data(item) for item in cv_data]
    elif isinstance(cv_data, str):
        return sanitize_string(cv_data)
    return cv_data

def format_date(date_str):
    """Format date from YYYY-MM to Month Year"""
    if not date_str:
        return ''
    try:
        year, month = date_str.split('-')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return f"{months[int(month)-1]} {year}"
    except:
        return date_str

def generate_pdf(cv_data, filename):
    """Generate PDF from CV data using ReportLab"""
    try:
        pdf_path = os.path.join(PDF_DIR, filename)
        doc = SimpleDocTemplate(
            pdf_path, 
            pagesize=A4, 
            topMargin=0.5*inch, 
            bottomMargin=0.5*inch
        )
    except Exception as e:
        logger.error(f"Failed to create PDF document: {str(e)}")
        raise
    
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles for ATS optimization
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.black,
        spaceAfter=12,
        spaceBefore=0,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        textTransform='uppercase'
    )
    
    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        leading=16,
        alignment=TA_LEFT
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        leading=14,
        leftIndent=12,
        bulletIndent=0
    )
    
    # Personal Info Header - ATS Optimized
    personal_info = cv_data.get('personalInfo', {})
    if personal_info.get('fullName'):
        story.append(Paragraph(f"<b>{personal_info['fullName']}</b>", title_style))
        
        contact_parts = []
        if personal_info.get('email'):
            contact_parts.append(personal_info['email'])
        if personal_info.get('phone'):
            contact_parts.append(personal_info['phone'])
        if personal_info.get('city') and personal_info.get('country'):
            contact_parts.append(f"{personal_info['city']}, {personal_info['country']}")
        
        if contact_parts:
            contact_text = ' | '.join(contact_parts)
            story.append(Paragraph(contact_text, normal_style))
        
        story.append(Spacer(1, 0.15*inch))
    
    # Professional Summary
    if personal_info.get('summary'):
        story.append(Paragraph('<b>PROFESSIONAL SUMMARY</b>', heading_style))
        summary_style = ParagraphStyle(
            'Summary',
            parent=normal_style,
            alignment=TA_JUSTIFY
        )
        story.append(Paragraph(personal_info['summary'], summary_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Skills - ATS Optimized (Single column bullets)
    skills = cv_data.get('skills', [])
    if skills:
        story.append(Paragraph('<b>SKILLS</b>', heading_style))
        for skill in skills:
            skill_text = f"• {skill['name']}"
            story.append(Paragraph(skill_text, bullet_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Education - ATS Optimized
    education = cv_data.get('education', [])
    if education:
        story.append(Paragraph('<b>EDUCATION</b>', heading_style))
        for edu in education:
            # Degree and field
            edu_title = f"<b>{edu.get('degree', '')}</b>"
            if edu.get('field'):
                edu_title += f" in {edu.get('field')}"
            story.append(Paragraph(edu_title, normal_style))
            
            # Institution
            institution = edu.get('institution', '')
            story.append(Paragraph(institution, normal_style))
            
            # Date and location on one line
            date_info = []
            if edu.get('startDate') or edu.get('endDate'):
                date_str = f"{format_date(edu.get('startDate', ''))} - {format_date(edu.get('endDate', '')) or 'Present'}"
                date_info.append(date_str)
            if edu.get('location'):
                date_info.append(edu['location'])
            
            if date_info:
                story.append(Paragraph(' | '.join(date_info), normal_style))
            
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.05*inch))
    
    # Work Experience - ATS Optimized
    experience = cv_data.get('experience', [])
    if experience:
        story.append(Paragraph('<b>WORK EXPERIENCE</b>', heading_style))
        for exp in experience:
            # Position (bold)
            exp_title = f"<b>{exp.get('position', '')}</b>"
            story.append(Paragraph(exp_title, normal_style))
            
            # Company
            company = exp.get('company', '')
            story.append(Paragraph(company, normal_style))
            
            # Date and location
            date_info = []
            if exp.get('startDate') or exp.get('endDate'):
                date_str = f"{format_date(exp.get('startDate', ''))} - {format_date(exp.get('endDate', '')) or 'Present'}"
                date_info.append(date_str)
            if exp.get('location'):
                date_info.append(exp['location'])
            
            if date_info:
                story.append(Paragraph(' | '.join(date_info), normal_style))
            
            # Description (if any)
            if exp.get('description'):
                story.append(Paragraph(exp['description'], normal_style))
            
            # Responsibilities as bullet points
            if exp.get('responsibilities'):
                for resp in exp['responsibilities']:
                    bullet_text = f"• {resp}"
                    story.append(Paragraph(bullet_text, bullet_style))
            
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.05*inch))
    
    # Licenses & Certifications - ATS Optimized
    licensing = cv_data.get('licensing', [])
    if licensing:
        story.append(Paragraph('<b>LICENSES & CERTIFICATIONS</b>', heading_style))
        for license_item in licensing:
            license_title = f"<b>{license_item.get('name', '')}</b>"
            story.append(Paragraph(license_title, normal_style))
            
            org = license_item.get('issuingOrganization', '')
            story.append(Paragraph(org, normal_style))
            
            date_info = []
            if license_item.get('issueDate'):
                date_str = f"Issued: {format_date(license_item.get('issueDate', ''))}"
                if license_item.get('expiryDate'):
                    date_str += f" | Expires: {format_date(license_item.get('expiryDate', ''))}"
                date_info.append(date_str)
            if license_item.get('credentialId'):
                date_info.append(f"ID: {license_item['credentialId']}")
            
            if date_info:
                story.append(Paragraph(' | '.join(date_info), normal_style))
            
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.05*inch))
    
    # Languages - ATS Optimized (Single column)
    languages = cv_data.get('languages', [])
    if languages:
        story.append(Paragraph('<b>LANGUAGES</b>', heading_style))
        for lang in languages:
            lang_text = f"• {lang['name']} - {lang.get('proficiency', 'Intermediate')}"
            story.append(Paragraph(lang_text, bullet_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Hobbies & Interests - ATS Optimized
    hobbies = cv_data.get('hobbies', '')
    if hobbies:
        story.append(Paragraph('<b>HOBBIES & INTERESTS</b>', heading_style))
        story.append(Paragraph(hobbies, normal_style))
        story.append(Spacer(1, 0.15*inch))
    
    # References - ATS Optimized
    references = cv_data.get('references', [])
    if references:
        story.append(Paragraph('<b>REFERENCES</b>', heading_style))
        for ref in references:
            ref_title = f"<b>{ref.get('name', '')}</b>"
            story.append(Paragraph(ref_title, normal_style))
            
            position_company = []
            if ref.get('position'):
                position_company.append(ref['position'])
            if ref.get('company'):
                position_company.append(f"at {ref['company']}")
            
            if position_company:
                story.append(Paragraph(' '.join(position_company), normal_style))
            
            contact_info = []
            if ref.get('phone'):
                contact_info.append(ref['phone'])
            if ref.get('email'):
                contact_info.append(ref['email'])
            
            if contact_info:
                story.append(Paragraph(' | '.join(contact_info), normal_style))
            
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.05*inch))
    
    doc.build(story)
    return pdf_path

@app.route('/api/initiate-payment', methods=['POST'])
def initiate_payment():
    """Initiate payment for CV download"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        phone_number = data.get('phoneNumber', '').strip()
        payment_method = data.get('paymentMethod', '').strip()
        amount = data.get('amount', 50)
        cv_data = data.get('cvData')
        
        # Input validation
        if not phone_number or not cv_data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        if not validate_phone_number(phone_number):
            return jsonify({'error': 'Invalid phone number format'}), 400
            
        if payment_method not in ['mtn', 'airtel', 'zamtel']:
            return jsonify({'error': 'Invalid payment method'}), 400
        
        # STRICT: CV generation costs exactly K50 - no discounts allowed
        if not isinstance(amount, (int, float)) or amount < 50:
            return jsonify({'error': 'Payment amount must be exactly K50'}), 400
        
        # Lock amount to K50 regardless of what was sent
        amount = 50
        
        # Sanitize CV data to prevent XSS
        cv_data = sanitize_cv_data(cv_data)
        
        # Generate transaction ID
        transaction_id = str(uuid.uuid4())
        
        # In production, integrate with real payment gateway (MTN, Airtel, etc.)
        # For demo, we'll simulate the payment process
        
        # Store transaction in database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transactions (transaction_id, phone_number, payment_method, amount, status, cv_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (transaction_id, phone_number, payment_method, amount, 'pending', json.dumps(cv_data)))
        
        logger.info(f"Payment initiated: {transaction_id} - {payment_method} - {phone_number}")
        
        # Real payment gateway or simulation mode
        if PAYMENT_GATEWAY_ENABLED:
            # Return payment URL for user to complete payment on Tumeny
            # Extract customer information from CV data
            customer_name = cv_data.get('personalInfo', {}).get('fullName', 'Customer')
            customer_email = cv_data.get('personalInfo', {}).get('email', '')
            description = f"CV Generation - {customer_name}"
            
            # Build payment URL with all parameters for auto-fill
            # d = description (locked), amount = payment amount (locked), ref = transaction reference
            # phone = customer phone, email = customer email, name = customer name
            payment_url = f"{PAYMENT_GATEWAY_URL}?d={description}&amount={amount}&ref={transaction_id}&phone={phone_number}&email={customer_email}&name={customer_name}"
            logger.info(f"Real payment URL generated: {payment_url}")
            
            return jsonify({
                'success': True,
                'transactionId': transaction_id,
                'paymentUrl': payment_url,
                'message': 'Redirecting to payment gateway...',
                'useGateway': True
            })
        else:
            # Simulation mode - auto-complete for testing
            import threading
            def complete_payment():
                import time
                try:
                    time.sleep(2)  # Simulate processing time
                    
                    # Generate PDF
                    full_name = cv_data.get('personalInfo', {}).get('fullName', 'CV')
                    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', full_name)
                    filename = f"CV_{safe_name}_{transaction_id[:8]}.pdf"
                    pdf_path = generate_pdf(cv_data, filename)
                    
                    # Update database with completed status
                    with get_db() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE transactions 
                            SET status = ?, pdf_filename = ?, completed_at = CURRENT_TIMESTAMP
                            WHERE transaction_id = ?
                        ''', ('completed', filename, transaction_id))
                    
                    logger.info(f"Payment completed: {transaction_id}")
                except Exception as e:
                    logger.error(f"Error completing payment {transaction_id}: {str(e)}")
                    # Mark as failed
                    try:
                        with get_db() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                'UPDATE transactions SET status = ? WHERE transaction_id = ?',
                                ('failed', transaction_id)
                            )
                    except Exception as db_error:
                        logger.error(f"Failed to update transaction status: {str(db_error)}")
            
            threading.Thread(target=complete_payment, daemon=True).start()
            
            return jsonify({
                'success': True,
                'transactionId': transaction_id,
                'message': 'Payment initiated successfully'
            })
    
    except Exception as e:
        print(f"Error initiating payment: {str(e)}")
        return jsonify({'error': 'Failed to initiate payment'}), 500

@app.route('/api/payment-callback', methods=['POST', 'GET'])
def payment_callback():
    """Webhook endpoint for Tumeny payment confirmation"""
    try:
        # Tumeny sends payment confirmation here
        data = request.json if request.method == 'POST' else request.args.to_dict()
        logger.info(f"Payment callback received: {data}")
        
        transaction_id = data.get('ref') or data.get('transaction_id')
        status = data.get('status', '').lower()
        
        if not transaction_id:
            return jsonify({'error': 'No transaction ID provided'}), 400
        
        if not validate_transaction_id(transaction_id):
            return jsonify({'error': 'Invalid transaction ID'}), 400
        
        # Update transaction status based on payment gateway response
        if status in ['success', 'completed', 'paid']:
            # Payment successful - generate PDF
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT cv_data FROM transactions WHERE transaction_id = ?',
                    (transaction_id,)
                )
                result = cursor.fetchone()
            
            if result and result['cv_data']:
                cv_data = json.loads(result['cv_data'])
                full_name = cv_data.get('personalInfo', {}).get('fullName', 'CV')
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', full_name)
                filename = f"CV_{safe_name}_{transaction_id[:8]}.pdf"
                
                try:
                    pdf_path = generate_pdf(cv_data, filename)
                    
                    with get_db() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE transactions 
                            SET status = ?, pdf_filename = ?, completed_at = CURRENT_TIMESTAMP
                            WHERE transaction_id = ?
                        ''', ('completed', filename, transaction_id))
                    
                    logger.info(f"Payment confirmed and PDF generated: {transaction_id}")
                    return jsonify({'success': True, 'message': 'Payment confirmed'})
                except Exception as pdf_error:
                    logger.error(f"PDF generation failed: {str(pdf_error)}")
                    return jsonify({'error': 'Payment received but PDF failed'}), 500
        elif status in ['failed', 'cancelled', 'declined']:
            # Payment failed
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE transactions SET status = ? WHERE transaction_id = ?',
                    ('failed', transaction_id)
                )
            logger.warning(f"Payment failed: {transaction_id}")
            return jsonify({'success': False, 'message': 'Payment failed'})
        else:
            logger.info(f"Payment status unknown: {status} for {transaction_id}")
            return jsonify({'success': False, 'message': 'Unknown status'})
    
    except Exception as e:
        logger.error(f"Payment callback error: {str(e)}")
        return jsonify({'error': 'Callback processing failed'}), 500

@app.route('/api/payment-status/<transaction_id>', methods=['GET'])
def check_payment_status(transaction_id):
    """Check payment status"""
    try:
        # Validate transaction ID format
        if not validate_transaction_id(transaction_id):
            return jsonify({'error': 'Invalid transaction ID'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT status, pdf_filename FROM transactions WHERE transaction_id = ?',
                (transaction_id,)
            )
            result = cursor.fetchone()
        
        if not result:
            return jsonify({'error': 'Transaction not found'}), 404
        
        pdf_url = None
        if result['status'] == 'completed' and result['pdf_filename']:
            pdf_url = f"{API_BASE_URL}/api/download-cv/{transaction_id}"
        
        return jsonify({
            'status': result['status'],
            'pdfUrl': pdf_url
        })
    except Exception as e:
        logger.error(f"Error checking payment status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/payment/verify/<reference>', methods=['GET'])
def verify_lenco_payment(reference):
    """Verify Lenco payment and generate CV"""
    try:
        if not LENCO_SECRET_KEY:
            return jsonify({'error': 'Payment system not configured'}), 503
        
        # Validate reference format
        if not reference or not reference.startswith('CV-'):
            return jsonify({'error': 'Invalid payment reference'}), 400
        
        # Call Lenco API to verify payment
        lenco_url = f"{LENCO_API_URL}/collections/status/{reference}"
        headers = {
            'Authorization': f'Bearer {LENCO_SECRET_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = http_requests.get(lenco_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Lenco API error: {response.status_code} - {response.text}")
            return jsonify({'error': 'Payment verification failed'}), 500
        
        payment_data = response.json()
        
        if not payment_data.get('status'):
            return jsonify({'error': 'Invalid response from payment gateway'}), 500
        
        collection = payment_data.get('data', {})
        payment_status = collection.get('status', '').lower()
        amount = float(collection.get('amount', 0))
        
        # Check if payment was successful
        if payment_status != 'successful':
            return jsonify({
                'success': False,
                'status': payment_status,
                'message': 'Payment not completed'
            })
        
        # Verify amount is at least K50
        if amount < 50:
            logger.warning(f"Insufficient payment: {reference} - Amount: {amount}")
            return jsonify({'error': 'Insufficient payment amount'}), 400
        
        # Get CV data from session storage (frontend sends it)
        # In this flow, we need to store CV data when payment is initiated
        # For now, check database or use callback
        
        # Check if already processed
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT status, pdf_filename, cv_data FROM transactions WHERE transaction_id = ?',
                (reference,)
            )
            existing = cursor.fetchone()
        
        if existing:
            if existing['status'] == 'completed' and existing['pdf_filename']:
                # Already processed
                pdf_url = f"{API_BASE_URL}/api/download-cv/{reference}"
                return jsonify({
                    'success': True,
                    'status': 'successful',
                    'pdfUrl': pdf_url,
                    'message': 'CV ready for download'
                })
        else:
            # New payment - need CV data from frontend callback
            # Store payment record for webhook processing
            phone = collection.get('mobileMoneyDetails', {}).get('phone', '')
            
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO transactions 
                    (transaction_id, phone_number, payment_method, amount, status, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (reference, phone, 'mobile-money', amount, 'pending_cv_data'))
            
            return jsonify({
                'success': True,
                'status': 'pending_cv_data',
                'message': 'Payment verified, waiting for CV data'
            })
        
    except http_requests.exceptions.RequestException as e:
        logger.error(f"Lenco API request failed: {str(e)}")
        return jsonify({'error': 'Payment gateway unreachable'}), 503
    except Exception as e:
        logger.error(f"Payment verification error: {str(e)}")
        return jsonify({'error': 'Verification failed'}), 500

@app.route('/api/payment/generate-cv', methods=['POST'])
def generate_cv_after_payment():
    """Generate CV after payment verification"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        reference = data.get('reference')
        cv_data = data.get('cvData')
        
        if not reference or not cv_data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Verify payment exists and is paid
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT status, amount FROM transactions WHERE transaction_id = ?',
                (reference,)
            )
            payment = cursor.fetchone()
        
        if not payment:
            return jsonify({'error': 'Payment not found'}), 404
        
        if payment['status'] not in ['pending_cv_data', 'completed']:
            return jsonify({'error': 'Payment not verified'}), 403
        
        if payment['amount'] < 50:
            return jsonify({'error': 'Insufficient payment'}), 403
        
        # Generate PDF
        full_name = cv_data.get('personalInfo', {}).get('fullName', 'CV')
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', full_name)
        filename = f"CV_{safe_name}_{reference[:15]}.pdf"
        
        pdf_path = generate_pdf(cv_data, filename)
        
        # Update database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE transactions 
                SET status = ?, pdf_filename = ?, cv_data = ?, completed_at = CURRENT_TIMESTAMP
                WHERE transaction_id = ?
            ''', ('completed', filename, json.dumps(cv_data), reference))
        
        pdf_url = f"{API_BASE_URL}/api/download-cv/{reference}"
        logger.info(f"CV generated for payment: {reference}")
        
        return jsonify({
            'success': True,
            'pdfUrl': pdf_url,
            'message': 'CV generated successfully'
        })
    except Exception as e:
        logger.error(f"CV generation error: {str(e)}")
        return jsonify({'error': 'CV generation failed'}), 500

@app.route('/api/download-cv/<transaction_id>', methods=['GET'])
def download_cv(transaction_id):
    """Download generated CV PDF"""
    try:
        # Validate transaction ID format
        if not validate_transaction_id(transaction_id):
            return jsonify({'error': 'Invalid transaction ID'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT status, pdf_filename, amount FROM transactions WHERE transaction_id = ?',
                (transaction_id,)
            )
            result = cursor.fetchone()
        
        if not result:
            return jsonify({'error': 'Transaction not found'}), 404
        
        if result['status'] != 'completed':
            return jsonify({'error': 'Payment not completed'}), 403
        
        # VERIFY FULL PAYMENT: Ensure K50 was paid
        if result['amount'] and result['amount'] < 50:
            logger.warning(f"Incomplete payment detected: {transaction_id} - Amount: {result['amount']}")
            return jsonify({'error': 'Incomplete payment. Full K50 required for CV download.'}), 403
        
        pdf_filename = result['pdf_filename']
        if not pdf_filename:
            return jsonify({'error': 'PDF not generated'}), 404
            
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return jsonify({'error': 'PDF file not found'}), 404
        
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
    except Exception as e:
        logger.error(f"Error downloading CV: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'CV Generator API is running'})

@app.route('/api/ai/suggest-summary', methods=['POST'])
def suggest_summary():
    """Generate professional summary using AI"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({'error': 'AI service not configured'}), 503
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        profession = data.get('profession', '').strip()
        years_experience = data.get('yearsExperience', 0)
        specialization = data.get('specialization', '').strip()
        
        if not profession:
            return jsonify({'error': 'Profession is required'}), 400
        
        # Validate years_experience is a number
        try:
            years_experience = int(years_experience)
        except (ValueError, TypeError):
            years_experience = 0
        
        # Create cache key
        cache_key = f"summary_{profession}_{years_experience}_{specialization}".lower()
        
        # Check cache first with LRU tracking
        cached_summary = get_from_cache(ai_cache, cache_key)
        if cached_summary:
            logger.info(f"Cache hit for summary: {profession}")
            return jsonify({'summary': cached_summary})
        
        # Generate prompt
        prompt = f"""Generate a professional CV summary for a {profession} with {years_experience} years of experience in Zambia.
        
{f"Specialization: {specialization}" if specialization else ""}

Requirements:
- 3-4 sentences maximum
- Professional and confident tone
- Highlight key strengths relevant to Zambian job market
- Include career goals
- Focus on value proposition to employers
- Use Zambian English conventions

Return ONLY the summary text, no additional formatting or labels."""

        # Generate response using REST API
        summary = call_gemini_api(prompt).strip()
        
        # Cache the response with size limit
        add_to_cache(ai_cache, cache_key, summary)
        logger.info(f"Generated summary for: {profession}")
        
        return jsonify({'summary': summary})
    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return jsonify({'error': 'Failed to generate summary', 'message': str(e)}), 500

@app.route('/api/ai/suggest-skills', methods=['POST'])
def suggest_skills():
    """Generate skills suggestions using AI"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({'error': 'AI service not configured'}), 503
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        profession = data.get('profession', '').strip()
        specialization = data.get('specialization', '').strip()
        
        if not profession:
            return jsonify({'error': 'Profession is required'}), 400
        
        # Create cache key
        cache_key = f"skills_{profession}_{specialization}".lower()
        
        # Check cache first with LRU tracking
        cached_skills = get_from_cache(ai_cache, cache_key)
        if cached_skills:
            logger.info(f"Cache hit for skills: {profession}")
            return jsonify({'skills': cached_skills})
        
        # Generate prompt
        prompt = f"""List 12-15 most important skills for a {profession} in Zambia.
        
{f"Focus on: {specialization}" if specialization else ""}

Requirements:
- Include both technical/hard skills and soft skills
- Relevant to Zambian job market
- Mix of industry-standard and local requirements
- Skills should be specific and measurable
- Return as JSON array with objects containing "name" field only

Example format:
[
  {{"name": "Skill 1"}},
  {{"name": "Skill 2"}},
  ...
]

Return ONLY the JSON array, no markdown formatting or additional text."""

        # Generate response using REST API
        skills_text = call_gemini_api(prompt).strip()
        
        # Clean up markdown if present
        if skills_text.startswith('```json'):
            skills_text = skills_text.replace('```json', '').replace('```', '').strip()
        elif skills_text.startswith('```'):
            skills_text = skills_text.replace('```', '').strip()
        
        # Parse JSON
        skills = json.loads(skills_text)
        
        # Cache the response with size limit
        add_to_cache(ai_cache, cache_key, skills)
        logger.info(f"Generated skills for: {profession}")
        
        return jsonify({'skills': skills})
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}, Response: {skills_text}")
        return jsonify({'error': 'Failed to parse AI response'}), 500
    except Exception as e:
        logger.error(f"Error generating skills: {str(e)}")
        return jsonify({'error': 'Failed to generate skills', 'message': str(e)}), 500

@app.route('/api/ai/enhance-description', methods=['POST'])
def enhance_description():
    """Enhance job experience description using AI"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({'error': 'AI service not configured'}), 503
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        position = data.get('position', '').strip()
        company = data.get('company', '').strip()
        basic_description = data.get('description', '').strip()
        
        if not position:
            return jsonify({'error': 'Position is required'}), 400
        
        # Generate prompt
        prompt = f"""Enhance this job experience description for a CV:

Position: {position}
Company: {company}
Current description: {basic_description if basic_description else "Not provided"}

Requirements:
- Write 3-5 bullet points describing key responsibilities and achievements
- Use action verbs (Led, Managed, Developed, Implemented, etc.)
- Include measurable results where possible
- Professional tone suitable for Zambian job market
- Each point should be concise (1-2 lines)

Return ONLY the enhanced description as plain text with bullet points, no additional formatting."""

        # Generate response using REST API
        enhanced = call_gemini_api(prompt).strip()
        logger.info(f"Enhanced description for: {position}")
        
        return jsonify({'description': enhanced})
    
    except Exception as e:
        logger.error(f"Error enhancing description: {str(e)}")
        return jsonify({'error': 'Failed to enhance description', 'message': str(e)}), 500

@app.route('/api/ai/suggest-responsibilities', methods=['POST'])
def suggest_responsibilities():
    """Generate high-impact responsibility bullets using AI"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({'error': 'AI service not configured'}), 503
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        position = data.get('position', '').strip()
        company = data.get('company', '').strip()
        profession = data.get('profession', '').strip()
        yearsExperience = data.get('yearsExperience', 0)
        
        if not position:
            return jsonify({'error': 'Position is required'}), 400
        
        # Validate years_experience
        try:
            yearsExperience = int(yearsExperience)
        except (ValueError, TypeError):
            yearsExperience = 0
        
        # Create cache key
        cache_key = f"resp_{position}_{company}_{profession}".lower()
        
        # Check cache first with LRU tracking
        cached_resp = get_from_cache(ai_cache, cache_key)
        if cached_resp:
            logger.info(f"Cache hit for responsibilities: {position}")
            return jsonify({'responsibilities': cached_resp})
        
        # Generate prompt
        prompt = f"""Generate 4-6 high-impact responsibility bullet points for a CV:

Position: {position}
Company: {company}
{f"Profession: {profession}" if profession else ""}
{f"Experience Level: {yearsExperience} years" if yearsExperience else ""}

Requirements:
- Start each bullet with strong action verbs (Led, Managed, Developed, Implemented, Coordinated, Achieved, etc.)
- Include quantifiable results and metrics where possible (e.g., "Managed team of 15", "Reduced costs by 30%", "Served 100+ clients")
- Focus on achievements and impact, not just duties
- Professional tone suitable for Zambian job market
- Each bullet should be 1-2 lines maximum
- Relevant to the position and local context

Return as JSON array of strings:
["First responsibility", "Second responsibility", ...]

Return ONLY the JSON array, no markdown formatting or additional text."""

        # Generate response using REST API
        resp_text = call_gemini_api(prompt).strip()
        
        # Clean up markdown if present
        if resp_text.startswith('```json'):
            resp_text = resp_text.replace('```json', '').replace('```', '').strip()
        elif resp_text.startswith('```'):
            resp_text = resp_text.replace('```', '').strip()
        
        # Parse JSON
        responsibilities = json.loads(resp_text)
        
        # Cache the response with size limit
        add_to_cache(ai_cache, cache_key, responsibilities)
        logger.info(f"Generated responsibilities for: {position}")
        
        return jsonify({'responsibilities': responsibilities})
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}, Response: {resp_text}")
        return jsonify({'error': 'Failed to parse AI response'}), 500
    except Exception as e:
        logger.error(f"Error generating responsibilities: {str(e)}")
        return jsonify({'error': 'Failed to generate responsibilities', 'message': str(e)}), 500

# ======================= ADMIN ROUTES =======================

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Admin login endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Input validation - prevent injection
        if len(username) > 50 or len(password) > 100:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check credentials using context manager
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash FROM admins WHERE username = ?', (username,))
            result = cursor.fetchone()
        
        if not result or not check_password_hash(result['password_hash'], password):
            logger.warning(f"Failed login attempt for username: {username}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate JWT token
        access_token = create_access_token(identity=username)
        logger.info(f"Successful login: {username}")
        
        return jsonify({
            'success': True,
            'token': access_token,
            'username': username
        })
    
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/admin/analytics', methods=['GET'])
@jwt_required()
def get_analytics():
    """Get analytics data for admin dashboard"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Total transactions
            cursor.execute('SELECT COUNT(*) as total FROM transactions')
            total_transactions = cursor.fetchone()['total']
            
            # Completed transactions
            cursor.execute('SELECT COUNT(*) as completed FROM transactions WHERE status = "completed"')
            completed = cursor.fetchone()['completed']
            
            # Total revenue
            cursor.execute('SELECT COALESCE(SUM(amount), 0) as revenue FROM transactions WHERE status = "completed"')
            total_revenue = cursor.fetchone()['revenue']
            
            # Today's transactions
            cursor.execute('''
                SELECT COUNT(*) as today 
                FROM transactions 
                WHERE DATE(created_at) = DATE('now', 'localtime')
            ''')
            today_transactions = cursor.fetchone()['today']
            
            # This week's transactions
            cursor.execute('''
                SELECT COUNT(*) as week 
                FROM transactions 
                WHERE created_at >= DATE('now', 'localtime', '-7 days')
            ''')
            week_transactions = cursor.fetchone()['week']
            
            # Payment method breakdown
            cursor.execute('''
                SELECT payment_method, COUNT(*) as count 
                FROM transactions 
                WHERE status = "completed"
                GROUP BY payment_method
            ''')
            payment_methods = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({
            'totalTransactions': total_transactions,
            'completedTransactions': completed,
            'totalRevenue': total_revenue,
            'todayTransactions': today_transactions,
            'weekTransactions': week_transactions,
            'paymentMethods': payment_methods
        })
    
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({'error': 'Failed to fetch analytics'}), 500

@app.route('/api/admin/customers', methods=['GET'])
@jwt_required()
def get_customers():
    """Get all customer transactions"""
    try:
        # Get query parameters for pagination and search
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(100, max(1, int(request.args.get('per_page', 20))))
        search = request.args.get('search', '').strip()[:100]  # Limit search length
        
        offset = (page - 1) * per_page
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Build query
            if search:
                query = '''
                    SELECT id, transaction_id, phone_number, payment_method, amount, 
                           status, created_at, completed_at, pdf_filename
                    FROM transactions
                    WHERE phone_number LIKE ? OR transaction_id LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                '''
                params = (f'%{search}%', f'%{search}%', per_page, offset)
            else:
                query = '''
                    SELECT id, transaction_id, phone_number, payment_method, amount, 
                           status, created_at, completed_at, pdf_filename
                    FROM transactions
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                '''
                params = (per_page, offset)
            
            cursor.execute(query, params)
            customers = [dict(row) for row in cursor.fetchall()]
            
            # Get total count
            if search:
                cursor.execute('''
                    SELECT COUNT(*) as total FROM transactions
                    WHERE phone_number LIKE ? OR transaction_id LIKE ?
                ''', (f'%{search}%', f'%{search}%'))
            else:
                cursor.execute('SELECT COUNT(*) as total FROM transactions')
            
            total = cursor.fetchone()['total']
        
        return jsonify({
            'customers': customers,
            'total': total,
            'page': page,
            'perPage': per_page,
            'totalPages': (total + per_page - 1) // per_page
        })
    
    except ValueError:
        return jsonify({'error': 'Invalid pagination parameters'}), 400
    except Exception as e:
        logger.error(f"Customers error: {str(e)}")
        return jsonify({'error': 'Failed to fetch customers'}), 500

@app.route('/api/admin/customer/<transaction_id>', methods=['GET'])
@jwt_required()
def get_customer_details(transaction_id):
    """Get detailed customer CV data"""
    try:
        # Validate transaction ID
        if not validate_transaction_id(transaction_id):
            return jsonify({'error': 'Invalid transaction ID'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT transaction_id, phone_number, payment_method, amount, 
                       status, cv_data, pdf_filename, created_at, completed_at
                FROM transactions
                WHERE transaction_id = ?
            ''', (transaction_id,))
            
            result = cursor.fetchone()
        
        if not result:
            return jsonify({'error': 'Customer not found'}), 404
        
        customer = dict(result)
        
        # Parse CV data if it exists
        if customer['cv_data']:
            try:
                customer['cv_data'] = json.loads(customer['cv_data'])
            except json.JSONDecodeError:
                logger.error(f"Invalid CV data for transaction: {transaction_id}")
                customer['cv_data'] = None
        
        return jsonify(customer)
    
    except Exception as e:
        logger.error(f"Customer details error: {str(e)}")
        return jsonify({'error': 'Failed to fetch customer details'}), 500

@app.route('/api/admin/change-password', methods=['POST'])
@jwt_required()
def change_admin_password():
    """Change admin password"""
    try:
        current_user = get_jwt_identity()
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        old_password = data.get('oldPassword', '')
        new_password = data.get('newPassword', '')
        
        if not old_password or not new_password:
            return jsonify({'error': 'Old and new passwords required'}), 400
        
        # Password strength check
        if len(new_password) < 8:
            return jsonify({'error': 'New password must be at least 8 characters'}), 400
        
        # Verify old password and update
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash FROM admins WHERE username = ?', (current_user,))
            result = cursor.fetchone()
            
            if not result or not check_password_hash(result['password_hash'], old_password):
                return jsonify({'error': 'Invalid current password'}), 401
            
            # Update password
            new_hash = generate_password_hash(new_password)
            cursor.execute('UPDATE admins SET password_hash = ? WHERE username = ?', 
                          (new_hash, current_user))
        
        logger.info(f"Password changed for user: {current_user}")
        return jsonify({'success': True, 'message': 'Password updated successfully'})
    
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        return jsonify({'error': 'Failed to change password'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting CV Generator Backend with Admin Panel...")
    logger.info(f"📊 Database: {DB_PATH}")
    logger.info(f"🌍 API Base URL: {API_BASE_URL}")
    logger.info(f"🔒 CORS Allowed Origins: {ALLOWED_ORIGINS}")
    logger.info("🔒 Admin login available at: /admin")
    
    # Get port from environment variable (Heroku sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Check if running in production
    if os.getenv('FLASK_ENV') == 'production':
        logger.info("⚠️  Running in PRODUCTION mode")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        logger.warning("🚫 Running in DEVELOPMENT mode")
        app.run(debug=True, port=5000)
