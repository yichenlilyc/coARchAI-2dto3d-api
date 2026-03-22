import os
import json
import base64
from firebase_admin import credentials, firestore, storage, initialize_app

# Decode the secret key from the environment 
# We will convert your .json file to a base64 string later so you can paste it 
firebase_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_B64")

if firebase_b64:
    # Decode the base64 string back into a JSON object
    cert_dict = json.loads(base64.b64decode(firebase_b64).decode('utf-8'))
    cred = credentials.Certificate(cert_dict)
    
    # Initialize the app (Replace with your actual Firebase Storage bucket URL)
    initialize_app(cred, {
        'storageBucket': 'coarchai-f7ac6.firebasestorage.app' 
    })
    
    db = firestore.client()
    bucket = storage.bucket()
    print("Firebase connected successfully!")
else:
    print("WARNING: No Firebase credentials found in environment.")
    db = None
    bucket = None

def get_db():
    return db

def get_bucket():
    return bucket