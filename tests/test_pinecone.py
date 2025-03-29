#!/usr/bin/env python3
"""
Test file to check Pinecone API import and setup
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    print("ERROR: PINECONE_API_KEY not found in environment variables")
    exit(1)

print("Testing pinecone 6.0.2...")

try:
    # Import pinecone
    import pinecone
    print(f"SUCCESS: 'import pinecone' works")
    
    # Try to get version
    try:
        version = pinecone.__version__
        print(f"Pinecone version: {version}")
    except:
        print("Could not get Pinecone version")
    
    # Try initializing
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    print("SUCCESS: pinecone.Pinecone() initialized")
    
    # Try listing indexes
    try:
        indexes = pc.list_indexes()
        print(f"SUCCESS: Listed indexes: {indexes}")
    except Exception as e:
        print(f"ERROR listing indexes: {e}")
    
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")

print("Test complete") 