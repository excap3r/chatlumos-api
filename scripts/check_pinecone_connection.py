#!/usr/bin/env python3
"""
Utility script to check Pinecone API connection and basic functionality.
This is intended for manual execution or deployment checks, not for automated testing.
"""

import os
import sys
from dotenv import load_dotenv
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Use ConsoleRenderer for development/scripts
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("check_pinecone")

# Load environment variables
load_dotenv()

# Get Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    logger.error("PINECONE_API_KEY not found in environment variables. Cannot proceed.")
    sys.exit(1)

logger.info("Attempting to connect to Pinecone...")

try:
    # Import pinecone
    import pinecone
    logger.info("Successfully imported 'pinecone' library.")
    
    # Try to get version
    try:
        version = pinecone.__version__
        logger.info(f"Pinecone library version: {version}")
    except:
        logger.warning("Could not determine Pinecone library version.")
    
    # Try initializing
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    logger.info("Successfully initialized pinecone.Pinecone client.")
    
    # Try listing indexes
    try:
        indexes = pc.list_indexes()
        logger.info("Successfully listed indexes.", index_count=len(indexes), indexes=indexes)
    except Exception as e:
        logger.error("Failed to list Pinecone indexes.", error=str(e), exc_info=True)
    
except ImportError as e:
    logger.error("Failed to import the 'pinecone' library. Is it installed?", error=str(e), exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error("An unexpected error occurred during Pinecone check.", error=str(e), exc_info=True)
    sys.exit(1)

logger.info("Pinecone connection check complete.")