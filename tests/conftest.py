"""
conftest.py - pytest configuration for architect-agent tests

Adds project root to sys.path so 'agent' module can be imported.
Loads environment variables from .env for API key.
"""

import sys
import os
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env (GOOGLE_API_KEY)
load_dotenv(os.path.join(project_root, ".env"))