"""
Environment variable loader with secure token handling.
"""

import os
from pathlib import Path
from typing import Optional

def load_api_token() -> Optional[str]:
    """
    Load HUD API token from environment variable or .env file.
    Returns None if token is not found.
    """
    # First try environment variable
    token = os.getenv('HUD_API_TOKEN')
    if token:
        return token
        
    # Try .env file in project root
    try:
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('HUD_API_TOKEN='):
                        return line.split('=', 1)[1].strip()
    except Exception:
        pass
        
    return None
