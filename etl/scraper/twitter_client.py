import os
import json
from typing import Dict
from dotenv import load_dotenv
from twikit import Client

def load_cookies() -> str:
    """Load and save cookies from the environment variable to a temporary file"""
    load_dotenv()
    try:
        cookie_json = os.getenv('EDITCOOKIE_JSON')
        if not cookie_json:
            raise ValueError("EDITCOOKIE_JSON not found in .env file")
        
        # Parse and reformat the cookies
        cookie_data = json.loads(cookie_json)
        formatted_cookies = {}
        for item in cookie_data:
            name = item.get("name")
            value = item.get("value")
            if name and value:
                formatted_cookies[name] = value

        # Save formatted cookies to a temporary file
        cookie_file = "./cookies.json"
        with open(cookie_file, "w", encoding="utf-8") as f:
            json.dump(formatted_cookies, f)  # Save as a key-value dictionary

        return cookie_file  # Return the file path

    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in EDITCOOKIE_JSON")
    except Exception as e:
        raise ValueError(f"Error loading cookies: {str(e)}")


def get_twitter_client() -> Client:
    """Initialize and return Twitter client with cookies"""
    cookie_file = load_cookies()  # Get the path to the saved cookie file
    client = Client(
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15'
    )
    client.load_cookies(cookie_file)  # Pass the file path
    return client

