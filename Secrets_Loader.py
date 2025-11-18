import os
import Encrypt_API_Codes
from dotenv import load_dotenv


def load_secrets():
    """
    Load secrets into environment variables
    """
    
    # First try .env file (for development)
    load_dotenv()
    
    # If secrets are already in environment, use them
    if os.getenv("algolia_app_id") and os.getenv("algolia_api_key") and os.getenv("openai_api_key"):
        print("Using environment variables...")
        return
    
    # Otherwise, decrypt from encrypted file
    print("Decrypting secrets from encrypted file...")
    secrets = Encrypt_API_Codes.decrypt_secrets()
    
    if secrets:
        for key, value in secrets.items():
            os.environ[key] = value
        print(f"Loaded {len(secrets)} secrets into environment")
    else:
        raise Exception("Failed to load secrets")