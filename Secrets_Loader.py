import os
from encrypt_secrets import decrypt_secrets
from dotenv import load_dotenv

def load_secrets():
    """Load secrets into environment variables"""
    
    # First try .env file (for development)
    load_dotenv()
    
    # If secrets are already in environment, use them
    if os.getenv('OPENAI_API_KEY'):
        print("Using environment variables...")
        return
    
    # Otherwise, decrypt from encrypted file
    print("Decrypting secrets from encrypted file...")
    secrets = decrypt_secrets()
    
    if secrets:
        for key, value in secrets.items():
            os.environ[key] = value
        print(f"Loaded {len(secrets)} secrets into environment")
    else:
        raise Exception("Failed to load secrets")

# Usage in your main app
if __name__ == "__main__":
    load_secrets()
    
    # Now you can use the secrets
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"API Key loaded: {api_key[:10]}...")