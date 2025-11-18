## main.py

from Secrets_Loader import load_secrets
import os

load_secrets()

## Now can use API keys safely
algolia_app_id = os.getenv("algolia_app_id")
algolia_api_key = os.getenv("algolia_api_key")
openai_api_key = os.getenv("openai_api_key")

