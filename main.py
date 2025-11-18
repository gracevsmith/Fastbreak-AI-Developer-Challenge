## main.py

from Secrets_Loader import load_secrets
import Extractor
import Loading_data
# from Extractor import PreMadeSportsExtractor
# from  Loading_data import CompleteSportsDataUploader_Algolia
import os



load_secrets()

## Now can use API keys safely
algolia_app_id = os.getenv("algolia_app_id")
algolia_api_key = os.getenv("algolia_api_key")
openai_api_key = os.getenv("openai_api_key")

upload_data = Loading_data.CompleteSportsDataUploader_Algolia(algolia_app_id, algolia_api_key)

def user_input_prompting_TorF(question):
    T_F = input(f"{question} \n Enter a value (True/False): ").lower().strip()
    if T_F == "true":
        is_true = True
        return is_true
    elif T_F == "false":
        is_true = False
        return is_true
    else:
        print("Invalid input. Please enter 'True' or 'False'.")

## does user want 2-grams
n_grams_bool = user_input_prompting_TorF("Would you like to incorporate 2-grams (more accurate, but slower)?")

## instance of extractor with user's n-gram choice
extractor = Extractor.PreMadeSportsExtractor(algolia_app_id, algolia_api_key, openai_api_key, n_gram = n_grams_bool)

## feedback
feedback = user_input_prompting_TorF("Would you like to give feedback on results to help improve the model?")
prompt = input("Please enter your prompt here: ")

if feedback == False:
    results = extractor.extract(prompt)

if feedback == True:
    results = extractor.interactive_extraction(prompt)

print(results)


