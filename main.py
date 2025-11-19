## main.py

from Secrets_Loader import load_secrets
import Extractor
import Loading_data
# from Extractor import PreMadeSportsExtractor
# from  Loading_data import CompleteSportsDataUploader_Algolia
import os


## load api keys
load_secrets()

## Now can use API keys safely
algolia_app_id = os.getenv("algolia_app_id")
algolia_api_key = os.getenv("algolia_api_key")
openai_api_key = os.getenv("openai_api_key")


## Upload Algolia Data
print("Loading data and extractor...")
upload_data = Loading_data.CompleteSportsDataUploader_Algolia(algolia_app_id, algolia_api_key)
extractor_1 = Extractor.PreMadeSportsExtractor(algolia_app_id, algolia_api_key, openai_api_key, n_gram = False)
extractor_2 = Extractor.PreMadeSportsExtractor(algolia_app_id, algolia_api_key, openai_api_key, n_gram = True)


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
        return None


## Loop until the user is done giving prompt
another_prompt = True
while another_prompt:

    ## does user want 2-grams?
    n_grams_bool = None
    while n_grams_bool is None:
        n_grams_bool = user_input_prompting_TorF("Would you like to incorporate 2-grams (more accurate, but slower)?")


    ## feedback
    feedback = None
    while feedback is None:
        feedback = user_input_prompting_TorF("Would you like to give feedback on results to help improve the model?")


    prompt = input("Please enter your prompt here: ")

    if feedback == False:
        if n_grams_bool == True:
            results = extractor_2.extract(prompt)
            extractor_2._nice_display(results)
        else:
            results = extractor_1.extract(prompt)
            extractor_1._nice_display(results)


    if feedback == True:
        if n_grams_bool == True:
            results = extractor_2.interactive_extraction(prompt)
        else:
            results = extractor_1.interactive_extraction(prompt)
        

    another_prompt = None
    while another_prompt is None:
        another_prompt = user_input_prompting_TorF("Input another prompt?")



