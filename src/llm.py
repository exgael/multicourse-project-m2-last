import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abc import ABC, abstractmethod
from auth.oauth_client import OAuthClient


from config import get_deployment_url, output_format, message_format, get_number_of_tokens
import requests
import json 
import threading
import time


class LLM(ABC):
    def __init__(self,model = "gpt4o", max_tokens = 4000, temperature=0.0) -> None:
        
        self.model = model
        self.max_tokens =  max_tokens
        self.temperature =  temperature
        self.total_token_used = 0

        self.lock = threading.Lock()
        self.req_counter = 0
        self.starting_time = None



    def reset_token_counter(self):
        self.total_token_used = 0

    def get_token_counter(self):
        return self.total_token_used

    @abstractmethod
    def inference(self, messages, model=None):
        pass

class Llm_GenAI_Hub(LLM):
    def __init__(self, model="gpt4o", max_tokens=4000, temperature=0):
        super().__init__(model, max_tokens, temperature)
        self.auth_client = OAuthClient()
        token = self.auth_client.fetch_token()
        self.headers={
            'AI-Resource-Group': 'pmge-llm',
            'Content-Type': 'application/json',
            "Authorization": f'Bearer {token["access_token"]}', 
        }

    def update_token(self):
        token = self.auth_client.fetch_token()
        self.headers["Authorization"] = f'Bearer {token["access_token"]}'

    def inference(self, messages, model=None):
        sleep_time = 10
        self.req_counter += 1
        if self.starting_time is None:
            self.starting_time = time.time()

        if model is None:
            model = self.model
        
        data = message_format(messages, model, self.max_tokens, self.temperature)
        deployment_url = get_deployment_url(model)

        with self.lock:
            response = requests.post(
                deployment_url, 
                headers=self.headers, 
                data = json.dumps(data)
            )
            if response.status_code != 200:
                print(response)
                print(response.text)
            if response.status_code == 401: #token exipred
                self.update_token()
                response = requests.post(
                deployment_url, 
                headers=self.headers, 
                data = json.dumps(data)
                )
            while response.status_code == 429:
                print(f"limit reached after {self.req_counter} requests and {time.time() - self.starting_time} seconds. Sleeping {sleep_time} seconds...")
                time.sleep(sleep_time)
                sleep_time *= 2
                
                response = requests.post(
                deployment_url, 
                headers=self.headers, 
                data = json.dumps(data)
                )
        output = response.json()
        print("output", output)
        
        self.total_token_used += get_number_of_tokens(output, model)
        
        return output_format(output, model) 
        



def llm_constructor(model_name, max_tokens=4000, temperature=0.0):
    genai_hub_models = ['gpt4', 'claude3.5', 'claude3.7', 'gpt4o']
    
    if model_name in genai_hub_models:
        return Llm_GenAI_Hub(model_name, max_tokens, temperature)
    else: 
        raise Exception(f'Model name not known. Please enter a valid model name from the following lists:\nModels available on the SAP GenAI Hub:{genai_hub_models}.\n')