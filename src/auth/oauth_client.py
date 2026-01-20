import requests 
import sys
import os
from requests.auth import HTTPBasicAuth

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from config import CLIENT_ID, CLIENT_SECRET, TOKEN_URL 

class OAuthClient:
    def __init__(self) -> None:
        self.token = None 
    
    def fetch_token(self):
        response = requests.post(
            TOKEN_URL, 
            auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET), 
            data = {'grant_type': 'client_credentials'} 
        )
        if response.status_code == 200:
            self.token = response.json()
            return self.token
        else:
            raise Exception(f"Failed to fetch token:{response.status_code}  {response.text}")
        
    def get_token(self):
        if not self.token:
            raise Exception("Token is missing. Please fetch token first")
        return self.token
    
    def make_authorized_request(self, url):
        if not self.token:
            raise Exception("Token is missing. Please fetch token first")
        
        headers = {
            'Authorization': f"Bearer {self.token['access_token']}",
            'AI-Resource-Group': 'pmge-llm'
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 401: #token exipred
            self.fetch_token()
            return self.make_authorized_request(self, url)
            
        return response.json()
    
    
