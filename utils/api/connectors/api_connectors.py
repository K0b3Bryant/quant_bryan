import requests
from .base import BaseConnector

class ApiConnector(BaseConnector):
    """Connects to a REST API."""
    def connect(self):
        print(f"Initializing API connection to {self._config.get('base_url')}")
        self.session = requests.Session()
        if 'api_key' in self._config:
            self.session.headers.update({'Authorization': f"Bearer {self._config['api_key']}"})
        self.connection = True  # Mark as "connected"
        return self

    def disconnect(self):
        if self.session:
            self.session.close()
            print("API session closed.")
            self.connection = False
            
    def get(self, endpoint, params=None):
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        url = f"{self._config['base_url']}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()  # Raises an exception for bad status codes
        return response.json()
