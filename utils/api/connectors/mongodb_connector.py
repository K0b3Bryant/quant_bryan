from pymongo import MongoClient
from .base import DatabaseConnector

class MongoConnector(DatabaseConnector):
    def connect(self):
        self.client = MongoClient(self._config['connection_string'])
        self.connection = self.client[self._config['database']]
        print("MongoDB connection established.")
        return self
    
    def disconnect(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

    def execute_query(self, collection_name: str, find_filter: dict = None):
        # In NoSQL, "query" is different. We adapt the method.
        collection = self.connection[collection_name]
        return list(collection.find(find_filter or {}))
