from abc import ABC, abstractmethod

class BaseConnector(ABC):
    """
    Abstract Base Class for all connectors.
    It ensures that all connectors can be used as context managers.
    """
    def __init__(self, **kwargs):
        self.connection = None
        self._config = kwargs

    @abstractmethod
    def connect(self):
        """Establish the connection to the external source."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close the connection."""
        pass

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and ensure disconnection."""
        self.disconnect()

class DatabaseConnector(BaseConnector):
    """A specialized base class for database-like connectors."""
    @abstractmethod
    def execute_query(self, query: str):
        """Execute a query and return the results."""
        pass
