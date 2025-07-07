import pyodbc
from .base import DatabaseConnector

class OdbcConnector(DatabaseConnector):
    """Connects to any database with an ODBC driver."""
    def connect(self):
        try:
            conn_str = (
                f"DRIVER={self._config['driver']};"
                f"SERVER={self._config['server']};"
                f"DATABASE={self._config['database']};"
                f"UID={self._config['username']};"
                f"PWD={self._config['password']};"
            )
            self.connection = pyodbc.connect(conn_str)
            print(f"ODBC connection established to {self._config['server']}.")
            return self
        except Exception as e:
            print(f"Failed to connect via ODBC: {e}")
            raise

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("ODBC connection closed.")

    def execute_query(self, query: str):
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        cursor = self.connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        # Convert to a list of dicts for easier use
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
