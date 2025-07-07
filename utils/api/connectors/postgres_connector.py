import psycopg2
from psycopg2.extras import RealDictCursor
from .base import DatabaseConnector

class PostgresConnector(DatabaseConnector):
    """Connects specifically to a PostgreSQL database."""
    def connect(self):
        try:
            self.connection = psycopg2.connect(**self._config)
            print(f"PostgreSQL connection established to {self._config['host']}.")
            return self
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            raise

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("PostgreSQL connection closed.")
            
    def execute_query(self, query: str):
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        # RealDictCursor returns rows as dictionaries
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        return [dict(row) for row in results]
