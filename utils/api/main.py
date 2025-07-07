from connection_factory import get_connection

def main():
    print("--- 1. Testing REST API Connection ---")
    try:
        # The 'with' statement automatically handles connect() and disconnect()
        with get_connection("json_placeholder_api") as api:
            # Get a list of posts
            posts = api.get("posts", params={'_limit': 5})
            print(f"Successfully fetched {len(posts)} posts. First post title:")
            print(f"  -> {posts[0]['title']}")

            # Get a specific user
            user = api.get("users/1")
            print(f"Successfully fetched user: {user['name']} ({user['email']})")

    except Exception as e:
        print(f"An error occurred with the API connection: {e}")

    print("\n--- 2. Testing PostgreSQL Connection ---")
    try:
        # NOTE: This will fail if you don't have a running PostgreSQL server
        # with the credentials from config.yaml
        with get_connection("my_postgres_db") as pg_db:
            # Example query, replace with your own table
            query = "SELECT version();" 
            results = pg_db.execute_query(query)
            print("Successfully executed query.")
            print("  -> PostgreSQL Version:", results[0]['version'])
            
    except Exception as e:
        print(f"Could not connect or query PostgreSQL. Is it running and configured correctly?")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
