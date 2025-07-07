import yaml

# Import all your connector classes
from connectors.api_connector import ApiConnector
from connectors.odbc_connector import OdbcConnector
from connectors.postgres_connector import PostgresConnector
from connectors.mongo_connector import MongoConnector

# Map the 'type' from the config file to the actual Python class
CONNECTOR_MAPPING = {
    "rest_api": ApiConnector,
    "odbc": OdbcConnector,
    "postgres": PostgresConnector,
    "mongodb": MongoConnector,
    # Add new mappings here as you create new connectors
}

def load_config(path="config.yaml"):
    """Loads the connection configurations from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_connection(connection_name: str, config_path="config.yaml"):
    """
    Factory function to get a connector instance by name.
    
    Args:
        connection_name (str): The name of the connection from the config file.
        config_path (str): The path to the configuration file.

    Returns:
        An instance of a BaseConnector subclass.
    """
    config = load_config(config_path)
    
    connection_config = config.get("connections", {}).get(connection_name)
    if not connection_config:
        raise ValueError(f"Connection '{connection_name}' not found in configuration.")

    conn_type = connection_config.get("type")
    if not conn_type:
        raise ValueError(f"Connection '{connection_name}' must have a 'type'.")

    ConnectorClass = CONNECTOR_MAPPING.get(conn_type)
    if not ConnectorClass:
        raise ValueError(f"Unknown connector type '{conn_type}'.")

    # Pass the specific connection config to the connector's constructor
    return ConnectorClass(**connection_config)
