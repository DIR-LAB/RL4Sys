import json
from typing import Dict, Any

class AgentConfigLoader:
    def __init__(self, config_path: str):
        """
        Initialize the AgentConfigLoader with a path to a configuration file.
        
        Args:
            config_path (str): Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Extract required fields
        self.client_id = self.config.get('client_id')
        self.algorithm_name = self.config.get('algorithm_name')
        self.algorithm_parameters = self.config.get('algorithm_parameters', {})
        self.train_server_address = self.config.get('train_server_address')
        self.send_frequency = self.config.get('send_frequency')
        self.algorithm_type = self.config.get('type')
        
        # Validate required fields
        if not self.client_id:
            raise ValueError("Missing 'client_id' in configuration file")
        if not self.algorithm_name:
            raise ValueError("Missing 'algorithm_name' in configuration file")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse the JSON configuration file.
        
        Returns:
            dict: The parsed configuration
        """
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {self.config_path}")
    
    def get_client_id(self) -> str:
        """Get the client ID from the configuration."""
        return self.client_id
    
    def get_algorithm_name(self) -> str:
        """Get the algorithm name from the configuration."""
        return self.algorithm_name
    
    def get_algorithm_parameters(self) -> Dict[str, Any]:
        """Get the algorithm parameters from the configuration."""
        return self.algorithm_parameters
    
    def get_train_server_address(self) -> str:
        """Get the train server address from the configuration."""
        return self.train_server_address
    
    def get_send_frequency(self) -> int:
        """Get the send frequency from the configuration."""
        return self.send_frequency 
    
    def get_algorithm_type(self) -> str:
        """Get the algorithm type from the configuration."""
        return self.algorithm_type
