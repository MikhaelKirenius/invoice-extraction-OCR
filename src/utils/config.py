import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.logger import default_logger as logger


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Optional path to configuration file
        """
        ROOT_DIR = Path(__file__).resolve().parents[2]
        default_config_path = ROOT_DIR / "config" / "config.yaml"

        self.config_path = Path(config_path) if config_path else default_config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)

config = Config()