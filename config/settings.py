"""
Application Configuration Settings
Centralized configuration for easy management and scalability.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Base configuration class."""
    
    # Flask Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "automind-secret-key-change-in-production")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Gemini API Settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set in .env file
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Agent Settings
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "True").lower() == "true"
    
    # Search Settings
    SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    
    # Upload Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    AGENT_VERBOSE = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    AGENT_VERBOSE = False


# Configuration dictionary for easy access
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}


def get_config():
    """Get configuration based on environment."""
    env = os.getenv("FLASK_ENV", "development")
    return config_by_name.get(env, DevelopmentConfig)
