# # config.py
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Default API Keys
# DEFAULT_GROQ_API_KEY = "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"

# class Config:
#     # API Keys
#     GROQ_API_KEY = os.getenv('GROQ_API_KEY', DEFAULT_GROQ_API_KEY)
    
#     # Server Configuration
#     PORT = int(os.getenv('PORT', 5000))
#     DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
#     # CORS Settings
#     CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')

#     @staticmethod
#     def get_groq_key(custom_key=None):
#         """Get GROQ API key - either custom or from environment"""
#         return custom_key if custom_key else Config.GROQ_API_KEY

# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Server Configuration
    PORT = int(os.getenv('PORT', 10000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # CORS Settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')

    @staticmethod
    def get_groq_key(custom_key=None):
        """Get GROQ API key - either custom or from environment"""
        if custom_key:
            return custom_key
        if not Config.GROQ_API_KEY:
            raise ValueError("No GROQ API key provided")
        return Config.GROQ_API_KEY