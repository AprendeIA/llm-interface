"""
Logging Configuration Examples for LLM Interface

This module demonstrates various logging configurations including:
- Basic logging setup
- API key filtering for security
- Structured logging with rotation
- Provider-specific logging
"""

import logging
import logging.config
import re
import os
from llm_interface import LLMManager, LLMConfig, ProviderType


class APIKeyFilter(logging.Filter):
    """Filter to mask API keys and sensitive data in log messages.
    
    This filter protects against accidental API key exposure by masking:
    - OpenAI-style keys (sk-...)
    - API key assignments in various formats
    - Bearer tokens
    - Other sensitive patterns
    """
    
    def filter(self, record):
        """Mask sensitive data in log record."""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            
            # Mask OpenAI-style keys
            msg = re.sub(
                r'(sk-[a-zA-Z0-9]{20,})',
                'sk-***REDACTED***',
                msg
            )
            
            # Mask API key assignments
            msg = re.sub(
                r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'\\s]+)',
                r'\1***REDACTED***',
                msg,
                flags=re.IGNORECASE
            )
            
            # Mask Bearer tokens
            msg = re.sub(
                r'(Bearer\s+)([^\s]+)',
                r'\1***REDACTED***',
                msg,
                flags=re.IGNORECASE
            )
            
            # Mask Azure keys (longer hex strings)
            msg = re.sub(
                r'\b([a-f0-9]{32,})\b',
                '***REDACTED***',
                msg,
                flags=re.IGNORECASE
            )
            
            record.msg = msg
        return True


def setup_basic_logging():
    """Setup basic logging configuration.
    
    Good for: Development, simple applications
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('llm_interface.log'),
            logging.StreamHandler()
        ]
    )
    
    # Enable debug for LLM Interface
    logging.getLogger('llm_interface').setLevel(logging.DEBUG)
    
    print("✅ Basic logging configured")


def setup_secure_logging():
    """Setup logging with API key filtering.
    
    Good for: Production environments, shared logs
    """
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('llm_interface_secure.log'),
            logging.StreamHandler()
        ]
    )
    
    # Add API key filter to all handlers
    api_filter = APIKeyFilter()
    for handler in logging.root.handlers:
        handler.addFilter(api_filter)
    
    # Also add to LLM Interface logger
    logger = logging.getLogger('llm_interface')
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.addFilter(api_filter)
    
    print("✅ Secure logging configured with API key filtering")


def setup_structured_logging():
    """Setup structured logging with rotation and detailed formatting.
    
    Good for: Production applications, debugging, monitoring
    """
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': (
                    '%(asctime)s - %(name)s - %(levelname)s - '
                    '%(funcName)s:%(lineno)d - %(message)s'
                )
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            },
            'json': {
                'format': (
                    '{"timestamp": "%(asctime)s", "name": "%(name)s", '
                    '"level": "%(levelname)s", "message": "%(message)s"}'
                )
            }
        },
        'filters': {
            'api_key_filter': {
                '()': APIKeyFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout',
                'filters': ['api_key_filter']
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'llm_interface_detailed.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'filters': ['api_key_filter']
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': 'llm_interface_errors.log',
                'filters': ['api_key_filter']
            }
        },
        'loggers': {
            'llm_interface': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'llm_interface.providers': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)
    print("✅ Structured logging configured with rotation and filtering")


def demo_logging():
    """Demonstrate logging with the LLM Interface."""
    logger = logging.getLogger('llm_interface')
    
    logger.info("Starting LLM Interface demo")
    
    try:
        # Create manager
        manager = LLMManager()
        logger.info(f"Manager created with {len(manager.list_providers())} providers")
        
        # Simulate adding a provider (won't work without real API key)
        # This demonstrates that API keys would be filtered in logs
        logger.debug("Attempting to add provider with API key: sk-1234567890abcdef")
        logger.info("Provider configuration loaded (API key: ***filtered***)")
        
        # List providers
        providers = manager.list_providers()
        logger.info(f"Available providers: {providers}")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
    
    logger.info("Demo completed")


def main():
    """Run logging examples."""
    print("\n=== LLM Interface Logging Examples ===\n")
    
    # Example 1: Basic Logging
    print("1. Basic Logging Setup")
    setup_basic_logging()
    demo_logging()
    
    # Clear handlers for next example
    logging.root.handlers.clear()
    logging.getLogger('llm_interface').handlers.clear()
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Secure Logging
    print("2. Secure Logging with API Key Filtering")
    setup_secure_logging()
    demo_logging()
    
    # Clear handlers for next example
    logging.root.handlers.clear()
    logging.getLogger('llm_interface').handlers.clear()
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Structured Logging
    print("3. Structured Logging with Rotation")
    setup_structured_logging()
    demo_logging()
    
    print("\n=== Logging Examples Complete ===")
    print("\nLog files created:")
    for filename in ['llm_interface.log', 'llm_interface_secure.log', 
                     'llm_interface_detailed.log', 'llm_interface_errors.log']:
        if os.path.exists(filename):
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
