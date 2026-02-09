"""
Pytest configuration for integration tests.

Loads .env file so tests can access LM Studio server configuration.
"""

from dotenv import load_dotenv


def pytest_configure(config):
    """Load .env file before tests run."""
    load_dotenv()
