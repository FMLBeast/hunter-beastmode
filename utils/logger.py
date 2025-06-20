"""
Logger utility stub.
"""
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("steg")
