"""
Logger utility stub.
"""
import logging

def setup_logging(level=logging.INFO, **kwargs):
    logging.basicConfig(level=level, **kwargs)
    return logging.getLogger("steg")
