"""logger.py — Structured logging for the AQI app."""
import logging, sys

def get_logger(name="aqi_app"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger

log = get_logger()
