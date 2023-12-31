import sys
import logging

from loguru import logger
from starlette.config import Config
from starlette.datastructures import Secret

from core.logging import InterceptHandler

config = Config(".env")

API_PREFIX = "/api"
VERSION = "0.1.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
SECRET_KEY: Secret = config("SECRET_KEY", cast=Secret, default="")

PROJECT_NAME: str = config("PROJECT_NAME", default="Summe")

# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

# 모델이 저장되어 있어야 합니다. 
MODEL_PATH = config("MODEL_PATH", default="./ml/model/")
MODEL_NAME_T5 = config("MODEL_NAME_T5", default="summary_with_news/checkpoint-11000")
MODEL_NAME_POLYGLOT = config("MODEL_NAME_POLYGLOT", default="summary_with_news/checkpoint-11000")
INPUT_EXAMPLE = config("INPUT_EXAMPLE", default="./ml/model/examples/example.json")
