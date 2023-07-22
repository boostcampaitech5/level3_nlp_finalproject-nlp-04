import sys

from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

sys.path.append(str(Path.home().joinpath("level3_nlp_finalproject-nlp-04")))
from keyword_extractor.model import KeyBert

app = FastAPI()

@app.get("/")
def hello_word():
	return {"hello": "world"}