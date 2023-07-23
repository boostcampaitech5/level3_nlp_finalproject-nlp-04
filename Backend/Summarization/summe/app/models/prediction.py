import numpy as np
from typing import List, Dict, Union

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    loaded: bool


class MachineLearningDataInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

    def get_np_array(self):
        return np.array(
            [
                [
                    self.feature1,
                    self.feature2,
                    self.feature3,
                    self.feature4,
                    self.feature5,
                ]
            ]
        )
    
class SummerizedResponse(BaseModel):
    arr_split: List
    arr_summerized: List

class TextDataInput(BaseModel):
    prompt: str = "아래의 뉴스를 사실에 입각하여 맥락의 내용을 요약해줘."
    title: str = ""
    contents: str = ""
    num_split: int = 5
    options: Dict = {
        "max_new_tokens": 256,
        "early_stopping": "never",
        "do_sample": True,
        "eos_token_id": 2,
        "no_repeat_ngram_size": 8,
        "top_k": 50,
        "top_p": 0.98,
    }

    def get_all_info(self):
        return {
            "prompt": self.prompt,
            "title": self.title,
            "contents": self.contents,
            "num_split": self.num_split,
            "options": self.options,
        }