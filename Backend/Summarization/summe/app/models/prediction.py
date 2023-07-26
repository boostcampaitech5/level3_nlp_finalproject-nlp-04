from typing import List, Dict

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    loaded: bool

    
class SummerizedResponse(BaseModel):
    arr_split: List
    arr_summerized: List
    

class TextDataInput(BaseModel):
    prompt: str = "아래의 뉴스를 사실에 입각하여 거짓된 내용 없이 맥락의 내용을 요약해줘."
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
        "temperature": 0.1,
    }

    def get_all_info(self):
        return {
            "prompt": self.prompt,
            "title": self.title,
            "contents": self.contents,
            "num_split": self.num_split,
            "options": self.options,
        }