import os
import re

from loguru import logger

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from kss import split_sentences

from core.errors import PredictException, ModelLoadException
from core.config import MODEL_NAME_T5, MODEL_NAME_POLYGLOT, MODEL_PATH

class MachineLearningModelHandler(object):
    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @classmethod
    def get_model_status(cls):
        return True if cls.model != None else False


class MachineLearningModelHandlerT5(MachineLearningModelHandler):
    model = None

    @classmethod
    def predict(cls, input, load_wrapper=None, method="predict"):
        if cls.model == None:
            cls.get_model(load_wrapper)
            
        n = input["num_split"]
        
        # 2개 이상의 공백 제거. 
        input["contents"] = re.sub(r'\s{2,}', ' ', input["contents"])

        news_split = split_sentences(input["contents"])
        arr_para = []
        arr_para_sum = []

        for i in range(0, len(news_split) // n + 1):
            arr_para.append("".join(news_split[i * n:i * n + n]))

        clf = cls.get_model(load_wrapper)
        if hasattr(clf, method):
            for para in arr_para:
                # news_tokenized = len(clf.tokenizer(para)["input_ids"])
                # arr_para_sum.append(clf(para, min_length=news_tokenized // 2, max_length=news_tokenized // 2 + (news_tokenized // 4))[0]["summary_text"])
                text = clf(para, **input["options"],)[0]["summary_text"]
                arr_para_sum.append(cls.clean_paragraph(text))

            return arr_para, arr_para_sum
        raise PredictException(f"'{method}' attribute is missing")

    @staticmethod
    def load(load_wrapper):
        model = None
        if MODEL_PATH.endswith("/"):
            path = f"{MODEL_PATH}{MODEL_NAME_T5}"
        else:
            path = f"{MODEL_PATH}/{MODEL_NAME_T5}"
        if not os.path.exists(path):
            message = f"Machine learning model at {path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        model = load_wrapper("summarization", path)
        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)
        return model

    @classmethod
    def clean_paragraph(cls, text) -> str:
        list_sentence = split_sentences(text)
        str_clean = ""

        for sentence in list_sentence:
            if "다." in sentence:
                str_clean = sentence + " "
                break
        
        return str_clean

class MachineLearningModelHandlerPolyglot(MachineLearningModelHandler):
    model = None
    tokenizer = None

    prompt = lambda prompt, document: f"### 질문: {prompt}\n\n###맥락: {document}\n\n### 답변:"

    @classmethod
    def predict(cls, input, load_wrapper=None, method="predict"):
        if cls.model == None:
            cls.get_model(load_wrapper)

        x_prompt = cls.prompt(input["prompt"], input["contents"])
        
        result = cls.model.generate(
            **cls.tokenizer(
                x_prompt,
                return_tensors='pt',
                return_token_type_ids=False
            ),
            **input["options"],
        )
        result = cls.tokenizer.decode(result[0])[len(x_prompt):]

        arr_para = [x_prompt]
        arr_para_sum = [cls.clean_paragraph(result)]

        return arr_para, arr_para_sum
        # raise PredictException(f"'{method}' attribute is missing")

    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model, cls.tokenizer = cls.load(load_wrapper)
        return cls.model, cls.tokenizer

    @staticmethod
    def load(load_wrapper):
        model = None
        if MODEL_PATH.endswith("/"):
            path = f"{MODEL_PATH}{MODEL_NAME_POLYGLOT}"
        else:
            path = f"{MODEL_PATH}/{MODEL_NAME_POLYGLOT}"
        if not os.path.exists(path):
            message = f"Machine learning model at {path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        config = PeftConfig.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)
        return model, tokenizer
    
    @classmethod
    def clean_paragraph(cls, text) -> str:
        list_sentence = list(dict.fromkeys(split_sentences(text)))
        str_clean = ""

        for sentence in list_sentence:
            if "다." in sentence:
                str_clean += sentence + " "
        
        return str_clean + "\n"