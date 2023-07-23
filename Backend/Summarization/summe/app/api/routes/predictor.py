from transformers import pipeline
from fastapi import APIRouter, HTTPException

from services.predict import MachineLearningModelHandlerT5 as model_T5
from services.predict import MachineLearningModelHandlerPolyglot as model_polyglot

from models.prediction import (
    TextDataInput,
    SummerizedResponse,
    HealthResponse,
)

router = APIRouter()


def get_summerization(model_name, data_point):
    if model_name == "t5":
        model = model_T5()
    elif model_name == "polyglot":
        model = model_polyglot()
    
    return model.predict(data_point, load_wrapper=pipeline, method="predict")


def get_model_health(model_name):
    if model_name == "t5":
        model = model_T5()
    elif model_name == "polyglot":
        model = model_polyglot()
    
    return model.get_model_status()


@router.post(
    "/summerize/{model_name}",
    response_model=SummerizedResponse,
    name="summerize news",
    description="모델을 이용하여 기사를 요약하는 API 입니다. `model_name`에는 `t5`, `polyglot` 중 하나를 선택하시면 됩니다. \
                `num_split`은 t5 모델에서만 사용되며, `options`의 내용은 \
                [link](https://huggingface.co/docs/transformers/v4.30.0/main_classes/text_generation#transformers.GenerationConfig)를 참조하시기 바랍니다. "
)
async def summerize(model_name: str, data_input: TextDataInput):
    if not model_name in ["t5", "polyglot"]:
        raise HTTPException(status_code=404, detail="'model_name' argument invalid! Type model name. \n Model: t5, polyglot")

    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        data_point = data_input.get_all_info()
        arr_para, arr_para_sum = get_summerization(model_name, data_point)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return SummerizedResponse(arr_split=arr_para, arr_summerized=arr_para_sum)


@router.get(
    "/summerize/{model_name}/health",
    response_model=HealthResponse,
    description="모델이 GPU에 로드되어있는지 확인하는 API 입니다. `model_name`에는 `t5`, `polyglot` 중 하나를 선택하시면 됩니다. \
                `load` 옵션을 이용하여 모델을 GPU에 로드시킬 수 있습니다. "
)
async def health_check(model_name: str, load: bool=False):
    if not model_name in ["t5", "polyglot"]:
        raise HTTPException(status_code=404, detail="'model_name' argument invalid! Type model name. \n Model: t5, polyglot")

    if load:
        get_summerization(model_name, {"prompt": "", "contents": "", "options": {}, "num_split": 1})
        
        return HealthResponse(loaded=True)
    else:
        return HealthResponse(loaded=get_model_health(model_name))

