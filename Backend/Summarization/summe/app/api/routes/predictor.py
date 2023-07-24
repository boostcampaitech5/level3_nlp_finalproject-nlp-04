from transformers import pipeline
from fastapi import APIRouter, HTTPException

from services.predict import MachineLearningModelHandlerT5 as model_T5
from services.predict import MachineLearningModelHandlerPolyglot as model_polyglot

from typing import Tuple

from models.prediction import (
    TextDataInput,
    SummerizedResponse,
    HealthResponse,
)

router = APIRouter()


def get_summerization(model_name: str, data_point: TextDataInput) -> Tuple:
    """선택된 모델과 입력값을 이용하여 결과를 반환하는 함수. 

    Args:
        model_name (str): 모델의 이름. 
        data_point (TextDataInput): HTTP Request로 받은 정보. 

    Returns:
        Tuple: 모델의 사용된 구문과 결과. 
    """
    if model_name == "t5":
        model = model_T5()
    elif model_name == "polyglot":
        model = model_polyglot()
    
    return model.predict(data_point, load_wrapper=pipeline, method="predict")


def get_model_health(model_name: str) -> bool:
    """모델의 GPU 로드 여부 확인용 함수. 

    Args:
        model_name (str): 모델의 이름. 

    Returns:
        bool: 모델의 GPU 로드 여부. 
    """

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
async def summerize(model_name: str, data_input: TextDataInput) -> SummerizedResponse:
    """선택된 모델을 이용하여 주어진 기사를 요약 및 반환을 위한 HTTP Request 처리 함수. 

    Args:
        model_name (str): 모델의 이름. 
        data_input (TextDataInput): HTTP Request로 받은 정보. 

    Raises:
        HTTPException: 운용중인 모델의 이름 외의 모델 요청이 오는경우. 
        HTTPException: 동작시 필요한 JSON 입력이 없는 경우. 
        HTTPException: 위의 예외 밖의 다른 에러가 발생하는 경우. 

    Returns:
        SummerizedResponse: 모델에서 추출된 결과를 반환하는 함수. 
    """

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
async def health_check(model_name: str, load: bool=False) -> HealthResponse:
    """모델의 상태에 확인과 GPU 로드를 위해 HTTP Request를 처리하는 함수. 

    Args:
        model_name (str): 모델의 이름. 
        load (bool, optional): 모델의 GPU 로드와 관계 없이 강제로 GPU에 로드 시도 여부. (기본값: False)

    Raises:
        HTTPException: 운용중인 모델의 이름 외의 모델 요청이 오는경우. 
        HTTPException: 위의 예외 밖의 다른 에러가 발생하는 경우. 

    Returns:
        HealthResponse: 모델의 GPU 로드 여부. (load=True인 경우 HTTPException가 발생하지 않는 경우 True.)
    """

    if not model_name in ["t5", "polyglot"]:
        raise HTTPException(status_code=404, detail="'model_name' argument invalid! Type model name. \n Model: t5, polyglot")

    if load:
        try:
            get_summerization(model_name, {"prompt": "", "contents": "", "options": {}, "num_split": 1})

        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Exception: {err}")
        
        return HealthResponse(loaded=True)
    else:
        return HealthResponse(loaded=get_model_health(model_name))

