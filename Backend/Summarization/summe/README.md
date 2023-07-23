# Summe

T5 모델과 polyglot 모델을 이용하여 뉴스 기사를 요약해주는 RESTful API 입니다. 

## Requirements

- Python3.11.0
- Pip
- Poetry (Python Package Manager)

## Requirements
실행시 모델이 필요합니다. 모델이 필요하신 경우 [email](mailto:kwak@minseok.me)로 문의 주시기 바랍니다. 

## Installation
### ⚠️ 주의! 아직 `poetry`가 호횐되지 않습니다. 
pip install을 이용하여 다음의 패키지를 설치해 주세요. 
```
pip install uvicorn
pip install fastapi
pip install loguru
```
실행을 위해 다음의 명령어를 이용하여 실행시켜 주세요. 
```
PYTHONPATH=app/ uvicorn main:app --reload --host 0.0.0.0 --port PORT_NUMBER
```

## Project structure

    app
    |
    | # Fast-API stuff
    ├── api                 - web related stuff.
    │   └── routes          - web routes.
    ├── core                - application configuration, startup events, logging.
    ├── models              - pydantic models for this application.
    ├── services            - logic that is not just crud related.
    ├── main-aws-lambda.py  - [Optional] FastAPI application for AWS Lambda creation and configuration.
    └── main.py             - FastAPI application creation and configuration.
    |
    | # ML stuff
    ├── data             - where you persist data locally
    │   ├── interim      - intermediate data that has been transformed.
    │   ├── processed    - the final, canonical data sets for modeling.
    │   └── raw          - the original, immutable data dump.
    │
    ├── notebooks        - Jupyter notebooks. Naming convention is a number (for ordering),
    |
    ├── ml               - modelling source code for use in this project.
    │   ├── __init__.py  - makes ml a Python module
    │   ├── pipeline.py  - scripts to orchestrate the whole pipeline
    │   │
    │   ├── data         - scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features     - scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── model        - scripts to train models and make predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │
    └── tests            - pytest
