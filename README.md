# 📄 프로젝트 소개

하루에도 수백, 수천 개의 경제 뉴스가 발간되어 기업의 경제 활동에 대한 정보를 제공합니다. 하지만 양이 지나치게 방대하여 개인이 전부 읽은 뒤 주식 투자에 활용 하기에는 어려움이 있습니다. 따라서 저희는 기사 요약, 키워드 추출, 키워드와 기사에 대한 긍/부정 분류를 통해 주식시장의 흐름을 가독성이 높은 형태로 제공하려고 합니다.

# 🗓️ 개발 기간

23.07.03 - 23.07.28(총 26일)

# 👨‍👨‍👧‍👧 멤버 구성 및 역할

| [곽민석](https://github.com/kms7530)                                                       | [이인균](https://github.com/lig96)                                                                                 | [임하림](https://github.com/halimx2)                                                                                                | [최휘민](https://github.com/ChoiHwimin)                  | [황윤기](https://github.com/dbsrlskfdk) |
|---|---|---|---| --- |
| <img src="https://avatars.githubusercontent.com/u/6489395" width="140px" height="140px" title="Minseok Kwak" /> | <img src="https://avatars.githubusercontent.com/u/126560547" width="140px" height="140px" title="Ingyun Lee" /> | <img src="https://ca.slack-edge.com/T03KVA8PQDC-U04RK3E8L3D-ebbce77c3928-512" width="140px" height="140px" title="ChoiHwimin" /> | <img src="https://avatars.githubusercontent.com/u/102031218?v=4" width="140px" height="140px" title="이름" /> | <img src="https://avatars.githubusercontent.com/u/4418651?v=4" width="140px" height="140px" title="yungi" /> |

- **곽민석**
    - 요약 모델
        - 데이터셋 수집
        - 모델 성능 실험
        - 요약 모델 API 설계
    - 기사 긍부정 분류 데이터셋
        - LLM을 이용한 데이터셋 라벨링
    - Backend
    - Frontend
- **이인균**
    - 뉴스 긍부정 분류
        - 최신 Model 탐색
        - 자체 Model 설계
- **임하림**
    - 서기
    - 뉴스 긍부정 분류
        - 자체 Model 설계
        - 모델 입력 데이터 설계
        - 뉴스 긍부정 api 설계
        - chat gpt 라벨링
    - 기사 전처리
- **최휘민**
    - 키워드 추출
        - 자체 Model 설계 및 실험
        - 키워드 추출 API 설계
        - 평가 데이터 수집
- **황윤기**
    - 프로젝트 리더
    - 네이버 뉴스 크롤링
    - Airflow
        - Crawling Scheduling
        - Serving Scheduling
    - 키워드 추출
        - KeyBERT 기반 Model

# ⚒️ 기능

## 키워드 추출

- 주어진 기사 내에서 중요한 키워드를 추출하기 위한 작업을 시행합니다.
- 기사 전체에서 단어의 가중치를 계산하고, 해당 가중치를 이용한 주요 단어 후보를 선정합니다.
- 한국어 키워드 추출의 성능을 측정하기 위한 데이터셋이 존재하지 않기 때문에 50개의 자체 평가 데이터셋을 구성하였습니다.

## 기사 감성 분석

- 추출된 키워드가 기업의 좋은 상황을 나타내는 단어인지, 나쁜 상황을 나타내는 단어인지 정보를 제공하기 위해 감성분석 모델을 사용합니다. 기사 전체를 감성 분석한 뒤에 키워드의 대용 지표로 채택합니다.
- 학습 데이터로 30개의 기업 기업의 기사 4개 기업(3800개)+26개 기업(520개)의 기사를 chat gpt API를 활용해 긍부정 labeling을 진행했고,  train set, dev set을 8:2로 나누어서 학습을 진행했습니다.

## 기사 요약

- IT / 경제분야 뉴스를 이용하여 학습한 모델(`T5`, `polyglot-ko`)을 이용하여 뉴스를 요약 제공합니다.
- `T5` 모델을 이용하여 한줄 요약을 만들어내고, 이후 자세한 내용은 `polyglot-ko` 모델을 이용하여 상대적으로 긴 요약 내용을 추가해 줍니다.
- 모델을 이용하여 생성된 결과는 유의미한 문장만을 가져와 후처리하여 반환합니다.

# 🏗️ 프로젝트 구조
![](/assets/img/proj_structure.png)

# 👨‍🔬 사용 모델

## 키워드 추출

- `TF-IDF` Vectorizer로 문서를 벡터로 표현하여, 상위 가중치를 갖는 단어를 주요 단어 후보로 선정했습니다.
- 주요 단어 후보와 기사를 한국어로 기학습된 `Sentence-Transformer`를 이용해서 Embedding을 계산한 후, 유사도를 계산하여 높은 점수를 낸 단어를 해당 기사의 주요 키워드로 선정하였습니다.
- 선정된 키워드들을 키워드의 형태(명사형 어구)로 표시하고자, 추출된 키워드에 대해 후처리를 진행하였습니다.

## 뉴스 긍부정 분류

- 최대 1900 토큰의 길이에 달하는 기사들을 첫 128 토큰과 마지막 384토큰만을 사용하여, 총 512토큰으로 요약하여 모델에 입력하였습니다.
- 모델은 `klue/RoBERTa-large`를 사용하였습니다.

## 기사 요약

- 논문 요약 데이터셋과 IT / 경제분야 뉴스 요약 데이터셋을 실험한 결과 선정된 IT / 경제분야 뉴스 요약 데이터셋을 선정하여 요약 모델을 학습시켰습니다.
- 선정된 데이터셋을 이용하여 본 서비스에서 사용될 `T5-base`, `polyglot-ko 1.3b` 모델을 학습시켰습니다.

# 데모 영상

![](/assets/img/demo.gif)

# 🔗 링크

- [랩업 리포트](/assets/docs/NLP_04_Wrap-Up_Report_FinalProj.pdf)
- [프로젝트 소개 노션 페이지](https://www.notion.so/0375bff1dc834ead9f6a8f9ae8baa90e?pvs=21)
- [최종 발표 영상](https://youtu.be/KgvPInfO6k8)
- [서비스 바로가기](https://deagul.netlify.app/)
