# Crawling Server
## 개요
- 정해진 종목 코드에 대해 네이버 뉴스 기사를 가져오는 RestAPI Server 코드 입니다. 

## 실행 방법
### ‼️ 아래의 필수 요소를 설치해주세요. ‼️
```
pip install fastapi
pip install "uvicorn[standard]"
```

### 실행 방법은 다음과 같습니다. 
```
uvicorn main:app --reload
```

### 다음 URL에서 확인 가능합니다. 
```
http://127.0.0.1:8000/docs
```