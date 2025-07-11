### Ajou University Deep Learning
# Team 7 Crack Vision  
CNN 기반 차량 파손 탐지
- 팀원: 이승준, 최용준, 최훈서
---

## 개요  
Team 7 Crack Vision 은 CNN(ResNet-50) 분류 모델과 YOLOv8 객체 탐지 모델을 활용하여 차량 파손 탐지 

- 차량 부위(Part) 분류  
- 파손 형태(Damage) 분류  
- 파손 부위별 바운딩 박스 탐지  

하나의 파이프라인에서 제공하는 통합 차량 파손 진단 시스템입니다.

---

##  주요 기능  
- **차량 부위 분류**: ResNet-50 기반 멀티헤드 분류기로 “부위 / 손상” 동시 예측  
- **차량 파손 탐지**: YOLOv8 기반 객체 탐지기로 작은 손상 영역까지 바운딩 박스로 시각화  
- **2×2 비교 시각화**: 원본 vs. 분류 결과, 원본 vs. 탐지 결과를 한눈에 비교  
- **Flask 웹 인터페이스**: 브라우저에서 이미지 크롭 → 분석 → 결과 확인  

---

## 데모 영상  
- 아래 영상에서 실제 시스템의 **크롭 → 업로드 → 추론 → 결과 시각화** 전체 흐름을 확인하세요.

- [▶️ 데모 영상 보기](cars_analysis/videos/demo.mp4)

---


## 사용법
requirements.txt 를 참고하여 라이브러리를 설치한다.
1. pip install -r requirements.txt 명령어를 통해 라이브러리 간편 설치 
2. cropperjs 라이브러리만 npm install cropperjs 하여 설치한다.
3. flask run --host=0.0.0.0 --port=5000 flask 서버 실행 
---

## 코드 구조
```
cars_analysis/
├─ app.py               # Flask 웹 서버
├─ inference.py         # ResNet50 & YOLOv8 통합 시각화
├─ requirements.txt     # Python 패키지 의존성
├─ templates/
│   ├─ index.html       # 이미지 크롭 & 업로드 UI
│   └─ result.html      # 분석 결과 페이지
├─ static/
│   ├─ uploads/         # 업로드된 이미지
│   └─ results/         # 결과 이미지
└─ videos/
    └─ demo.mp4
```