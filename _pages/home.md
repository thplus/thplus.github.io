---
layout: splash
title: Buyoung's Profile
permalink: /
hidden: true
header:
  overlay_color: "#5e616c"
  overlay_image: /assets/images/mm-home-page-feature.jpg
  # actions:
  #   - label: "<i class='fas fa-download'></i> Install now"
  #     url: "/docs/quick-start-guide/"
  #   - label: "<i class='fas fa-download'></i> test button"
  #     url: "/docs/quick-start-guide/"
excerpt: >
  {% include excerpt-profile.html %}

feature_row:
  - image_path: /assets/images/onthetop_main.png
    alt: "Onthetop-main"
    title: "OnTheTop"
    excerpt: >
      <span style="font-size: 16px;">
        <i class="fa fa-calendar" aria-hidden="true"></i> 2025.03.31. ~ 2025.08.01.
      </span><br />
      "Desk 사진을 올리면 AI가 Desk Setup 사진을 추천해주는 서비스입니다."<br/>
      <br/>
      <a class="btn btn--success" href="https://onthe-top.com" target="_blank">Service Page</a>
      <a class="btn btn--primary" href="https://github.com/100-hours-a-week/16-Hot6-wiki/wiki">Wiki</a>
      <a class="btn btn--info" href="https://github.com/100-hours-a-week/16-Hot6-ai" target="_blank">GitHub</a>
    
  - image_path: /assets/images/rice_teaser.png
    alt: "Rice Image Dataset"
    title: "Feature Map 기반 CNN 모델 최적화: Rice Image Dataset 사례 연구"
    excerpt: >
      <span style="font-size: 16px;">
        <i class="fa fa-calendar" aria-hidden="true"></i> 2025.03.19. ~ 2025.03.30.
      </span><br />
      "Rice Image Dataset을 이용하여 Feature Map을 토대로 CNN 계열 모델을 시각적으로 경량화 해 본 개인 프로젝트입니다."<br/>
      <br/>
      <a class="btn btn--primary" href="/project/ktb/cnn_project/">Detalis</a>
    
  - image_path: /assets/images/hackathon_4.png
    alt: "Hackathon"
    title: "노래 추천 AI 서비스"
    excerpt: >
      <span style="font-size: 16px;">
        <i class="fa fa-calendar" aria-hidden="true"></i> 2025.02.26. ~ 2025.02.28.
      </span><br />
      "LLM을 활용한 노래 추천 서비스로 배포까지 완료해 본 프로젝트입니다. 프롬프트 엔지니어링과 API 설계, FastAPI 설계, 디버깅 등을 담당했습니다."<br/>
      <br/>
      <a class="btn btn--primary" href="/project/ktb/hackathon/">Detalis</a>

  - image_path: /assets/images/tps_teaser.png
    alt: "TPS Teaser"
    title: "TPS Project"
    excerpt: >
      <span style="font-size: 16px;">
        <i class="fa fa-calendar" aria-hidden="true"></i> 2024.12.03. ~ 2024.12.13.
      </span><br />
      "Unity와 C#을 통해 TPS 게임을 만들어 본 개인 프로젝트입니다."<br/>
      <br/>
      <a class="btn btn--primary" href="/project/ajou%20university/tps_project/">Detalis</a>

  - image_path: /assets/images/sentiment_1.png
    alt: "Sentiment"
    title: "감성분석을 활용한 언론사의 양극화 분석"
    excerpt: >
     <span style="font-size: 16px;">
        <i class="fa fa-calendar" aria-hidden="true"></i> 2024.09.11. ~ 2024.12.04.
     </span><br />
     "감성분석을 활용하여 언론사의 양극화 현상을 분석해 본 프로젝트입니다. 감성분석 모델 설계, Topic 모델 설계, 결과분석, 인사이트 분석을 담당했습니다."<br/>
     <br/>
     <a class="btn btn--primary" href="/project/ajou%20university/sentiment_analysis_with_headline/">Detalis</a>

  - image_path: /assets/images/honeybee_yolo_varroa2.png
    alt: "honeybee_varroa"
    title: "Honeybee Diease Diagnosis"
    excerpt: >
      <span style="font-size: 16px;">
        <i class="fa fa-calendar" aria-hidden="true"></i> 2024.05.30. ~ 2024.06.06.
      </span><br />
      "꿀벌의 질병을 실시간으로 탐지해주는 AI 모델을 만들어 본 프로젝트 입니다. 본 프로젝트는 YOLOv8을 이용하여 실시간으로 바로아 기생충, 백묵병, 말벌을 탐지하였습니다."<br/>
      <div style="padding-left: 1em;">
        <li><b>주도한 역할</b>: YOLOv8 기반 객체 탐지 모델 설계 및 학습 파이프라인 구축</li>
        <li><b>기술 스택</b>: YOLOv8, Roboflow, Python, OpenCV</li>
        <li><b>문제 해결</b>: CNN 모델의 낮은 정확도 및 과탐지 문제를 보완하기 위해 YOLO 전환 제안 및 실험 수행</li>
        <li><b>성과</b>: CNN에 비해 Accuracy 낮은 점을 극복하기 위해 YOLOv8 오탐(FN) 감소 달성</li>
      </div>
      <br/>
      <a class="btn btn--primary" href="/project/ajou%20university/honeybee_diease_diagnosis/">Detalis</a>

---

# Projects
<div class="project-card">
  <img src="/assets/images/sentiment_lda_topic_5.png" alt="sentiment_5" class="project-thumb" />
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2024.09.11. ~ 2024.12.04.
  </div>

  <div class="project-summary">
    <strong>Honeybee Disease Diagnosis</strong><br/>
    감성분석을 활용하여 언론사의 양극화 현상을 분석해 본 팀 프로젝트입니다. Transformer 계열과 Lexicon based, Topic 모델링을 이용하여 Topic별 각 언론사의 Headline에 대한 감성분석을 진행하였습니다.
  </div>

  <ul class="project-detail">
    <li><strong>주요 역할:</strong> RoBERTa, VADER, TextBlob, Topic 모델링, 인사이트 분석, 모델링 결과 분석</li>
    <li><strong>사용 기술:</strong> RoBERTa, VADER, TextBlob, LDA Topic, NumPy, Pandas, Matplotlib</li>
    <li><strong>문제 해결:</strong> CNN 정확도 문제 해결을 위한 YOLO 도입</li>
    <li><strong>성과:</strong> 실시간성 확보 및 탐지 정확도 향상</li>
  </ul>

  <a class="btn btn--primary" href="/project/ajou%20university/honeybee_diease_diagnosis/">Details</a>
</div>

<div class="project-card">
  <img src="/assets/images/honeybee_yolo_varroa2.png" alt="honeybee_varroa" class="project-thumb" />
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2024.05.30. ~ 2024.06.06.
  </div>

  <div class="project-summary">
    <strong>Honeybee Disease Diagnosis</strong><br/>
    YOLOv8을 활용해 꿀벌 질병(바로아, 백묵병, 말벌)을 실시간으로 탐지한 팀 프로젝트입니다.
  </div>

  <ul class="project-detail">
    <li><strong>주요 역할:</strong> YOLOv8 탐지 모델 설계 및 학습</li>
    <li><strong>사용 기술:</strong> YOLOv8, Tensorflow</li>
    <li><strong>주요 트러블:</strong> CNN은 정확도는 높으나 서비스 목적에 맞지 않음</li>
    <li><strong>성과:</strong> YOLOv8로 추가 위협 바로아 기생충, 말벌 탐지 가능 및 실시간 검출로</li>
  </ul>

  <a class="btn btn--primary" href="/project/ajou%20university/honeybee_diease_diagnosis/">Details</a>
</div>


<!-- {% include feature_row %} -->


## KaKao Tech Bootcamp - Today I Learn

| Week     | Date     | Topic                                                        |
| -------- | ------   | ------------------------------------------------------------ |
| 1st Week - [Github](/categories/#1st-week) | 25.01.30.| [Github Page](https://github.com/100-hours-a-week/2-brix-kim-til)|
| 2nd Week - [데이터 분석](/categories/#2nd-week)| 25.02.03.| [1주차 내용 복습](/today%20i%20learn/2nd%20week/first_week_review/)|
|  | 25.02.04. | [NumPy & Pandas](/today%20i%20learn/2nd%20week/numpy_pandas/)|
|  | 25.02.05. | [NumPy 심화학습](/today%20i%20learn/2nd%20week/numpy_advanced/) |
|  | 25.02.06. | [Pandas 심화학습](/today%20i%20learn/2nd%20week/pandas_advanced/) |
|  | 25.02.07. | [NumPy & Pandas 딥 다이브](/today%20i%20learn/2nd%20week/pandas_numpy_deepdive/) |
| 3rd Week - [데이터 시각화](/categories/#3rd-week)| 25.02.10.| [Data 시각화](/today%20i%20learn/3rd%20week/data_visualization/)|
|  | 25.02.11. | [Krampoline IDE 사용해보기](/today%20i%20learn/3rd%20week/krampoline/) |
|  | 25.02.12. | [확률분포](/today%20i%20learn/3rd%20week/probability_distribution/) |
|  | 25.02.13. | [가설검정](/today%20i%20learn/3rd%20week/hypothesis_test/) |
|  | 25.02.14. | [데이터 시각화 딥 다이브](/today%20i%20learn/3rd%20week/data_visualization_deepdive/) |
|  | 25.02.15. | [AI 모델링](/today%20i%20learn/3rd%20week/ai_modeling/) |
| 4th Week - [인공지능](/categories/#4th-week)| 25.02.17.| [다변수함수의 미분](/today%20i%20learn/4th%20week/differential/)|
|  | 25.02.18. | [인공지능 기본 정리](/today%20i%20learn/4th%20week/AI_summary/) |
|  | 25.02.20. | [FastAPI 사용해보기](/today%20i%20learn/4th%20week/fastapi/) |
|  | 25.02.22. | [LangChain 사용해보기](/today%20i%20learn/4th%20week/langchain/) |
| 5th Week - [딥러닝 및 해커톤](/categories/#5th-week)| 25.02.24.| [CNN](/today%20i%20learn/5th%20week/cnn/)|
|  | 25.02.28. | [Hackathon](/project/ktb/hackathon/) |
| 6th Week - [모델](/categories/#6th-week)| 25.03.04.| [Pre-Trained Model](/today%20i%20learn/6th%20week/pretrainedmodel/)|
|  | 25.03.05. | [Pre-Trained Model](/today%20i%20learn/6th%20week/addoverfitting/) |
| 7th Week - [NLP](/categories/#7th-week)| 25.03.10.| [Natural Language Processing](/today%20i%20learn/7th%20week/nlp/)|
|  | 25.03.11. | [Application Model](/today%20i%20learn/7th%20week/application_model/) |