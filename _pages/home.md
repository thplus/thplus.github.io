---
layout: splash
title: Buyoung's Profile
permalink: /
hidden: true
header:
  overlay_color: "#fff"
  overlay_image: /assets/images/home_page.jpg
  # actions:
  #   - label: "<i class='fas fa-download'></i> Install now"
  #     url: "/docs/quick-start-guide/"
  #   - label: "<i class='fas fa-download'></i> test button"
  #     url: "/docs/quick-start-guide/"
excerpt: >
  <div style="display: flex; align-items: flex-start; gap: 30px; flex-wrap: wrap; color: white;">
    <div>
      <img src="/assets/images/buyoung_profile.png" alt="김부영 프로필" width="180" style="border-radius: 10px;" />
    </div>

    <div>
      <h3>AI 개발자 <strong>김부영</strong>입니다 😊</h3>
      <p>
        저는 실용적인 AI 기술로 사용자의 문제를 해결하는 데 집중하는 개발자입니다.<br/>
        <strong>AI Solution 설계, LLM 기반 Pipeline 설계, 이미지 처리, 모델 경량화, FastAPI 기반 MLOps 구축</strong>에 관심이 많습니다.<br/>
        작은 개선도 놓치지 않고 서비스에 녹일 수 있는 <strong>사용자 중심 AI</strong>를 지향합니다.<br/>
      </p>
      <p>
        📧 Email: <a href="mailto:glanz6670@naver.com">glanz6670@naver.com</a><br/>
        💻 GitHub: <a href="https://github.com/thplus" target="_blank">github.com/thplus</a><br/>
      </p>

      <h4>🧰 Tech Stack</h4>
      <ul style="columns: 2; font-size: 17px; padding-left: 1em;">
        <li><strong>AI 개발</strong>: PyTorch, TensorFlow, Huggingface 기반 모델 학습 및 Fine-tuning</li>
        <li><strong>서비스 구현</strong>: FastAPI를 활용한 AI 모델 서빙 및 API 설계</li>
        <li><strong>클라우드 및 협업</strong>: AWS S3 기반 데이터 관리, Google Colab 실험 환경, GitHub 버전 관리</li>
        <li><strong>데이터 분석</strong>: OpenCV, NumPy, Pandas, Scikit-learn을 통한 전처리 및 모델링</li>
        <li><strong>데이터베이스</strong>: MySQL을 활용한 데이터 저장 및 조회, JSON/Markdown 데이터 포맷 활용</li>
        <li><strong>개발환경</strong>: Jupyter Notebook, VSCode 중심 개발</li>
      </ul>
    </div>
  </div>


feature_row:

---

# Projects
## OnTheTop
<div class="project-card">
  <img src="/assets/images/OTT_AI_Architect_v1.drawio.png" alt="onthetop_main" class="project-thumb" />
  <div style="max-width: 1200px; margin: 0 auto;">
    <iframe width="1200" height="675" 
            src="https://www.youtube.com/embed/muK_FWTBECk" 
            title="AI Presentation" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen 
            style="width: 100%; height: auto; aspect-ratio: 16/9; display: block; margin: 0 auto;">
    </iframe>
  </div>
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2025.04.01. ~ 2025.08.01.
  </div>

  <div class="project-summary">
    해당 서비스는 Desk Setup 이미지를 기반으로 AI가 맞춤형 데스크테리어를 제안하는 서비스 입니다.<br/>
    사용자가 추가적인 자료 탐색 과정 없이도 자신만의 최적화된 작업 공간을 빠르고 간편하게 구축할 수 있으며, 추천된 제품은 바로 구매할 수 있도록 연결되어 실제 구매까지 이어질 수 있습니다.<br/>
    사진을 올리면 AI가 공간에 맞는 Desk Setup을 제안하고 곧바로 구매까지 이어지는 <strong>'번거로움 없이 나만의 공간을 완성할 수 있는 경험'</strong>이 해당 서비스가 제공하는 핵심 가치입니다.
  </div>
  
  <a class="btn btn--primary" href="https://youtu.be/6sqmKsrexIE">Demo Video</a>
  <a class="btn btn--primary" href="https://youtu.be/kv849T0BPTM">Ad Video</a>
  <a class="btn btn--danger" href="https://onthe-top.com">Services Link</a>
  <a class="btn btn--info" href="https://github.com/100-hours-a-week/16-Hot6-wiki/wiki/AI-Wiki">Gihub Wiki</a>
  <a class="btn btn--success" href="https://github.com/100-hours-a-week/16-Hot6-ai">AI Repo</a>

  <ul class="project-detail">
    <li><strong>주요 역할</strong></li>
      <ul>
        <li><strong>이미지 생성 Pipeline 설계:</strong> CNN 도입 및 제작, Inpainting 모델 설계, Masking 자동화 설계 및 구현, LoRA 설계 및 제작</li>
        <li><strong>아이템 추천 Pipeline 설계:</strong> OpenAI API를 활용한 Desk Setup 맞춤형 아이템 추천 설계, Naver API를 활용한 구매링크 연결</li>
        <li><strong>AI Model Serving:</strong> FastAPI 기반 AI 서버 제작, CNN 서버 분리, Redis Streams을 활용한 Multi-GPU 구현</li>
      </ul>
    <li><strong>기술 스택</strong></li>
      <ul>
        <li><strong>Vison:</strong> CNN, Gronding DINO, SAM2, Open CV</li>
        <li><strong>Generation:</strong> SDXL Inpainting, LoRA</li>
        <li><strong>Serving:</strong> FastAPI, PyTorch, Diffusers</li>
        <li><strong>Supporting Services:</strong> Redis Streams/Sentinel, AWS S3</li>
      </ul>
  </ul>
</div>

## Feature Map 기반 CNN 모델 최적화
<div class="project-card">
  <img src="/assets/images/cnnproject_final_layer4.png" alt="feature_map" class="project-thumb" />
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2025.03.19. ~ 2025.03.30.
  </div>

  <div class="project-summary">
    CNN 기반 이미지 분류 모델의 성능을 유지하면서 연산 비용과 메모리 사용량을 줄이기 위해 경량화에 대해 연구해 본 개인 프로젝트입니다.<br/>
    해당 프로젝트는 다른 연구들과는 달리 Feature Map 기반으로 모델이 어떤 특징들을 뽑아내는지 살펴보며 불필요한 Filter 및 Layer는 제거하거나 없애 시각적으로 경량화했습니다.
  </div>

  <a class="btn btn--primary" href="/project/ktb/cnn_project/">Details</a>
  <a class="btn btn--info" href="https://github.com/100-hours-a-week/2-brix-kim-personal-project?tab=readme-ov-file">MySQL</a>

  <ul class="project-detail">
    <li><strong>기술 스택:</strong> CNN, VGG16, GoogLeNet</li>
    <li><strong>주요 연구 내용: </strong></li>
      <ul>
        <li>모델 구성 및 비교:</li>
          <ul>
            <li>기본 CNN, VGG16, GoogLeNet 모델을 구현하고 성능 평가</li>
            <li>각 모델의 Feature Map을 시각화하여 불필요한 필터나 Layer 식별</li>
            <li>식별된 요소를 제거하여 경량화된 모델을 구성하고 성능 비교</li>
          </ul>
        <li>Feature Map 시각화:</li>
          <ul>
            <li>각 Convolution Layer를 틍과한 Feature Map을 시각화하여 정보 손실 여부 확인</li>
            <li>깊은 Layer에서 Feature Map 활성화가 희소해지고 의미있는 특징이 소실되는 현상을 관찰</li>
            <li>사람이 직접 확인했을 때도 특정 Filter 및 Layer에서는 Feature Map이 점 몇 개 수준으로만 남아 활용도가 낮을 수 있음을 확인</li>
          </ul>
        <li>설명 가능한 경량화</li>
          <ul>
            <li>기존의 무작위 기반 pruning과 달리, Feature Map 시각화를 통해 제거 대상 Filter 및 Layer를 선정</li>
            <li>직관적으로 설명 가능한 근거에 기반한 pruning 수행</li>
            <li>GoogLeNet 구조에서 해당 현상을 재현하고, 불필요한 Filter 및 Layer 제거를 통해 효율성 향상</li>
          </ul>
      </ul>
        <li><strong>성능 평가 및 결과:</strong></li>
      <ul>
        <li>연구 내용을 바탕으로 Rice Image Dataset에 적합한 모델 설계 및 제작</li>
        <li>모델별 성능 비교:</li>
          <ul>
            <img src="/assets/images/cnnproject_final_summary.png" alt="model_summary" />
            <img src="/assets/images/cnnproject_final_filter.png" alt="model_filter" />
            <img src="/assets/images/cnnproject_final_flops.png" alt="model_flops" />
            <li>기본 모델과 경량화된 모델의 정확도, 필터의 개수, 연산량을 비교</li>
            <li>모델의 정확도와 필터, 연산량 등은 상관관계가 없고 경량화한 모델의 일반화 성능이 더욱 뛰어나며, 실제 응용 환경에서도 Feature Map을 통한 경량화는 의미가 있음</li>
          </ul>
      </ul>
  </ul>
</div>

## HarmonAI
<div class="project-card">
  <video controls style="width: 1200px; max-width: 100%; height: auto; display: block; margin: 0 auto;">
    <source src="/assets/videos/harmonai_demo.mp4" type="video/mp4" />
  </video>
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2025.02.26. ~ 2025.02.28.
  </div>

  <div class="project-summary">
    사용자의 위치(위도, 경도), 날씨, 감성의 정도와 함께 사용자의 기분을 입력받아 본인에게 맞는 음악을 추천해주는 서비스로 카카오테크부트캠프 해커톤에서 진행한 팀 프로젝트입니다.<br/>
    해당 서비스에서 원하는 사용자 경험은 "이 기분, 장소, 날씨 그리고 감성"에 현재의 음악이 잘 어울린다는 느낌을 주는 것입니다.<br/>
  </div>

  <a class="btn btn--primary" href="/project/ktb/hackathon/">Details</a>
  <a class="btn btn--info" href="https://github.com/KTB-Hackerton-24Team/HarmonAI_AI">AI Github</a>

  <ul class="project-detail">
    <li><strong>기술 스택:</strong> FastAPI</li>
    <li>
      <strong>주요 기능 및 구현 내용</strong>
      <ul>
        <li>AI 백엔드 설계: Python, FastAPI</li>
        <li>AI 파이프라인 설계: Backend(위도, 경도) → Google Maps API(위치 정보) → 기상청 API(날씨 정보) → OpenAI API(노래 추천) → Spotify API(할루시네이션 검증 및 노래 추천 활용) → Backend(추천 노래 정보 전달)</li>
        <li>ChatGPT 프롬프트 엔지니어링: 위치, 날씨, 기분 등을 고려한 노래를 추천해주기 위한 프롬프트 작성</li>
      </ul>
    </li>
  </ul>
</div>

## TPS Project
<div class="project-card">
  <video controls style="width: 1200px; max-width: 100%; height: auto; display: block; margin: 0 auto;">
    <source src="/assets/videos/TPS_Project_Play.mp4" type="video/mp4" />
  </video>
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2024.12.03. ~ 2024.12.13.
  </div>

  <div class="project-summary">
    TPS 게임 개발을 통해 객체지향 프로그래밍의 핵심 개념인 캡슐화, 상속, 다형성을 실습하고 Singleton 패턴, 벡터 계산 등을 구현해 보면서 객체지향 설계의 원리를 체득한 개인 프로젝트 입니다.<br/>
  </div>

  <a class="btn btn--success" href="https://drive.google.com/file/d/13pQfE7MHxG-DHtxnUWtvNgebXMDKbohJ/view?usp=sharing">Source Code</a>

  <ul class="project-detail">
    <li><strong>기술 스택:</strong> C#, Unity</li>
    <li><strong>주요 기능 및 구현 내용:</strong>
      <ul>
        <li>캐릭터 컨트롤러: 3인칭 시점에서의 캐릭터 이동 및 조작 구현</li>
        <li>시네마틱 카메라 시스템: 게임 플레이에 몰입감을 더하는 카메라 연출 구현</li>
        <li>적 AI: 적 캐릭터의 행동 패턴 및 반응 구현</li>
        <li>무기 시스템: 다양한 무기의 사용 및 전환 기능 구현</li>
        <li>Singleton 패턴: 게임의 전체 상태 관리를 위한 Singleton 패턴 적용</li>
        <li>벡터 계산: 무기 시스템에서의 정확한 위치 및 방향 계산을 위한 벡터 수학 활용</li>
      </ul>
    </li>
  </ul>
</div>

## 감성분석을 활용한 언론사의 양극화 분석
<div class="project-card">
  <img src="/assets/images/sentiment_lda_ronegative.png" alt="sentiment_5" class="project-thumb" />
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2024.09.11. ~ 2024.12.04.
  </div>

  <div class="project-summary">
    미국 대선 기간동안 주요 언론사의 헤드라인을 분석하여 정치적 양극화 현상을 감성분석을 통해 정량적으로 평가해 본 팀 프로젝트입니다.<br/>
    언론의 정치적 편향성 증가로 인해 독자들은 자신과 일치하는 정보만을 소비하는 경향이 강화되고 있으며 이는 사회적 갈등을 심화시킬 수 있습니다. 따라서 해당 프로젝트에서는 언론사의 헤드라인을 감성 분석하여 정치적 성향에 따른 보도 경향을 정량적으로 분석하고자 하였습니다.<br/>
  </div>

  <a class="btn btn--primary" href="/project/ajou%20university/sentiment_analysis_with_headline/">Details</a>

  <ul class="project-detail">
    <li><strong>주요 역할:</strong> 여러 감성 분석 기법(RoBERTa, VADER, TextBlob, Topic 모델링)을 적용하여 결과를 비교 분석 후 인사이트 도출, 언론사의 정치적 성향에 따른 감성 분석 결과를 시각화하고 해석 </li>
    <li><strong>기술 스택:</strong> RoBERTa, VADER, TextBlob, LDA Topic Modeling, NumPy, Pandas, Matplotlib</li>
    <li><strong>분석 결과:</strong>
      <ul>
        <li>진보 성향 언론사는 Biden 관련 헤드라인에서 긍정적인 감성이 높게 나타남</li>
        <li>보수 성향 언론사는 Trump 관련 헤드라인에서 긍정적인 감성이 높게 나타남</li>
        <li>부정적인 감성의 경우 극단적인 성향의 언론사일수록 키워드와 무관하게 높게 나타남</li>
        <li>TextBlob 분석결과 부정적인 감성과 주관성은 연관이 있음을 알 수 있음</li>
        <li>편향성이 두드러질수록 부정적인 감성과 주관성이 두드러짐</li>
      </ul>
    </li>
    <li><strong>성과:</strong>
      <ul>
        <li>정성적인 부분을 정량적으로 평가함</li>
        <li>편향은 정치뿐만 아니라 제품군에서도 두드러지므로 부정적 리뷰를 모아 분석하면 어떤 부분에서 경쟁사 대비 선호도가 낮은지 수치적으로 분석할 수 있음</li>
      </ul>
    </li>
  </ul>
</div>

## Honeybee Disease Diagnosis
<div class="project-card">
  <video controls style="width: 1200px; max-width: 100%; height: auto; display: block; margin: 0 auto;">
    <source src="/assets/videos/honeybee_video1.mp4" type="video/mp4" />
  </video>
  <div class="project-date">
    <i class="fa fa-calendar"></i> 2024.05.30. ~ 2024.06.06.
  </div>

  <div class="project-summary">
    꿀벌의 주요 질병(백묵병), 기생충(바로아), 천적(말벌)을 실시간으로 탐지하는 AI 모델 개발을 한 팀 프로젝트입니다.<br/>
    꿀벌은 농작물의 수분을 담당하는 필수적인 화분매개자로 국내 농업 경제에 약 6조 원의 가치를 창출하고 있습니다. 그러나 최근 꿀벌의 집단 폐사가 심각한 문제로 대두되고 있으며 그 주요 원인 중 하나로 각종 질병이 지목되고 있습니다. 따라서 우리는 이러한 질병을 조기에 감지하고 대응하기 위한 시스템을 개발하고자 하였습니다.<br/>
  </div>

  <a class="btn btn--primary" href="/project/ajou%20university/honeybee_diease_diagnosis/">Details</a>

  <ul class="project-detail">
    <li><strong>주요 역할:</strong> YOLOv8 탐지 모델 설계 및 학습 파이프라인 구축</li>
    <li><strong>기술 스택:</strong> YOLOv8, Tensorflow</li>
    <li><strong>성능 평가:</strong> Confusion Matrix 및 PR Curve를 통해 성능 평가를 하였으며 백묵병 탐지 정확도 및 실시간 탐지 성능이 우수</li>
    <li><strong>모델 개발 및 개선 과정:</strong>
      <ul>
        <li>초기 모델은 CNN을 사용 하였으나 높은 정확도 (99%)에도 불구하고 환경에 민감하여 일반화가 어려움</li>
        <li>또한 백묵병과 바로아, 말벌 등 꿀벌에게 위협이 되는 것들은 짧은 시간 내에 막대한 피해를 입히기 때문에 질병을 잘 감지하는 것만이 좋은 방향은 아님</li>
        <li>YOLO의 경우 이미지 전체를 한 번에 처리하여 객체를 감지하며 실시간 탐지가 가능, 다양한 환경에서도 높은 정확도 유지</li>
        <li>기존 백묵병 이외에도 바로아 기생충, 말벌 등을 추가 탐지</li>
      </ul>
    </li>
  </ul>
</div>


<!-- {% include feature_row %} -->