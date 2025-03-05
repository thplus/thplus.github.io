---
title: Krampoline IDE 사용해보기
date: 2025-02-11
categories: [Today I Learn, 3rd Week]
tags: [Krampoline]     # TAG names should always be lowercase
math: true
---

## Krampoline IDE
- Krampoline IDE는 컨테이너 기반의 환경에서 동작하고 있는 kakao사 제품이다. 컨테이너는 우분투 운영체제에서 다양한 소프트웨어 스택으로 구성되어 있으며 컨테이너 콘솔은 이러한 컨테이너들을 생성하고 관리할 수 있는 제품으로 다양한 기능을 제공한다.

- 컨테이너는 어떤 환경에서나 실행 가능한 소프트웨어 패키지로 소프트웨어 실행에 필요한 모든 요소를 포함하고 있다. Krampoline IDE에서는 우분투 운영체제를 가상화한 컨테이너를 통해 파이썬과 자바스크립 등 다양한 소프트웨어 스택을 제공하고 있다. Krampoline IDE를 이용하면 사용자는 어디서든지 웹 브라우저를 통해 동일한 개발 환경을 실행할 수 있다.

- KaKao Cloud를 활용한 배포
    - Krampoline IDE와 카카오 D2Hub, Kargo 시스템의 연동을 통해 애플리케이션의 이미지를 간편하게 빌드하고 배포할 수 있다. Krampoline IDE를 통해 카카오 클라우드 자원을 활용하여 서비스를 배포하는 플로우는 아래와 같다.<br/>
    ![alt text](/assets/images/krampolineide.png)

## 오늘의 회고
- Krampoline IDE를 활용해 배포 방법에 대해 배웠다. AWS 말고도 다른 배포 방법이 많지만 국내 서비스를 이용하는 것은 처음 접해봤다. 좀 더 나중에 활용해보고 실제 배포를 해봐야할 듯 싶다.