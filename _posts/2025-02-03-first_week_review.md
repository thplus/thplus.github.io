---
title: 1주차 내용 복습
date: 2025-02-03
categories: [Today I Learn, 1st Week]
tags: [review, python]     # TAG names should always be lowercase
math: true
---

## 재귀함수
- Subroutine call: 프로그램 내에서 반복적으로 수행되는 코드를 별도로 작성하여 필요할 때, 호출하는 것.<br/>
    Subroutine은 재사용 가능성, 모듈화, 디버깅 등 코드 작업을 할 때, 다양한 이점이 있다.

- Tail call: 어떤 subroutine이나 함수의 return이 함수인 경우이다.
    ```csharp
    return Function(x); // Tail call
    ```
    ```csharp
    return Function(x) + a; // Tail call이 아님
    ```
    위의 예시와 같이 fucntion 자체만 불러야하며 다른 추가적인 연산을 하면 tail call이 아니다. factorial 함수는 아래와 같이 작성되는데 이 또한 tail call이 아니다.
    ```csharp
    return n * factorial(n-1);
    ```

- 재귀함수는 일반적으로 메모리를 더 많이 사용한다. Stack메모리를 사용하기 때문이다.
  ```csharp
  static int Factorial(int n) {
    if (n == 1) return 1;
    return n * Factorial(n-1);
  }
  ```
  위와 같이 Factorial 함수를 정의하면 함수 호출 과정은 아래와 같다.
  ```csharp
  Factorial(5)
  -> Factorial(4)
    -> Factorial(3)
        -> Factorial(2)
            -> Factorial(1)
  ```
  각 함수가 호출 될때마다 스택이 생성되어 메모리를 차지하게 되고 마지막 Factorial(1)이 끝나면 스택이 역순으로 해제되어 메모리 사용량이 많다고 할 수 있다.

## Class의 이름의 첫 글자는 대문자로
- 코드 컨벤션 or 네이밍 컨벤션이라고도 하며 일반적인 프로그래밍 언어에서는 class 이름을 대문자로 시작하는 것을 권장하며 이를 **PascalCase**라고 한다.<br/>
매서드나 변수는 일반적으로 소문자로 시작하는 **camelCase**를 사용한다.

- 이런 컨벤션, 규칙이 있는 이유는 코드의 가독성과 구분성을 향상시키기 위함이다. 따라서 1주차 과제에서 Class 네이밍을 수정하도록 하자.

## CLI내용 Clear하기
- 기존에 알고 있는 ```os.system('cls')```는 Windows에서 작동하는 명령어다 따라서 그전에 OS type을 먼저 확인할 필요가 있다.<br/>
  os를 확인은 아래와 같이 확인할 수 있다.
  ```python
  import platform

  ostype = platform.system()
  ```
  ostype의 return 값은 아래와 같다.
  - Windows : "Windows"
  - macOS : "Darwin"
  - Linux : "Linux"<br/>
  
- 따라서 함수를 다시 작성하면 아래와 같은 예시로 작성할 수 있다.
  ```python
  import platform

  def clear():
    ostype = platform.system()

    if ostype == "Linux" or ostype == "Darwin":
      os.system('clear')
    elif ostype == "Windws"
      os.system('cls')
  ```

## 오늘의 회고
- 재귀함수는 일반적으로 메모리를 많이 사용한다고 알고 있었는데, 그 구조와 이유에 대해서 제대로 학습할 수 있는 기회가 되었다.
- 재귀함수를 공부하면서 subroutine call과 tail call에 대해서 배웠으며 그 예시를 factorial 함수를 이용해 개념을 학습할 수 있었다.
- 재귀함수보다는 반복문을 사용하는 것이 더 좋을 것 같다.
- 명명에 대한 규칙을 크게 신경써보지 않았는데, Alex의 피드백으로 코드 컨벤션을 다시 한 번 공부하고 그 중요성을 배울 수 있는 시간이었다.
- 1주차 과제를 ```VSCode```로 다시 작성해보고 GitHub에 올려보며 git에 좀 더 익숙해지는 시간이었다.
- 1주차 과제 수정 중 Victoria의 도움으로 os.system()을 이용해 CLI를 clear하는 것을 다시 익혔다.