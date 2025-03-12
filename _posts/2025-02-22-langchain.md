---
title: LangChain 사용해보기
date: 2025-02-22
categories: [Today I Learn, 4th Week]
tags: [python, langchain]
math: true
---

## LangChain
- LangChain is a framework for developing applications powered by large language models (LLMs).
    ![alt text](/assets/images/langchain.png)

- LangChain 설치
    ```shell
    $pip install langchain
    ```

- Environment variables 설정<br/>
    Unix 계열
    ```shell
    export LANGSMITH_TRACING="true"
    export LANGSMITH_API_KEY="..."
    ```

    MS
    ```shell
    set LANGSMITH_TRACING="true"
    set LANGSMITH_API_KEY="..."
    ```

    Jupyter Notebook
    ```python
    import getpass
    import os

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
    ```

    다양한 방법이 가능하지만 우리는 `.env`를 사용한다.

- Using Language Model<br/>
    Chat Model 설치
    ```shell
    $ pip install -qU "langchain[openai]"
    ```

    모델 가져오기
    ```python
    from langchain.chat_models import init_chat_model

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    # langchain을 이용해 'gpt-4o-mini' 모델을 가져온다.
    ```

- `.env`를 이용해 환경설정 저장하기<br/>
    dotenv 설치
    ```shell
    $ pip install dotenv
    ```
    `.env`는 Hidden File로 숨김이 가능하다.
    ```shell
    OPENAI_API_KEY="api key 입력"
    ```

- API 사용하기
    ```python
    from dotenv import load_dotenv
    load_dotenv()

    from langchain.chat_models import init_chat_model
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    from langchain_core.prompts import ChatPromptTemplate

    system_template = "Translate the following from English into {language}"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    prompt = prompt_template.invoke({"language": "Korean", "text": "hi!"})
    response = model.invoke(prompt)
    print(response.content)
    ```
    출력
    ```
    안 녕 하 세 요 !
    ```

## API화 하기
- GET 요청에 부가 정보로 경로 파라미터 외에 쿼리 파라미터를 쓸 수 있다.<br/>
    쿼리 파라미터: 경로 뒤에 `?key1=value1&key2=value2`<br/>
    >  https://…../say?text=hi<br/>

- class로 바꿔서 사용하기<br/>
    `app_model.py`
    ```python
    from dotenv import load_dotenv

    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

    class AppModel:
    def __init__(self):
        load_dotenv() 
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
        system_template = "Translate the following from English into {language}"
        self.prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
        )

    def get_response(self, message):
        return self.model.invoke([HumanMessage(message)])

    def get_prompt_response(self, language, message):
        prompt = self.prompt_template.invoke({"language": language, "text": message})
        return self.model.invoke(prompt)

    def get_streaming_response(self, messages):
        return self.model.astream(messages)
    ```

- 배포하기<br/>
    `server.py`
    ```python
    from fastapi import FastAPI, Query
    from fastapi.responses import StreamingResponse
    from fastapi.staticfiles import StaticFiles

    import app_model

    app = FastAPI()

    model = app_model.AppModel()

    @app.get("/say")
    def say_app(text: str = Query()):
        response = model.get_response(text)
        return {"content" :response.content}

    @app.get("/traslate")
    def translater(language: str = Query(), text: str = Query()):
        response = model.get_prompt_response(language, text)
        return {"content" :response.content}
    ```

- 스트리밍<br/>
    SSE: Server-Side Event 웹 기술을 사용하여 이벤트 소스를 클라이언트에서 연결하고 서버는 이벤트 스트림으로 내려준다.

    현재의 구조: LangChain LLM + FastAPI 서버

## ChatBot
- ChatBot은 질문에 대한 적절한 답변을 받는 것이다. 하지만 ChatBot은 이전에 했던 말은 기억하지 못한다. 간단하게 구현해보면 아래와 같다.<br/>

    ```python
    from langchain_core.messages import HumanMessage

    model.invoke([HumanMessage(content="Hi! I'm Bob")])
    ```

    출력
    ```
    AIMessage(content='Hi Bob! How can I assist you today?'... 이하 생략
    ```
    <br/>
    ```python
    model.invoke([HumanMessage(content="What's my name?")])
    ```

    출력
    ```
    AIMessage(content="I'm sorry, but I don't have access to personal information about users unless it has been shared with me in the course of our conversation. How can I assist you today?" ... 이하 생략
    ```

- 따라서 대화를 위해서는 기본적으로 여태 했던 내용을 같이 넣어주어야 한다.
    ```python
    from langchain_core.messages import AIMessage

    model.invoke(
        [
            HumanMessage(content="Hi! I'm Bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
            HumanMessage(content="What's my name?"),
        ]
    )
    ```

    출력

    ```
    AIMessage(content='Your name is Bob! How can I help you today, Bob?' ... 이하 생략
    ```

- 대화내용을 기억하면서 소통하기위해선 메모리가 필요하며 LangGraph를 이용할 수 있다.
    ```python
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import START, MessagesState, StateGraph

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)


    # Define the function that calls the model
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}


    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    ```

## 오늘의 회고
- ChatBot의 기본 원리를 학습할 수 있었고 LLM을 이용한 해커톤을 할 때, 굉장히 유용한 학습이었다.