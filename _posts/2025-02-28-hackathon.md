---
title: 카카오테크부트캠프 해커톤 review
date: 2025-02-28
categories: [Project, KTB]
tags: [python, langchain, fastapi]
math: true
---

## 해커톤 소개
- 이번 해커톤 기간은 25.02.26(수) ~ 25.02.28(금) 3일간 진행된다.<br/>
    하지만 28일 10시에 예선발표가 있으므로 실제 개발할 수 있는 기간은 2일 남짓이다.

- 기본적으로 풀스택 2명, 클라우드 2명, AI 2명이 기본 팀 구성이지만 우리 팀은 풀스택 인원이 구성되지 않은채 클라우드 2명, AI 3명이서 팀이 구성되었다.

- 주제는 LLM을 이용한 서비스

- 어떤 서비스가 되었던 LLM에게 어떠한 정보를 주고 return을 받아 우리 서비스에 입력해야한다.

## 서비스 소개
- 우리는 알 수 없는 알고리즘에 파묻혀 살고 있다. 하루에도 알 수 없는 알고리즘이 우리를 이끈다. 우리는 이 영상을 왜 보는지, 이 노래를 왜 듣고 있는지 알 수 없는 일이 대부분이다. 이번 서비스는 알 수 있는 알고리즘 즉, 내가 이 노래를 왜 추천 받았는지 사용자가 알 수 있도록 하는 것이다.

    ![alt text](/assets/images/hackathon_2.png)

- 사람들은 노래 추천을 받을 때, 장소, 기분, 날씨 등을 입력하기 마련이다. 예를 들어 '우울할 때 듣기 좋은 노래', '비오는 날 듣기 좋은 노래', 한강에서 듣기 좋은 노래' 등으로 검색해보면 그에 맞게 노래 리스트를 만들어 놓았으며 수요 또한 많이 있는 걸 확인할 수 있다.

- 이뿐만 아니라 '마이너한 감성'을 꼭 선택하는 사람도 있다. 시간을 돌이켜 보면 '혁오 밴드'나 '잔나비' 같은 가수들이 공중파 TV에 나왔을 때, 확인할 수 있던 반응을 생각해보면 '아... 나만 알고 있는 밴드였는데...' 등이 많았다. 이러한 뜻은 메이저한 감성보다는 마이너한 감성을 찾는 수요도 많다는 뜻이다.

    ![alt text](/assets/images/hackathon_3.png)

- 위와 같이 **장소, 기분, 날씨, 감성의 모든 조화(Harmony)**를 충족시키며 **노래(Harmony)**와 **AI 추천 서비스**를 합쳐 **HarmonAI**라는 이름이 탄생했다.

    ![alt text](/assets/images/hackathon_4.png)

- HarmonAI가 원하는 User eXperience는 추천받은 노래로 '아... 이 기분, 장소, 날씨 그리고... 감성까지'라는 말이 나오는 것이다.

    ![alt text](/assets/images/hackathon_5.png)

### 서비스 데모
![alt text](/assets/images/hackathon_6.png)
![alt text](/assets/images/hackathon_7.png)
![alt text](/assets/images/hackathon_8.png)


## FlowChart

- 우리 팀은 음악 추천 서비스를 하기로 하였다. 우리가 구상한 FlowChart는 아래와 같다.

    ![alt text](/assets/images/hackathon_1.png)

- AI 팀의 주 목적은 프롬프트 엔지니어링과 FastAPI로 ChatGPT가 추천해준 가수와 노래 제목을 백엔드로 전달하는 역할이다.

- 하지만 우리는 백엔드 개발자가 없으므로 대부분의 API를 Python으로 구축하여 프롬프트 엔지니어링 후 백엔드로 전달하기로 하였다.

## main.py (FastAPI 설계)

```python
from location import GetLocation
from wheather import Wheather
from recommend_songs import Recommend_songs

app = FastAPI()

class RequestData(BaseModel):
    latitude: float
    longitude: float
    query: str
    pop: int

class ResponseData(BaseModel):
    title: str
    artist: str

class RecommendationResponse(BaseModel):
    recommendations: List[ResponseData]

@app.post("/api/music/recommend", response_model = RecommendationResponse)
async def response_process(data: RequestData):
    
    loca = GetLocation(data).convert_coordinates_to_address()
    now_whea = Wheather(f"{loca.split(sep = " ")[1]}", f"{loca.split(sep = " ")[2]}")

    playlist = Recommend_songs(data)
    my_musics = playlist.recommend(f"{loca}", f"{now_whea.get_sky()}", 5, 
    {"configurable": {"thread_id": "Censored"}}, "Korean")

    df = pd.DataFrame(my_musics.items(), columns=['artist', 'title'])
    df = df[['title', 'artist']]
    songs_list = df.to_dict(orient = 'records')

    return JSONResponse(content={"recommendations": songs_list})
```

- 백엔드와 Json형식으로 주고 받기로 하였다. 해당 규약으로 RequestData class를 짰다. 위도, 경도, 기분, temperature 순이다.

- 우리가 보낼 ResponseData는 제목, 가수 순이다. 하지만 백엔드와의 규약으로 `recommendations` 안에 Json형태로 보내야하기 때문에 `List`형식으로 다시 묶었다.

## location.py (위도, 경도 → 지번 주소)

```python
class GetLocation:
  def __init__(self, data):
    load_dotenv()
    self.data = data
    self.google_map_key = os.getenv("GOOLEMAPS_API")

  def convert_coordinates_to_address(self):
    """
    입력받은 위도, 경도를 도로명 주소 및 지번 주소로 변환하여 반환
    """
    data_dict = self.data.dict()  # Pydantic 모델을 dict로 변환
    lat = float(data_dict["latitude"])
    long = float(data_dict["longitude"])
    self.gmaps = googlemaps.Client(key=self.google_map_key)
    result = self.gmaps.reverse_geocode((lat, long), language="ko")  # language="ko" 추가!
    return result[0]['formatted_address']
```

- API 키는 기본적으로 `.env`로 숨겨서 처리하였다.

- 우리는 지오코딩의 기능을 이용할 것이므로 [해당 링크](https://developers.google.com/maps/documentation/javascript/geocoding?hl=ko)를 참조하면 된다.

    ```
    results[]: {
    types[]: string,
    formatted_address: string,
    address_components[]: {
    short_name: string,
    long_name: string,
    postcode_localities[]: string,
    types[]: string
    },
    partial_match: boolean,
    place_id: string,
    postcode_localities[]: string,
    geometry: {
    location: LatLng,
    location_type: GeocoderLocationType
    viewport: LatLngBounds,
    bounds: LatLngBounds
    }
    }
    ```

- 우리가 필요한 건 `formatted_address`이고 나머지는 wheather에서 처리한다.

## wheather.py (지번 주소 → 날씨)

```python
class Wheather:
    def __init__(self, si, gu):
        data = pd.read_excel('./location_grids.xlsx')

        self.serviceKey = os.getenv("WEATHER_API")
        now = datetime.now()

        self.base_date = now.strftime("%Y%m%d")
        base_time = now.strftime("%H%M")
        self.si = si
        self.gu = gu
        grid = data[(data['1단계'] == self.si) & (data['2단계'] == self.gu)]
        if not grid.empty:
            self.nx = f"{grid.iloc[0]['격자 X']}"
            self.ny = f"{grid.iloc[0]['격자 Y']}"

        else:
            self.nx = '60'
            self.ny = '127'


        input_d = datetime.strptime(self.base_date + base_time, "%Y%m%d%H%M") - timedelta(hours = 1)
        input_datetime = input_d.strftime("%Y%m%d%H%M")

        input_date = input_datetime[:-4]
        input_time = input_datetime[-4:]

        self.url = f"http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst?serviceKey={self.serviceKey}&numOfRows=60&pageNo=1&dataType=json&base_date={self.base_date}&base_time={base_time}&nx={self.nx}&ny={self.ny}"

        self.deg_code = {0 : 'N', 360 : 'N', 180 : 'S', 270 : 'W', 90 : 'E', 22.5 :'NNE',
           45 : 'NE', 67.5 : 'ENE', 112.5 : 'ESE', 135 : 'SE', 157.5 : 'SSE',
           202.5 : 'SSW', 225 : 'SW', 247.5 : 'WSW', 292.5 : 'WNW', 315 : 'NW',
           337.5 : 'NNW'}

        self.pyt_code = {0 : '강수 없음', 1 : '비', 2 : '비/눈', 3 : '눈', 5 : '빗방울', 6 : '진눈깨비', 7 : '눈날림'}
        self.sky_code = {1 : '맑음', 3 : '구름많음', 4 : '흐림'}

    def get_info(self):
        response = requests.get(self.url, verify=False)
        res = json.loads(response.text)

        informations = dict()
        
        items = res.get('response', {}).get('body', {}).get('items', {}).get('item')
        if not items:
            # raise ValueError("예보 데이터를 가져오지 못했습니다. API 응답: " + json.dumps(res, ensure_ascii=False))
            return "오", "류"
        
        for item in items:
            cate = item['category']
            fcstTime = item['fcstTime']
            fcstValue = item['fcstValue']
            if fcstTime not in informations:
                informations[fcstTime] = dict()
            informations[fcstTime][cate] = fcstValue
            
        key = list(informations.keys())[-1]
        val = informations[key]

        return key, val

    def __call__(self):
        key, val = self.get_info()

        template = f"""{self.base_date[:4]}년 {self.base_date[4:6]}월 {self.base_date[-2:]}일 {key[:2]}시 {key[2:]}분 {(int(self.nx), int(self.ny))} 지역의 날씨는 """

        if val['SKY']:
            sky_temp = self.sky_code[int(val['SKY'])]
            template += sky_temp + " "

        if val['PTY'] :
            pty_temp = self.pyt_code[int(val['PTY'])]
            template += pty_temp
            if val['RN1'] != '강수없음' :
                rn1_temp = val['RN1']
                template += f"시간당 {rn1_temp}mm "

        if val['T1H'] :
            t1h_temp = float(val['T1H'])
            template += f" 기온 {t1h_temp}℃ "

        if val['REH'] :
            reh_temp = float(val['REH'])
            template += f"습도 {reh_temp}% "

        if val['VEC'] and val['WSD']:
            vec_temp = self.deg_to_dir(float(val['VEC']))
            wsd_temp = val['WSD']
            template += f"풍속 {vec_temp} 방향 {wsd_temp}m/s"

        return template

    def get_sky(self):
        key, val = self.get_info()
        if val == "류":
            return "맑음"
        
        template = ""

        if val['SKY']:
            sky_temp = self.sky_code[int(val['SKY'])]
            template += sky_temp

        return template



    def deg_to_dir(self, deg) :
        close_dir = ''
        min_abs = 360
        if deg not in self.deg_code.keys() :
            for key in self.deg_code.keys() :
                if abs(key - deg) < min_abs :
                    min_abs = abs(key - deg)
                    close_dir = self.deg_code[key]
        else :
            close_dir = self.deg_code[deg]
        return close_dir
```

- 기상청 단기 예보 서비스 API를 기본적으로 사용한다. [해당 링크](https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15084084)를 참조하면 된다.

- 기상청에서 제공하는 `location_grid`는 3단계로 나누는데, 3단계까지 하면 search하는데 너무 오래 걸리고 날씨라는게 바로 옆동네라고 아주 달라지지 않으니 3단계는 제거하고 사용하였다.

- 1단계는 `si`로 받아오고 2단계는 `gu`로 받아와 사용하였다.

- 제대로된 주소를 불러오지 못한다면 60, 127 `서울특별시 종로구`로 설정하고 기상청API를 못 불러오는 경우가 있는데 이때 날씨는 `맑음`으로 return하기로 합의했다.

## recommend_songs.py (주소, 날씨, query, pop → 추천 노래)

```python
class Recommend_songs:
    def __init__(self, data):
        self.recommended_songs = {}
        self.data = data
        load_dotenv()
        
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")

    def recommend(self, my_location, my_weather, target, config, language):
        data_dict = self.data.dict()  # Pydantic 모델을 dict로 변환
        pop = int(data_dict["pop"])
        query = data_dict["query"]
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "사용자가 기분을 입력하면 감성을 분석해서 해당 감성에 맞는 장르의 노래를 추천해줘. "
                    f"현재 장소는 {my_location}이고 오늘의 날씨는 {my_weather}이야. "
                    f"오늘의 장소와 날씨, 그리고 사용자의 감성을 분석해서 어울리는 노래 {target * 2}개를 추천해줘. "
                    "사용자의 언어를 고려하여 해당 언어가 속한 국가의 노래 위주로 70%, "
                    "이외 글로벌한 국가에 대해 30% 비중으로 노래를 추천해줘. "
                    "출력 형식은 반드시 JSON이어야 하며, 자연어는 출력하지 마. "
                    "아티스트나 노래 제목에 쌍따옴표가 있는 경우 작은따옴표로 변환해서 출력해줘."
                    "출력 형식 예시는 다음과 같아: "
                    '{{ "iu": "좋은 날", "blackpink": "How You Like That" , "Justin Timberlake": "Can\'t Stop the Feeling!"}}. '
                    "반드시 Spotify에서 검색 가능한 공식 아티스트명과 곡 제목을 사용해줘."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
            )
        class State(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            language: str

        class CustomState(State):
            messages: list
            language: str

        while len(self.recommended_songs) < target:
            def call_model(state: CustomState):
                prompt = self.prompt_template.invoke(
                    {"messages": state["messages"], "language": state["language"]}
                )
                response = self.model.invoke(prompt)
                return {"messages": response}

            workflow = StateGraph(state_schema=CustomState)
            workflow.add_edge(START, "model")
            workflow.add_node("model", call_model)

            app = workflow.compile()

            input_messages = [HumanMessage(query)]
            output = app.invoke(
                {"messages": input_messages, "language": language}
            )

            music_dict = output["messages"].content
            music_dict = music_dict.replace("'", "")
            if not music_dict:
                continue
            try:
                music_dict = json.loads(music_dict)
            except:
                continue

            for key, value in music_dict.items():
                artist, track = key, value
                query = f"{artist} {track}"  # 아티스트 + 곡 제목 검색
                results = self.sp.search(q=query, type="track", limit=1)

                try:
                    track_popularity = results["tracks"]["items"][0]["popularity"]
                    if track_popularity <= pop:
                        self.recommended_songs[artist] = track

                    if len(self.recommended_songs) == target:
                        break
                except:
                    continue
        
        return self.recommended_songs
```

- 프롬프트 엔지니어링은 엔지니어링은 위와 같이 한 것을 알 수 있으며

- `pop`으로 불러 온 Temperature 점수를 spotipyAPI로 검증하는 과정을 볼 수 있다.

## Local 실행 결과

- 전달

    ```
        -H "Content-Type: application/json" \
        -d '{
            "latitude": 37.5665,
            "longitude": 126.9780,
            "question": "기분 좋은 노래 추천",
            "pop": 5
            }'
    ```

- 출력
    ```
    {
        "recommendations": [
            {
                "artist": "아이유",
                "title": "좋은 날"
            },
            {
                "artist": "블랙핑크",
                "title": "Lovesick Girls"
            },
            {
                "artist": "백예린",
                "title": "우주를 건너"
            },
            {
                "artist": "적재",
                "title": "나쁜 사람"
            },
            {
                "artist": "키아라",
                "title": "Gold"
            }
        ]
    }
    ```

- 정상적으로 출력되는 것을 확인할 수 있었으며 Temperature 점수가 절반(`pop = 5`)정도면 5곡 중 2곡 정도가 유명하지 않은 노래로 확인되었다.

## 발전 방향

- Spotify API를 사용한 김에 Spotify로 노래 리스트를 뽑으려 했지만 Spotify가 유료라 할 순 없었다. 나중에 Frontend 측에서 Spotify로 로그인 할 수 있게 하면 연동하여 나만의 추천 리스트를 만들 수 있을 것이다.

- 위와 같은 사항으로 YouTube API를 사용하였는데, YouTube API 정책상 play버튼만 만들어 노래를 재생할 수 없다. 따라서 링크로 대체하였는데 이 또한 유료 계정이 있으면 해결할 수 있다. 나중에 Spotify로 바꾼다면 이 걱정은 없어질 것이다.

## 회고

- 풀스택 인원이 없는 상태에서 2일이라는 짧은 시간동안 밤 세워가며 배포까지 완료해보았다. 실제 테스트 결과 아주 잘 나왔으며 풀스택 인원이 있다면 좀 더 수월하지 않았을까 생각한다. 다른 팀에 비해서 조금 완성도가 떨어진 감이 있지만 이번 해커톤의 목표는 MVP모델이었고 인원도 부족한 상태에서 상당히 만족한 결과가 나왔다.<br/>
 첫 목표는 완성이었지만 어떨결에 본선까지 진출했다. 상을 타면 더 좋았겠지만 아쉽게 수상하지는 못했다. 해커톤이 끝난 후로 서비스를 종료하였지만 아주 좋은 경험이었다. 해커톤이 왜 필요한지 협업이 왜 중요한지 제대로 알 수 있는 기회였다.<br/>
 인원이 부족한 상태에서도 서로 그때 그때 공부하면서 디버깅하였다. 같이 밤 세워가며 배포까지 무사히 마칠 수 있도록 도와준 팀원들에게 감사하다.