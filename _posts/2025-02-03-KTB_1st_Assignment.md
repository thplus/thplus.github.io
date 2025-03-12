---
title: 1주차 과제
date: 2025-02-03
categories: [Assignment, 1st Week]
tags: [source code, python]     # TAG names should always be lowercase
math: true
---

# Library Import


```python
# import 하기
import random
import re
import time
import os
import platform
```

# Class 정의


```python
class Character:
  def __init__(self,name,health,power,level,exp):
    self.name = name
    self.health = health
    self.power = power
    self.level = level
    self.exp = exp
  def attack(self,target):
    target.health -= self.power
  def death(self):
    return self.health <= 0
  def acc(self):
    acc = random.random()
    return acc
```


```python
class Player(Character):
  def __init__(self,name,health,power,level,exp):
    super().__init__(name,health,power,level,exp)
    self.max_exp = level * 100
    self.max_health = level * 120
  def level_up(self):
    self.level += 1
    self.health += 10
    self.power += 5
    self.exp -= self.max_exp
    print(f"{self.name}의 레벨이 올랐습니다.")
    print(f"레벨 : {self.level}")
    print(f"현재 체력 : {self.health}")
    print(f"공격력 : {self.power}")
    print(f"최대 체력 : {self.max_health}")
    print(f"남은 경험치 : {self.max_exp - self.exp}")
```


```python
class Monster(Character):
  def __init__(self,name,health,power,level,exp):
    super().__init__(name,health,power,level,exp)
```


```python
characters = {} # character 슬롯 초기화
```

# 함수정의


```python
def clear():
  ostype = platform.system()

  if ostype == "Linux" or ostype == "Darwin":
    time.sleep(1.5)
    os.system('clear')
  elif ostype == "Windows":
    time.sleep(1.5)
    os.system('cls')
```


```python
def select_char():
  global igp
  if len(characters) == 0:
    print("선택할 캐릭터가 없습니다.")
    print("캐릭터를 만드시겠습니까? Y or N")
    sel = input()
    while True:
      if sel == "Y" or sel == "y":
        clear()
        create_char()
        break
      elif sel == "N" or sel == "n":
        clear()
        start()
        break
      else:
        print("Y or N 중에 선택해주세요.")
        sel = input()
  else:
    print("-------------------------------")
    print("캐릭터를 선택해주세요.")
    print("0. 되돌아가기")
    for i in range(len(characters)):
        print(f"{i+1}. {list(characters)[i]}")
    print("-------------------------------")
    sel_char = input("원하는 캐릭터를 숫자로 입력해주세요.")
    while True:
        if re.match('[0-9]', sel_char) == None:
            sel_char = input("원하는 캐릭터를 숫자로 입력해주세요.")
        else:
            sel_char = int(sel_char)
            if sel_char > len(characters)+1:
                print("없는 캐릭터입니다.")
                sel_char = input("원하는 캐릭터를 숫자로 입력해주세요.")
            elif sel_char < 0:
                print("없는 캐릭터입니다.")
                sel_char = input("원하는 캐릭터를 숫자로 입력해주세요.")
            elif sel_char == 0:
                clear()
                start()
                break
            break
    print(f"{list(characters)[sel_char - 1]}(이)가 선택되었습니다.")
    print("플레이 하시겠습니까? Y or N")
    sel = input()
    while True:
        if sel == "Y" or sel == "y":
            igp = characters[list(characters)[sel_char - 1]]
            time.sleep(1.5)
            clear()
            play()
            break
        elif sel == "N" or sel == "n":
            clear()
            start()
            break
        else:
            print("Y or N 중에 선택해주세요.")
            sel = input()
```


```python
def create_char():
  print("------------------------------")
  print("캐릭터를 생성합니다.")
  print("캐릭터 이름을 입력해주세요.")
  char_name = input()
  characters[char_name] = Player(char_name,100,10,1,0)
  print("캐릭터 생성이 완료 되었습니다.")
  print("------------------------------")
  clear()
  start()
```


```python
def start():
  print("------------------------------")
  print("원하시는 메뉴를 선택해주세요.")
  print("1. 캐릭터 선택")
  print("2. 캐릭터 생성")
  print("3. 종료")
  print("------------------------------")
  sel_menu = input("1, 2, 3 중 하나만 선택해주세요.")
  while True:
    if sel_menu == "1":
      clear()
      select_char()
      break
    elif sel_menu == "2":
      clear()
      create_char()
      break
    elif sel_menu == "3":
      break
    else:
      print("1,2,3 중에 선택해주세요.")
      sel_menu = input()
```


```python
def play():
  print(f"{igp.name}님 환영합니다.")
  while True:
    print("무얼 하시겠습니까?")
    print("1. 정보확인")
    print("2. 탐험")
    print("3. 메뉴로 돌아가기")
    sel = input()
    if sel == "1":
      clear()
      status(igp)
    elif sel == "2":
      clear()
      rpg()
      break
    elif sel == "3":
      clear()
      start()
      break
    else:
      print("1,2,3 중에 선택해주세요.")
      sel = input()
```


```python
def status(char):
  print(f"이름 : {char.name}")
  print(f"레벨 : {char.level}")
  print(f"현재 체력 : {char.health}")
  print(f"공격력 : {char.power}")
  print(f"경험치 : {char.exp}")
```


```python
def rpg():
  #일단 몬스터 초기화
  mosters = {}
  slime = Monster("슬라임",15,5,1,100)
  mosters[slime.name] = slime
  oak = Monster("오크",30,10,3,100)
  mosters[oak.name] = oak
  barlog = Monster("발록",50,15,10,100)
  mosters[barlog.name] = barlog
  searching = random.random()
  #print(searching)
  #몬스터 만날 확률 random함수가 정규분포를 따르는거 같아서 양 끝단에 완전 강한 애, 그 이후에 만나지 못함, 그 다음 오크, 그 다음 슬라임
  if (searching >= 0.05 and searching < 0.1) or (searching > 0.9 and searching <= 0.95):
    print("몬스터를 만나지 못했습니다.")
    play()
  elif searching >= 0.35 and searching <= 0.65:
    fight(igp, slime)
  elif (searching >= 0.1 and searching < 0.35) or (searching > 0.65 and searching <= 0.9):
    fight(igp, oak)
  elif searching < 0.05 or searching > 0.95:
    fight(igp, barlog)
```


```python
def mon_atk(monster, igp):
  if monster.acc() <= 0.75:
    monster.attack(igp)
    print("-----------------------")
    print(f"{monster.power}의 데미지를 입었습니다.")
    print(f"{igp.name}의 현재 체력: {igp.health}")
    print("-----------------------")
  elif monster.acc() > 0.75:
    print("-----------------------")
    print(f"{monster.name}의 공격이 실패했습니다.")
    print("-----------------------")
```


```python
def player_death(igp):
  if igp.death() == True:
    print("-----------------------")
    print(f"{igp.name}이 죽었습니다.")
    print(f"획득한 경험치를 모두 잃습니다.")
    print("-----------------------")
    igp.health = igp.max_health
    igp.exp = 0
```


```python
def player_atk(igp, monster):
  if igp.acc() <= 0.95:
    igp.attack(monster)
    print("-----------------------")
    print(f"{monster.name}에게 {igp.power}의 데미지를 입혔습니다.")
    print(f"{monster.name}의 현재 체력 : {monster.health}")
    print("-----------------------")
  elif igp.acc() > 0.95:
    print("-----------------------")
    print("공격에 실패했습니다.")
    print("-----------------------")
```


```python
def get_exp(igp, monster):
  igp.exp += monster.exp
  print(f"{monster.exp}만큼의 경험치를 획득합니다.")
  print("-----------------------")
  #경험치 max시 레벨업
  if igp.exp >= igp.max_exp:
    igp.level_up()
```


```python
def fight(igp, monster):
  print(f"{monster.name}를 만났습니다.")
  while True:
    print("-----------------------")
    print(f"{monster.name}의 체력: {monster.health}")
    print(f"{monster.name}의 공격력: {monster.power}")
    print("어떤 행동을 하시겠습니까?")
    print("1. 공격")
    print("2. 도망")
    print("3. 정보확인")
    print("-----------------------")
    sel = input()
    if sel == "1":
      #공격 확률 95%
      if monster.death() == False and igp.death() == False:
        player_atk(igp, monster)
        #몬스터 죽으면 경험치
        if monster.death() == True:
          print("-----------------------")
          print(f"{monster.name}이 죽었습니다.")
          get_exp(igp, monster)
          clear()
          play()
          break
      #player 공격 끝내면 monster 공격
        else:
          mon_atk(monster, igp)
          if igp.death() == True:
            player_death(igp)
            clear()
            start()
            break
    elif sel == "2":
      #도망확률 70% 양끝단 0.15씩 제외
      runs = random.random()
      if runs >= 0.15 and runs <= 0.85:
        print("-----------------------")
        print("도망쳤습니다.")
        print("-----------------------")
        clear()
        play()
        break
      else:
        print("-----------------------")
        print("도망치지 못했습니다.")
        print("-----------------------")
        if monster.death() == False and igp.death() == False:
          mon_atk(monster, igp)
          if igp.death() == True:
            player_death(igp)
            clear()
            start()
            break
    elif sel == "3":
      status(igp)
      status(monster)
    else:
      print("1,2,3 중에 선택해주세요.")
      sel = input()
```


```python
# 실제 실행 main()함수 부분
def main():
  igp = None #in game player 초기화
  start()
```


```python
main()
```

    ------------------------------
    원하시는 메뉴를 선택해주세요.
    1. 캐릭터 선택
    2. 캐릭터 생성
    3. 종료
    ------------------------------
    1, 2, 3 중 하나만 선택해주세요.1
    선택할 캐릭터가 없습니다.
    캐릭터를 만드시겠습니까? Y or N
    Y
    ------------------------------
    캐릭터를 생성합니다.
    캐릭터 이름을 입력해주세요.
    brix
    캐릭터 생성이 완료 되었습니다.
    ------------------------------
    ------------------------------
    원하시는 메뉴를 선택해주세요.
    1. 캐릭터 선택
    2. 캐릭터 생성
    3. 종료
    ------------------------------
    1, 2, 3 중 하나만 선택해주세요.1
    -------------------------------
    캐릭터를 선택해주세요.
    0. 되돌아가기
    1. brix
    -------------------------------
    원하는 캐릭터를 숫자로 입력해주세요.1
    brix(이)가 선택되었습니다.
    플레이 하시겠습니까? Y or N
    Y
    brix님 환영합니다.
    무얼 하시겠습니까?
    1. 정보확인
    2. 탐험
    3. 메뉴로 돌아가기
    1
    이름 : brix
    레벨 : 1
    현재 체력 : 100
    공격력 : 10
    경험치 : 0
    무얼 하시겠습니까?
    1. 정보확인
    2. 탐험
    3. 메뉴로 돌아가기
    2
    슬라임를 만났습니다.
    -----------------------
    슬라임의 체력: 15
    슬라임의 공격력: 5
    어떤 행동을 하시겠습니까?
    1. 공격
    2. 도망
    3. 정보확인
    -----------------------
    1
    -----------------------
    슬라임에게 10의 데미지를 입혔습니다.
    슬라임의 현재 체력 : 5
    -----------------------
    -----------------------
    5의 데미지를 입었습니다.
    brix의 현재 체력: 95
    -----------------------
    -----------------------
    슬라임의 체력: 5
    슬라임의 공격력: 5
    어떤 행동을 하시겠습니까?
    1. 공격
    2. 도망
    3. 정보확인
    -----------------------
    1
    -----------------------
    슬라임에게 10의 데미지를 입혔습니다.
    슬라임의 현재 체력 : -5
    -----------------------
    -----------------------
    슬라임이 죽었습니다.
    100만큼의 경험치를 획득합니다.
    -----------------------
    brix의 레벨이 올랐습니다.
    레벨 : 2
    현재 체력 : 105
    공격력 : 15
    최대 체력 : 120
    남은 경험치 : 100
    brix님 환영합니다.
    무얼 하시겠습니까?
    1. 정보확인
    2. 탐험
    3. 메뉴로 돌아가기
    3
    ------------------------------
    원하시는 메뉴를 선택해주세요.
    1. 캐릭터 선택
    2. 캐릭터 생성
    3. 종료
    ------------------------------
    1, 2, 3 중 하나만 선택해주세요.3
    
