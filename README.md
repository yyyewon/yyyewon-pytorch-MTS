# walkvsrun 데이터 분석

- **date**: 데이터가 기록된 날짜 (YYYY-MM-DD 형식)
- **time**: 데이터가 기록된 시간 (HH:MM:SS:나노초 형식)
- **username**: 데이터를 제공한 사용자 이름 (여기서는 "viktor")
- **wrist**: 손목 장치를 착용한 손(0은 왼손, 1은 오른손)
- **activity**: 활동 유형(0은 걷기, 1은 달리기)
- **acceleration_x**: X축 방향의 가속도 값 (m/s² 단위)
- **acceleration_y**: Y축 방향의 가속도 값 (m/s² 단위)
- **acceleration_z**: Z축 방향의 가속도 값 (m/s² 단위)
- **gyro_x**: X축 방향의 자이로스코프 값 (회전 속도, rad/s 단위)
- **gyro_y**: Y축 방향의 자이로스코프 값 (회전 속도, rad/s 단위)
- **gyro_z**: Z축 방향의 자이로스코프 값 (회전 속도, rad/s 단위)
- **x축**: 왼쪽-오른쪽 방향의 움직임.
- **y축**: 앞뒤 방향의 움직임.
- **z축**: 위아래 방향의 움직임.
- **가속도계 (Accelerometer)**: 물체의 가속도를 측정. 예를 들어, 손목을 빠르게 위로 올리거나 내릴 때 z축 가속도 값이 변함.
- **자이로스코프 (Gyroscope)**: 물체의 각속도(회전 속도)를 측정. 예를 들어, 손목을 돌릴 때 x, y, z축 각속도 값이 달라짐.

# 상관관계 분석
![상관관계분석](https://github.com/yyyewon/yyyewon-pytorch-MTS/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-11-29%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2010.27.27.png)

- 대각선 값 (자기 자신과의 상관 관계): 모두 1
- 가속도 데이터 간의 상관 관계:
    - `acceleration_x`와 `acceleration_y`: 상관 계수 = **0.27**→ 약한 음의 상관 관계 (한 값이 증가하면 다른 값은 약간 감소).
    - `acceleration_x`와 `acceleration_z`: 상관 계수 = **0.55**→ 중간 정도의 음의 상관 관계 (x와 z가 서로 반대 방향으로 움직이는 경향).
    - `acceleration_y`와 `acceleration_z`: 상관 계수 = **0.11**→ 거의 상관 관계 없음 (약한 양의 상관 관계)
- 가속도와 자이로스코프 데이터 간의 상관 관계: 모두 0에 가까움, 즉 유의미한 관계가 거의 없음.
- 자이로스코프 데이터 간의 상관 관계:
    - `gyro_x`와 `gyro_y`: 상관 계수 = **0.094**→ 거의 상관 관계 없음.
    - `gyro_x`와 `gyro_z`: 상관 계수 = **0.32**→ 약한 양의 상관 관계.
    - `gyro_y`와 `gyro_z`: 상관 계수 = **0.29**→ 약한 양의 상관 관계.
 
# 시각화
![x와 z간의 산점도](https://github.com/yyyewon/yyyewon-pytorch-MTS/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-11-29%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2011.07.15.png)

- `acceleration_x` (x축)와 **`acceleration_z` (y축)** 간의 산점도
- 점들이 아래로 기울어진 대각선 형태 = 음의 상관관계

# main.ipynb
다양한 머신러닝 및 딥러닝 분류 모델(FCN, ResNet 등)을 여러 데이터셋에 대해 학습하고 평가하는 멀티-모델/멀티-데이터셋 실험 프레임워크
