import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 불러오기
file_path = 'WalkvsRun.csv'  # CSV 파일 경로를 여기에 입력하세요.
df = pd.read_csv(file_path)
df = df.select_dtypes(include=['float64', 'int64'])

# 필요한 열만 선택 (가속도와 자이로스코프 데이터)
selected_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']

# 상관 행렬 계산
corr_matrix = df[selected_columns].corr()

# 산점도 (이상치 제거 전 데이터)
plt.figure(figsize=(8, 6))  # 그래프 크기 조절
plt.scatter(df['acceleration_x'], df['acceleration_z'], alpha=0.7)
plt.title('Scatter Plot of Acceleration X vs Z')  # 제목 설정
plt.xlabel('Acceleration X')  # X축 라벨
plt.ylabel('Acceleration Z')  # Y축 라벨
plt.grid(True)  # 그리드 추가
plt.show()

