import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 불러오기
file_path = 'WalkvsRun.csv'  # CSV 파일 경로를 여기에 입력하세요.
df = pd.read_csv(file_path)

# 필요한 열만 선택 (가속도와 자이로스코프 데이터)
selected_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']

# 상관 행렬 계산
corr_matrix = df[selected_columns].corr()

# 상관 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('WalkVsRun correlation matrix')
plt.show()
