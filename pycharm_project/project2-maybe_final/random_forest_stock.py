import FinanceDataReader as fdr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

fdr.__version__
df_spx = fdr.StockListing('S&P500')
# 한국거래소 상장종목 요약정보
df_kprice = fdr.StockListing('KRX')
df_krx = fdr.StockListing('KRX-DESC')
# 실물 주식만 추출
df_krx = df_krx.dropna(subset=['Sector'])
df = fdr.DataReader('005930', '2023-01-01', '2023-06-30')

X = df.drop('Close', axis=1)
y = df['Close']

# Train, Test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest 모델 구축
model = RandomForestClassifier(n_estimators=123, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 모델 평가
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# 분류 보고서 출력
# print("Classification Report:")
# print(classification_report(y_test, predictions))
