import FinanceDataReader as fdr
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 결정 트리 모델 생성
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# 모델 평가
score = model.score(X_test, y_test)
print(model.predict(X_test))
print(y_test)

# 결과 출력
print("모델 정확도: ", score)


# 결론, 시계열에 맞지 않는 예측 방식인듯