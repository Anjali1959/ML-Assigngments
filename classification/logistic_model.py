import pandas as pd
from sklearn.linear_model import LogisticRegression

# heart.csv load karna
df = pd.read_csv('heart.csv')
X = df[['trestbps']]
y = df['target']

model = LogisticRegression().fit(X, y)
print("Prediction for BP 150:", model.predict([[150]]))
