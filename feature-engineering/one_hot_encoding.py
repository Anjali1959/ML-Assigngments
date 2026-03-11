import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Dataset load karna (Jo hum abhi upload karenge)
df = pd.read_csv('student_data.csv')

# One-Hot Encoding ka logic
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['Education']])

print("Encoded Categories:\n", encoded_data.toarray())
