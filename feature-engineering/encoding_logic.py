import pandas as pd
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('student_data.csv')


encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['Education']])

print("One-Hot Encoding complete!")
print(encoded_data.toarray())
