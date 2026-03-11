import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Dataset load karna jo aapne upload kiya hai
df = pd.read_csv('student_data.csv')

# One-Hot Encoding ka logic
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['Education']])

print("One-Hot Encoding complete!")
print(encoded_data.toarray())
