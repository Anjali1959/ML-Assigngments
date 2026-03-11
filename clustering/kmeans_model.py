from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('heart.csv')
X = df[['trestbps']]

model = KMeans(n_clusters=2, random_state=0).fit(X)
print("Cluster Labels:", model.labels_)
