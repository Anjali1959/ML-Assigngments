import pandas as pd
from sklearn.model_selection import KFold


try:
    df = pd.read_csv('student_data.csv')
    data_marks = df['Marks'].values

    kf = KFold(n_splits=5)
    print("--- K-Fold Cross Validation Results ---")
    for train_index, test_index in kf.split(data_marks):
        print(f"Train indices: {train_index} | Test indices: {test_index}")
except:
    print("Please upload student_data.csv in this folder too!")
