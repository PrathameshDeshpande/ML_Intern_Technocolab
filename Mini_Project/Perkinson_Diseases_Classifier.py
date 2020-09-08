import pandas as pd
import xgboost
import csv
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/91797/Desktop/ML_Intern_Technocolab/Mini_Project/data.csv")
x = df.drop(["name","status"],axis=1)
y = df["status"]
y = pd.DataFrame(y)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    stratify=y,
                                                    test_size=0.20)

