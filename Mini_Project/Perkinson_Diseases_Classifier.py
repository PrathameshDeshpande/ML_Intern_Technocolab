import pandas as pd
from xgboost import XGBClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/91797/Desktop/ML_Intern_Technocolab/Mini_Project/data.csv")
x = df.drop(["name","status"],axis=1)
y = df["status"]
y = pd.DataFrame(y)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    stratify=y,
                                                    test_size=0.20)
XGBC = XGBClassifier()
XGBC.fit(x_train,y_train)
print(XGBC)
y_pred = XGBC.predict(x_test)
y_pred = pd.DataFrame(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)