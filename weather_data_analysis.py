import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import data
df = pd.read_csv("f:\project\ML_projects\logistic_regression\seattle-weather.csv")

df.info()

# Split data into traing and test set
X = df.drop(columns=["weather", "date"])
y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
lr = LogisticRegression(max_iter=1000)

# Train the model
lr.fit(X_train, y_train)

# PREDICT
y_pred = lr.predict(X_test)

print ("accuracy: %.2f" % accuracy_score(y_test, y_pred))