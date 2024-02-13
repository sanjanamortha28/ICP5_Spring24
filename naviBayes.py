import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
glass_data = pd.read_csv('glass.csv')
x = glass_data.drop(["Type"], axis=1) 
y=glass_data["Type"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
model = GaussianNB() 
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
Score =  model.score(X_test, y_test)
report = classification_report(y_test, y_pred)
print("Accuracy score: {:.2f}%".format(Score * 100)) 
print("\nClassification Report:\n", report)