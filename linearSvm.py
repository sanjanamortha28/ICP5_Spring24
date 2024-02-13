import warnings 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC 
from sklearn.metrics import classification_report 
warnings.filterwarnings("ignore")
glass_data = pd.read_csv('glass.csv')
X = glass_data.drop(['Type'], axis=1) 
y = glass_data["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = LinearSVC(random_state=42) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test) 
report = classification_report(y_test, y_pred)
print("Accuracy Score: {:.2f}%".format(score * 100)) 
print("\nClassification Report:\n", report)