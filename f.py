import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

#set random seed for reproducibility
np.random.seed(0)
student_ids = np.arange(1,21)
english_scores = np.random.randint(50,100,size=20)
science_scores = np.random.randint(50,100,size=20)

#create math score with real relationship
noise = np.random.normal(0,5,size=20)
math_scores = 0.5 * english_scores + 0.3 *science_scores + 10 + noise
math_score = math_scores.round().astype(int)

#create a binary target:1 if math >= 75(pass),0 otherwise (fail)
passed = (math_scores >= 75).astype(int)

df = pd.DataFrame({
    'StudentID': student_ids,
    'Math': math_scores,
    'English' : english_scores,
    'Science' : science_scores,
    'Passed' : passed
})

#Features and target
X = df[['English','Science']]
y = df['Passed']

#Split into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Create and train and test sets
model = LogisticRegression()
model.fit(X_train,y_train)

#Predicit on test set
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print("Accuraccy:",accuracy)
print("\nClassification Report:\n",classification_report(y_test,y_pred))

#Show predictions vs actual
print("Predicted Passed:",y_pred)
print("Actual Passed:",y_test.values)