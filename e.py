import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1.Generate random data (same as before)
np.random.seed(0)
student_ids = np.arange(1,21)
math_scores = np.random.randint(50,100,size=20)
english_scores = np.random.randint(50,100,size=20)
science_scores = np.random.randint(50,100,size=20)

#2.Create a DataFrame with pandas 
df = pd.DataFrame({
    'StudentID': student_ids,
    'Math': math_scores,
    'English' : english_scores,
    'Science' : science_scores 
})

#Feature and target
x = df[['English','Science']]
y = df['Math']

#Split into train and test sets
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Create and train the odel
model = LinearRegression()
model.fit(X_train,y_train)

#Predict on test set
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

#Show Predictions vs actual
print("\nPredicted Math scores:",y_pred)
print("Actual Math scores:",y_test.values)