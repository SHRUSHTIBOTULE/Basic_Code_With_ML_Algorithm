from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#Simple dataset:[height,weight]
X = [
    [150,50],
    [160,60],
    [170,65],
    [180,80]
]
#Label:0 =short, 1=Tall
y = [0,0,1,1]

#create and train the model
model = DecisionTreeClassifier()
model.fit(X,y)

#predict for the same data(for demonstartion)
y_pred = model.predict(X)

#compute confusion matrix
cm = confusion_matrix(y,y_pred)
print("Confusion Matrix:")
print(cm)







#precision=tp/tp+fp
#recall=tp/tp+fn
#accuracy=tp+tn/tp+fp+fn+tn
#f1 score=2*  (precision*recall/precision+recall)