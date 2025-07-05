import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#1.Generate random data with NUMPY
np.random.seed(0)
student_ids = np.arange(1,21)
ENTC_scores = np.random.randint(25,100,size=20)
CSE_scores = np.random.randint(25,100,size=20)
MECH_scores = np.random.randint(25,100,size=20)

#2.Create a DataFrame with pandas 
df = pd.DataFrame({
    'StudentID': student_ids,
    'ENTC': ENTC_scores,
    'CSE' : CSE_scores,
    'MECH' : MECH_scores 
})

print("First 5 rows of the DataFrame:")
print(df.head())

#3.Basic analysis with pandas
print("\nAverage scores:")
print(df[['ENTC','CSE','MECH']].mean())

print("\nStudent with highest ENTC score:")
print(df.loc[df['ENTC'].idxmax()])

#4.Visualization with Matplotlib
plt.figure(figsize=(8,5))
plt.plot(df['StudentID'],df['ENTC'],marker='o',label='ENTC')
plt.plot(df['StudentID'],df['CSE'],marker='s',label='CSE')
plt.plot(df['StudentID'],df['MECH'],marker='^',label='MECH')
plt.xlabel('Student ID')
plt.ylabel('Score')
plt.title('Student Scores by Branch')
plt.legend()
plt.show()

#5.Visualization with Seaborn
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['ENTC','CSE','MECH']])
plt.title('Score Distribution by Branch')
plt.ylabel('Score')
plt.show()