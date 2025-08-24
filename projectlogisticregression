import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv(r'C:\Users\Panshul\Downloads\archive (2)\data.csv')
print(data.head())
print(data.columns)

## lets visualise it we are using heatmap so that we could also filter the data which is null
## in simple terms we are doing eda
sns.heatmap(data.isnull())
plt.show()
## thus there is a whole column which has null values 
## so we will drop that column
data.drop(['Unnamed: 32'],axis=1, inplace= True)
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
data["diagnosis"] = data['diagnosis'].astype("category", copy=False)
data["diagnosis"].value_counts().plot(kind="bar")
plt.show()


y = data["diagnosis"]
X = data.drop(["diagnosis"], axis=1)

from sklearn.preprocessing import StandardScaler

# Create a scaler object
scaler = StandardScaler()
# Fit the scaler to the data and transform the data
X_scaled = scaler.fit_transform(X)
# X_scaled is now a numpy array with normalized data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)


from sklearn.linear_model import LogisticRegression
# Create logistic regression model
lr = LogisticRegression()
# Train the model on the training data
lr.fit(X_train, y_train)
# Predict the target variable on the test data
y_pred = lr.predict(X_test)



from sklearn.metrics import accuracy_score
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))