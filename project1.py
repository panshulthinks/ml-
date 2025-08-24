import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'C:\Users\Panshul\Downloads\archive (1)/Ecommerce Customers.csv')
print(df.head())

#lets do some data visualization(also knawn as exploratory data analysis)
#sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)
#plt.show()
#sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)
#plt.show()
sns.pairplot(df)
plt.show()
#sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)
#plt.show()
from sklearn.model_selection import train_test_split
x = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
lm.coef_
cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
print(cdf)


# predicting the test set results
predictions = lm.predict(x_test)
sns.scatterplot(y_test, predictions)
plt.xlabel('evaluation of our model')

#calculating matrice like mean absolute error, mean squared error, root mean squared error
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
plt.show()

# plotting the residuals
sns.displot((y_test - predictions), bins=50)    
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()

