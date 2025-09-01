import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Download data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# 2. Create DataFrame and shift target
df = data[['Open','High','Low','Volume','Close']].copy()
df['Target'] = df['Close'].shift(-1)        # tomorrow’s close
df = df.dropna()                            # drop last row which has no target

# 3. Visualize (optional)
sns.pairplot(df[['Open','High','Low','Close','Target']])
plt.show()

# 4. Prepare features and target
X = df[['Open','High','Low','Volume']]
y = df['Target']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict and evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse:.2f}")

# 8. Plot actual vs predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, preds, alpha=0.6)
plt.xlabel("Actual Tomorrow's Close Price")
plt.ylabel("Predicted Tomorrow's Close Price")
plt.title("Actual vs Predicted Tomorrow's Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
