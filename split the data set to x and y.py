import pandas as pd

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Extract features (X) and target variable (y)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Display X as DataFrame
print("Features (X):")
df_X = pd.DataFrame(X[:11], columns=["Age", "Estimated Salary"])
print(df_X)

# Display y as DataFrame
print("\nTarget Variable (y):")
df_y = pd.DataFrame(y[:11], columns=["Purchased SUV"])
print(df_y)
