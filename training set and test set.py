import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print("Training Set for X")
print(X_train[:11],

print("\nTesting Set for X"),
print(X_test[:11]),

print("\nTraining Set for Y"),
print(y_train[:11]),

print("\nTesting Set for Y"),
print(y_test[:11]))
