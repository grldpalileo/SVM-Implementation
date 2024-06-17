import pandas as pd

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Extract the 'Purchased' variable
purchased_counts = dataset['Purchased'].value_counts()

# Display total counts for 'Purchased'
print("Total Counts for 'Purchased':")
print(purchased_counts)
