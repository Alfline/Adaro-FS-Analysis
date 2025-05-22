# Import necessary modules
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For enhanced visualization
from sklearn.preprocessing import StandardScaler  # For standardizing data
from sklearn.cluster import KMeans  # For K-means clustering

# Load the dataset
# We're using pandas to read the CSV file into a DataFrame data structure
df = pd.read_excel(r"C:\Users\asus\Desktop\Desktop\Books\Tugas Kuliah\CODING DL SELASA\Working File\1. Variable to be imported.xlsx")

# Select the three variables we want to analyze
# We're using list data type to store our variable names
variables = ['Revenue', 'Operating income', 'Profit for the period']

# Filter the DataFrame to only include these variables
# We're using boolean indexing to filter the DataFrame
filtered_df = df[df['Item'].isin(variables)]

print("Filtered DataFrame with selected variables:")
print(filtered_df)

# Extract data for each year
# We're creating separate arrays for each year's data
X_2023 = filtered_df['Q4_2023'].values.reshape(-1, 1)
X_2024 = filtered_df['Q4_2024'].values.reshape(-1, 1)

# Standardize the data for each year
# StandardScaler transforms data to have mean=0 and variance=1
scaler = StandardScaler()
X_2023_scaled = scaler.fit_transform(X_2023)
X_2024_scaled = scaler.fit_transform(X_2024)

# Define a function to perform K-means clustering
# This is a user-defined function that encapsulates the clustering logic
def perform_kmeans(X, n_clusters=2):
    """
    Perform K-means clustering on financial data.
    
    Parameters:
    X (array): Data array to cluster
    n_clusters (int): Number of clusters to form
    
    Returns:
    tuple: (cluster_labels, kmeans_model)
    """
    # Create K-means model
    # We're using the KMeans class from scikit-learn
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Control flow: try-except to handle potential errors
    try:
        # Fit the model to the data
        kmeans.fit(X)
        # Return the cluster labels and the model
        return kmeans.labels_, kmeans
    except Exception as e:
        print(f"Error in clustering: {e}")
        return None, None

# Perform clustering for each year
labels_2023, model_2023 = perform_kmeans(X_2023_scaled)
labels_2024, model_2024 = perform_kmeans(X_2024_scaled)

# Create DataFrames with the cluster labels
# We're combining data and labels in DataFrames for easier analysis
df_2023 = pd.DataFrame({
    'Variable': filtered_df['Item'].values,
    'Value': X_2023.flatten(),
    'Cluster': labels_2023 if labels_2023 is not None else np.nan
})

df_2024 = pd.DataFrame({
    'Variable': filtered_df['Item'].values,
    'Value': X_2024.flatten(),
    'Cluster': labels_2024 if labels_2024 is not None else np.nan
})

print("\n2023 Clusters:")
print(df_2023)

print("\n2024 Clusters:")
print(df_2024)

# Visualize the clusters using Bar Charts
# We'll create bar plots with color-coded clusters
plt.figure(figsize=(14, 6))

# 2023 Clusters Bar Chart
plt.subplot(1, 2, 1)
# Control flow: if statement to check if clustering was successful
if labels_2023 is not None:
    # Create a bar plot with colors based on cluster assignment
    # We're using a for loop to iterate through unique cluster labels
    for cluster in np.unique(labels_2023):
        # Filter the DataFrame for the current cluster
        cluster_data = df_2023[df_2023['Cluster'] == cluster]
        # Plot the bars for this cluster
        plt.bar(cluster_data['Variable'], cluster_data['Value'], 
                label=f'Cluster {cluster}', alpha=0.7)
    
    plt.title('Q4 2023 Financial Variables Clusters (Bar Chart)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value (USD)')
    plt.legend()
    # Format y-axis to show values in millions
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# 2024 Clusters Bar Chart
plt.subplot(1, 2, 2)
# Control flow: if statement to check if clustering was successful
if labels_2024 is not None:
    # Create a bar plot with colors based on cluster assignment
    # We're using a for loop to iterate through unique cluster labels
    for cluster in np.unique(labels_2024):
        # Filter the DataFrame for the current cluster
        cluster_data = df_2024[df_2024['Cluster'] == cluster]
        # Plot the bars for this cluster
        plt.bar(cluster_data['Variable'], cluster_data['Value'], 
                label=f'Cluster {cluster}', alpha=0.7)
    
    plt.title('Q4 2024 Financial Variables Clusters (Bar Chart)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value (USD)')
    plt.legend()
    # Format y-axis to show values in millions
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.show()

# Print cluster assignments for Bar Chart analysis
print("\nBar Chart Cluster Analysis:")
print("Q4 2023 - Variables in each cluster:")
# Control flow: for loop to iterate through clusters
for cluster in np.unique(labels_2023):
    cluster_vars = df_2023[df_2023['Cluster'] == cluster]['Variable'].tolist()
    print(f"  Cluster {cluster}: {cluster_vars}")

print("\nQ4 2024 - Variables in each cluster:")
# Control flow: for loop to iterate through clusters
for cluster in np.unique(labels_2024):
    cluster_vars = df_2024[df_2024['Cluster'] == cluster]['Variable'].tolist()
    print(f"  Cluster {cluster}: {cluster_vars}")
