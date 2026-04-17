import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------
# Title
# -----------------------
st.title("🛍️ Customer Segmentation App")
st.write("K-Means Clustering on Mall Customers Dataset")

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv("Mall_Customers.csv")

# Show data (optional)
if st.checkbox("Show Dataset"):
    st.write(df.head())

# -----------------------
# Feature Selection
# -----------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# Train Model
# -----------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# -----------------------
# User Input
# -----------------------
st.sidebar.header("Enter Customer Details")

income = st.sidebar.slider("Annual Income (k$)", 0, 150, 50)
spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# -----------------------
# Prediction
# -----------------------
user_data = scaler.transform([[income, spending]])
cluster = kmeans.predict(user_data)[0]

# -----------------------
# Cluster Meaning
# -----------------------
cluster_names = {
    0: "💎 High Value Customer",
    1: "💡 Potential Customer",
    2: "🛒 Impulsive Buyer",
    3: "📉 Low Engagement",
    4: "📊 Average Customer"
}

st.subheader("Prediction Result")
st.write(f"Customer belongs to: **{cluster_names[cluster]}**")

# -----------------------
# Visualization
# -----------------------
st.subheader("Customer Segmentation Visualization")

fig, ax = plt.subplots()

scatter = ax.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=df['Cluster'],
)

# Plot user point
ax.scatter(income, spending, marker='X', s=200)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments")

st.pyplot(fig)