import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Ethereum Transaction Analysis", page_icon="💰", layout="wide")

# Title and introduction
st.title("Ethereum Transaction Analysis Dashboard")
st.markdown("This dashboard provides insights into Ethereum transaction data, highlighting trends, patterns, and potential anomalies.")

# Load data
@st.cache_data
def load_data():
    # In a real app, you would load your CSV file here
    # For this example, we'll use the data provided in the question
    data = pd.read_csv("transaction_dataset.csv", index_col=0)
    
    # Preprocess the data
    # Convert column names to be more readable
    data.columns = [col.strip() for col in data.columns]
    
    # Handle missing values
    data = data.fillna(0)
    
    return data

# Load the data
data = load_data()

# Show raw data
with st.expander("View Raw Data"):
    st.dataframe(data)

# Sidebar for filtering
st.sidebar.header("Filters")
flag_filter = st.sidebar.selectbox("Filter by FLAG", ["All"] + list(data["FLAG"].unique()))
min_transactions = st.sidebar.slider("Minimum Total Transactions", int(data["total transactions (including tnx to create contract"].min()), 
                                   int(data["total transactions (including tnx to create contract"].max()), 
                                   int(data["total transactions (including tnx to create contract"].min()))

# Apply filters
filtered_data = data
if flag_filter != "All":
    filtered_data = filtered_data[filtered_data["FLAG"] == flag_filter]
filtered_data = filtered_data[filtered_data["total transactions (including tnx to create contract"] >= min_transactions]

# Main Dashboard
st.header("Transaction Analysis")

# In the Key Metrics section (around line 88)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Addresses", len(filtered_data))
with col2:
    st.metric("Total Transactions", int(filtered_data["total transactions (including tnx to create contract"].sum()))
with col3:
    st.metric("Total Ether Sent", f"{filtered_data['total Ether sent'].sum():.2f} ETH")
with col4:
    # Check if the column exists before using it
    if " Total ERC20 tnxs" in filtered_data.columns:
        st.metric("Total ERC20 Transactions", f"{filtered_data[' Total ERC20 tnxs'].sum():.0f}")
    else:
        st.metric("Total Ether Received", f"{filtered_data['total ether received'].sum():.2f} ETH")

# Transaction Distribution
st.subheader("Transaction Distribution")
col1, col2 = st.columns(2)

with col1:
    # Pie chart for sent vs received transactions
    fig = px.pie(
        names=["Sent Transactions", "Received Transactions"],
        values=[filtered_data["Sent tnx"].sum(), filtered_data["Received Tnx"].sum()],
        title="Sent vs Received Transactions",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig)

with col2:
    # Bar chart for transaction types
    # Bar chart for transaction types
    transaction_types = pd.DataFrame({
        'Type': ['Regular', 'Contract Creation'],
        'Count': [
            filtered_data["Sent tnx"].sum() + filtered_data["Received Tnx"].sum(),
            filtered_data["Number of Created Contracts"].sum()
        ]
    })
    # Add ERC20 transactions if the column exists
    if " Total ERC20 tnxs" in filtered_data.columns:
        transaction_types = pd.concat([
            transaction_types,
            pd.DataFrame({
                'Type': ['ERC20'],
                'Count': [filtered_data[" Total ERC20 tnxs"].sum()]
            })
        ], ignore_index=True)
    fig = px.bar(
        transaction_types, 
        x='Type', 
        y='Count',
        title="Transaction Types",
        color='Type',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig)

# Ether Flow Analysis
st.subheader("Ether Flow Analysis")
col1, col2 = st.columns(2)

with col1:
    # Scatter plot for sent vs received ether
    fig = px.scatter(
        filtered_data, 
        x="total Ether sent", 
        y="total ether received",
        size="total transactions (including tnx to create contract",
        hover_name="Address",
        title="Ether Sent vs Received (Size = Total Transactions)",
        color="FLAG",
        log_x=True, 
        log_y=True
    )
    fig.add_shape(
        type="line", 
        line=dict(dash="dash", color="gray"),
        x0=filtered_data["total Ether sent"].min(), 
        y0=filtered_data["total Ether sent"].min(),
        x1=filtered_data["total Ether sent"].max(), 
        y1=filtered_data["total Ether sent"].max()
    )
    st.plotly_chart(fig)

with col2:
    # Histogram of ether balance
    fig = px.histogram(
        filtered_data, 
        x="total ether balance",
        title="Distribution of Ether Balance",
        color_discrete_sequence=["#3366CC"]
    )
    st.plotly_chart(fig)

# Transaction Patterns
st.subheader("Transaction Patterns")
col1, col2 = st.columns(2)

with col1:
    # Time between transactions
    time_data = pd.melt(
        filtered_data, 
        id_vars=["Address"], 
        value_vars=["Avg min between sent tnx", "Avg min between received tnx"],
        var_name="Transaction Type", 
        value_name="Average Time (minutes)"
    )
    fig = px.box(
        time_data, 
        x="Transaction Type", 
        y="Average Time (minutes)",
        title="Time Between Transactions",
        color="Transaction Type",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig)

with col2:
    # Transaction value analysis
    value_data = pd.melt(
        filtered_data, 
        id_vars=["Address"], 
        value_vars=["avg val received", "avg val sent"],
        var_name="Transaction Type", 
        value_name="Average Value (ETH)"
    )
    fig = px.violin(
        value_data, 
        x="Transaction Type", 
        y="Average Value (ETH)",
        title="Transaction Value Analysis",
        color="Transaction Type",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig)

# Network Analysis
st.subheader("Network Activity")
col1, col2 = st.columns(2)

with col1:
    # Unique addresses
    unique_addr_data = pd.melt(
        filtered_data, 
        id_vars=["Address"], 
        value_vars=["Unique Received From Addresses", "Unique Sent To Addresses"],
        var_name="Address Type", 
        value_name="Count"
    )
    fig = px.bar(
        unique_addr_data.groupby("Address Type").sum().reset_index(), 
        x="Address Type", 
        y="Count",
        title="Unique Addresses Interaction",
        color="Address Type",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig)

with col2:
    # ERC20 Token Analysis
    if " Total ERC20 tnxs" in filtered_data.columns:
        erc20_data = filtered_data[filtered_data[" Total ERC20 tnxs"] > 0]
        if not erc20_data.empty:
            fig = px.scatter(
                erc20_data, 
                x=" ERC20 total Ether received", 
                y=" ERC20 total ether sent",
                size=" Total ERC20 tnxs",
                hover_name="Address",
                title="ERC20 Token Activity",
                color="FLAG",
                log_x=True, 
                log_y=True
            )
            st.plotly_chart(fig)
        else:
            st.info("No ERC20 transaction data available for the filtered dataset.")
    else:
        st.info("ERC20 transaction data not available in the dataset.")

# Identify Anomalies
st.header("Anomaly Detection")

# Calculate some anomaly scores
filtered_data["sent_received_ratio"] = filtered_data["Sent tnx"] / (filtered_data["Received Tnx"] + 1)  # Add 1 to avoid division by zero
filtered_data["ether_balance_ratio"] = filtered_data["total ether balance"] / (filtered_data["total Ether sent"] + filtered_data["total ether received"] + 0.001)  # Add small value to avoid division by zero

# Find potential anomalies
potential_anomalies = filtered_data[
    (filtered_data["sent_received_ratio"] > 10) |  # Very high ratio of sent to received transactions
    (filtered_data["sent_received_ratio"] < 0.1) |  # Very low ratio of sent to received transactions
    (filtered_data["ether_balance_ratio"] < -0.5) |  # Large negative balance relative to transaction volume
    (filtered_data["total transactions (including tnx to create contract"] > 1000)  # Very high number of transactions
]

if not potential_anomalies.empty:
    st.subheader("Potential Anomalies Detected")
    st.dataframe(potential_anomalies[["Address", "FLAG", "Sent tnx", "Received Tnx", "total Ether sent", "total ether received", "total ether balance", "total transactions (including tnx to create contract"]])
    
    # Plot anomalies
    fig = px.scatter(
        filtered_data, 
        x="sent_received_ratio", 
        y="ether_balance_ratio",
        size="total transactions (including tnx to create contract",
        color="FLAG",
        hover_name="Address",
        title="Anomaly Detection Plot",
        log_x=True
    )
    
    # Highlight anomalies
    fig.add_trace(
        go.Scatter(
            x=potential_anomalies["sent_received_ratio"],
            y=potential_anomalies["ether_balance_ratio"],
            mode="markers",
            marker=dict(size=12, color="red", line=dict(width=2, color="black")),
            name="Potential Anomalies"
        )
    )
    
    st.plotly_chart(fig)
else:
    st.info("No anomalies detected based on current criteria.")

# Time-based Analysis
st.header("Time-based Analysis")

# Filter addresses with sufficient time data
time_filtered_data = filtered_data[filtered_data["Time Diff between first and last (Mins)"] > 0]

if not time_filtered_data.empty:
    # Sort by time difference
    time_sorted_data = time_filtered_data.sort_values("Time Diff between first and last (Mins)", ascending=False).head(10)
    
    fig = px.bar(
        time_sorted_data,
        x="Address",
        y="Time Diff between first and last (Mins)",
        title="Top 10 Addresses by Activity Duration (minutes)",
        color="FLAG",
        hover_data=["Sent tnx", "Received Tnx", "total Ether sent", "total ether received"]
    )
    st.plotly_chart(fig)
    
    # Transaction frequency analysis
    st.subheader("Transaction Frequency Analysis")
    
    # Calculate transaction frequency (transactions per day)
    time_filtered_data["txns_per_day"] = time_filtered_data["total transactions (including tnx to create contract"] / (time_filtered_data["Time Diff between first and last (Mins)"] / (60 * 24))
    
    fig = px.histogram(
        time_filtered_data,
        x="txns_per_day",
        title="Transactions per Day Distribution",
        color_discrete_sequence=["#9C27B0"]
    )
    st.plotly_chart(fig)
else:
    st.info("Insufficient time data for analysis.")

# ERC20 Token Analysis
st.header("ERC20 Token Analysis")

# Check if we have ERC20 data
if " Total ERC20 tnxs" in filtered_data.columns:
    erc20_data = filtered_data[filtered_data[" Total ERC20 tnxs"] > 0]
    
    if not erc20_data.empty:
        # Most common token types
        if " ERC20_most_rec_token_type" in erc20_data.columns:
            token_counts = erc20_data[" ERC20_most_rec_token_type"].value_counts().reset_index()
            token_counts.columns = ["Token", "Count"]
            
            fig = px.pie(
                token_counts.head(10), 
                values="Count", 
                names="Token",
                title="Most Common ERC20 Token Types",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig)
        
        # ERC20 Transaction value analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if " ERC20 avg val sent" in erc20_data.columns and " ERC20 avg val rec" in erc20_data.columns:
                erc20_value_data = pd.melt(
                    erc20_data, 
                    id_vars=["Address"], 
                    value_vars=[" ERC20 avg val sent", " ERC20 avg val rec"],
                    var_name="Transaction Type", 
                    value_name="Average Value"
                )
                
                fig = px.box(
                    erc20_value_data, 
                    x="Transaction Type", 
                    y="Average Value",
                    title="ERC20 Transaction Value Analysis",
                    color="Transaction Type",
                    color_discrete_sequence=px.colors.qualitative.Pastel2,
                    log_y=True
                )
                st.plotly_chart(fig)
        
        with col2:
            if " ERC20 uniq sent token name" in erc20_data.columns:
                # Number of unique tokens per address
                erc20_data["num_unique_tokens"] = erc20_data[" ERC20 uniq sent token name"].apply(lambda x: 0 if pd.isna(x) else len(str(x).split(",")))
                
                fig = px.histogram(
                    erc20_data,
                    x="num_unique_tokens",
                    title="Number of Unique ERC20 Tokens per Address",
                    color_discrete_sequence=["#4CAF50"]
                )
                st.plotly_chart(fig)
    else:
        st.info("No ERC20 transaction data available for the filtered dataset.")
else:
    st.info("ERC20 transaction data not available in the dataset.")

# Predictive Analysis
st.header("Transaction Classification")

# Simple classification based on transaction patterns
def classify_address(row):
    if row["FLAG"] == 1:
        return "Flagged"
    
    # These are just example rules - in a real application, you'd use machine learning models
    if row["Sent tnx"] > 1000 and row["Received Tnx"] < 100:
        return "Potential One-way Flow"
    elif row["sent_received_ratio"] > 10:
        return "Heavy Sender"
    elif row["sent_received_ratio"] > 10:
        return "Heavy Sender"
    elif row["sent_received_ratio"] < 0.1:
        return "Heavy Receiver"
    elif row["total ether balance"] > 100:
        return "Whale"
    elif row["total transactions (including tnx to create contract"] > 500:
        return "Active Trader"
    else:
        return "Normal"

# Apply classification
filtered_data["classification"] = filtered_data.apply(classify_address, axis=1)

# Classification distribution
st.subheader("Address Classification Distribution")
classification_counts = filtered_data["classification"].value_counts().reset_index()
classification_counts.columns = ["Classification", "Count"]

fig = px.bar(
    classification_counts,
    x="Classification",
    y="Count",
    title="Address Classification Distribution",
    color="Classification",
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig)

# AI/ML Analysis
st.header("AI/ML Based Analysis")

# Prepare data for ML
ml_features = filtered_data[["Sent tnx", "Received Tnx", "total Ether sent", "total ether received", 
                          "total ether balance", "total transactions (including tnx to create contract"]]

# Check if we have enough data
if len(ml_features) > 10:
    # Normalize the data
    scaler = StandardScaler()
    ml_features_scaled = scaler.fit_transform(ml_features)
    
    # Anomaly detection with Isolation Forest
    st.subheader("Anomaly Detection with Isolation Forest")
    
    # Create and train the model
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(ml_features_scaled)
    
    # Convert predictions to binary (1: normal, -1: anomaly)
    filtered_data["anomaly"] = predictions
    anomalies = filtered_data[filtered_data["anomaly"] == -1]
    
    # Display anomalies
    st.write(f"Detected {len(anomalies)} potential anomalies out of {len(filtered_data)} addresses.")
    
    if not anomalies.empty:
        st.dataframe(anomalies[["Address", "FLAG", "Sent tnx", "Received Tnx", "total Ether sent", "total ether received", "total ether balance", "total transactions (including tnx to create contract", "classification"]])
        
        # Visualize anomalies
        fig = px.scatter(
            filtered_data,
            x="total Ether sent",
            y="total ether received",
            color="anomaly",
            size="total transactions (including tnx to create contract",
            hover_name="Address",
            title="Anomaly Detection Results",
            color_discrete_map={1: "blue", -1: "red"},
            labels={"anomaly": "Status", 1: "Normal", -1: "Anomaly"},
            log_x=True,
            log_y=True
        )
        st.plotly_chart(fig)
    
    # Clustering analysis
    st.subheader("Transaction Pattern Clustering")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered_data["cluster"] = kmeans.fit_predict(ml_features_scaled)
    
    # Visualize clusters
    fig = px.scatter(
        filtered_data,
        x="total Ether sent",
        y="total ether received",
        color="cluster",
        symbol="classification",
        size="total transactions (including tnx to create contract",
        hover_name="Address",
        title="Transaction Pattern Clusters",
        log_x=True,
        log_y=True
    )
    st.plotly_chart(fig)
    
    # Cluster analysis
    st.subheader("Cluster Analysis")
    cluster_stats = filtered_data.groupby("cluster").agg({
        "Sent tnx": "mean",
        "Received Tnx": "mean",
        "total Ether sent": "mean",
        "total ether received": "mean",
        "total ether balance": "mean",
        "total transactions (including tnx to create contract": "mean",
        "FLAG": "sum"
    }).reset_index()
    
    # Format the values
    for col in cluster_stats.columns:
        if col != "cluster" and col != "FLAG":
            cluster_stats[col] = cluster_stats[col].round(2)
    
    st.dataframe(cluster_stats)
    
    # Visualize cluster characteristics
    cluster_stats_melted = pd.melt(
        cluster_stats,
        id_vars=["cluster"],
        value_vars=["Sent tnx", "Received Tnx", "total Ether sent", "total ether received", "total ether balance"],
        var_name="Metric",
        value_name="Value"
    )
    
    fig = px.bar(
        cluster_stats_melted,
        x="cluster",
        y="Value",
        color="Metric",
        barmode="group",
        title="Cluster Characteristics",
        log_y=True
    )
    st.plotly_chart(fig)
else:
    st.info("Not enough data points for meaningful ML analysis. Try adjusting your filters.")

# Flag analysis
st.header("Flag Analysis")
if "FLAG" in filtered_data.columns:
    flag_counts = filtered_data["FLAG"].value_counts().reset_index()
    flag_counts.columns = ["FLAG", "Count"]
    
    # Pie chart for flag distribution
    fig = px.pie(
        flag_counts,
        values="Count",
        names="FLAG",
        title="Flag Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig)
    
    # Compare flagged vs non-flagged transactions
    if 1 in filtered_data["FLAG"].values:
        st.subheader("Flagged vs Non-Flagged Transactions")
        
        # Prepare comparison data
        flag_comparison = filtered_data.groupby("FLAG").agg({
            "Sent tnx": "mean",
            "Received Tnx": "mean",
            "total Ether sent": "mean",
            "total ether received": "mean",
            "total ether balance": "mean",
            "total transactions (including tnx to create contract": "mean"
        }).reset_index()
        
        # Format the values
        for col in flag_comparison.columns:
            if col != "FLAG":
                flag_comparison[col] = flag_comparison[col].round(2)
        
        # Visualize the comparison
        flag_comparison_melted = pd.melt(
            flag_comparison,
            id_vars=["FLAG"],
            value_vars=["Sent tnx", "Received Tnx", "total Ether sent", "total ether received", "total ether balance", "total transactions (including tnx to create contract"],
            var_name="Metric",
            value_name="Value"
        )
        
        fig = px.bar(
            flag_comparison_melted,
            x="Metric",
            y="Value",
            color="FLAG",
            barmode="group",
            title="Flagged vs Non-Flagged Transactions",
            log_y=True
        )
        st.plotly_chart(fig)

# Conclusion
st.header("Conclusion and Recommendations")
st.write("""
Based on the analysis of the Ethereum transaction data, we have identified several patterns and potential anomalies:

1. **Transaction Patterns**: The dashboard shows the distribution of sent vs received transactions, highlighting potential one-way flow addresses.

2. **Anomaly Detection**: Using Isolation Forest, we detected addresses with unusual transaction patterns that may warrant further investigation.

3. **Clustering Analysis**: We identified distinct groups of addresses with similar transaction behaviors, which can help in understanding different types of users in the Ethereum network.

4. **Flag Analysis**: The comparison between flagged and non-flagged addresses reveals significant differences in transaction patterns.

**Recommendations:**
- Investigate addresses classified as anomalies by the Isolation Forest algorithm
- Pay special attention to addresses with extremely high sent-to-received transaction ratios
- Monitor addresses with large negative ether balances relative to their transaction volumes
- Consider developing a more sophisticated flagging system based on the patterns identified in this analysis
""")

# Add a download button for the processed data
st.sidebar.header("Download Data")
csv = filtered_data.to_csv(index=False)
st.sidebar.download_button(
    label="Download Processed Data as CSV",
    data=csv,
    file_name="ethereum_transaction_analysis.csv",
    mime="text/csv"
)

# Add information about the app
st.sidebar.header("About")
st.sidebar.info("""
This dashboard provides comprehensive analysis of Ethereum transaction data. It includes:
- Basic transaction metrics
- Anomaly detection using Isolation Forest
- Transaction pattern clustering
- Flag analysis and comparison

The dashboard is built using Streamlit, Pandas, Scikit-learn, and Plotly.
""")