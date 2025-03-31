import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer
from datetime import datetime
import plotly.express as px
from sklearn.datasets import make_blobs

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {background-color: #000000;}
    .stButton>button {background-color: #682F2F; color: white;}
    .stSelectbox>div>div>select {background-color: #F3AB60;}
    .stAlert {background-color: #FFE8D6;}
    .elbow-plot {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {transform: scale(1);}
        50% {transform: scale(1.02);}
        100% {transform: scale(1);}
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛍️ Customer Segmentation Analysis")
st.markdown("""
This app performs customer segmentation using clustering techniques on marketing campaign data.
Upload your data file or use the sample data to explore customer segments.
""")

# Sidebar controls
st.sidebar.header("Controls")

# Sample data option
use_sample_data = st.sidebar.checkbox("Use sample data", value=True)

# File uploader
uploaded_file = None
if not use_sample_data:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load data function
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Feature engineering function
@st.cache_data
def engineer_features(data):
    if 'Dt_Customer' in data.columns:
        data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
    
    if 'Year_Birth' in data.columns:
        data["Age"] = datetime.now().year - data["Year_Birth"]
    
    spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", 
                    "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    if all(col in data.columns for col in spending_cols):
        data["Total_Spent"] = data[spending_cols].sum(axis=1)
    
    if 'Marital_Status' in data.columns:
        data["Living_With"] = data["Marital_Status"].replace({
            "Married": "Partner", "Together": "Partner", 
            "Absurd": "Alone", "Widow": "Alone", 
            "YOLO": "Alone", "Divorced": "Alone", 
            "Single": "Alone"
        })
    
    kid_cols = ["Kidhome", "Teenhome"]
    if all(col in data.columns for col in kid_cols):
        data["Children"] = data["Kidhome"] + data["Teenhome"]
        data["Is_Parent"] = np.where(data["Children"] > 0, 1, 0)
    
    if 'Education' in data.columns:
        data["Education"] = data["Education"].replace({
            "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
            "Graduation": "Graduate", "Master": "Postgraduate", 
            "PhD": "Postgraduate"
        })
    
    # Drop unnecessary columns
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", 
               "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop([col for col in to_drop if col in data.columns], axis=1)
    
    return data

def perform_elbow_analysis(scaled_data, k_range):
    """Perform elbow analysis with proper warning handling"""
    model = KMeans(random_state=42, n_init='auto')
    
    visualizer = KElbowVisualizer(
        model, 
        k=(k_range[0], k_range[1]),
        locate_elbow=True,
        timings=False,
        metric='distortion'
    )
    
    try:
        visualizer.fit(scaled_data)
        fig = visualizer.fig
        
        if hasattr(visualizer, 'elbow_value_') and visualizer.elbow_value_ is not None:
            elbow_value = visualizer.elbow_value_
            st.success(f"Suggested number of clusters: {elbow_value}")
            st.info("This is determined by locating the 'elbow' in the distortion curve.")
        else:
            elbow_value = (k_range[0] + k_range[1]) // 2
            st.warning("""
            Could not automatically determine optimal clusters. Possible reasons:
            - Data may not have clear clusters
            - Try different feature combinations
            - Using midpoint of selected range
            """)
        
        return fig, elbow_value
        
    except Exception as e:
        st.error(f"Error in elbow analysis: {str(e)}")
        return None, (k_range[0] + k_range[1]) // 2

def evaluate_clusters(scaled_data, clusters):
    """Calculate multiple cluster validation metrics"""
    try:
        silhouette = silhouette_score(scaled_data, clusters)
        calinski = calinski_harabasz_score(scaled_data, clusters)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Silhouette Score", f"{silhouette:.2f}",
                     help="Higher values indicate better defined clusters (-1 to 1)")
        with col2:
            st.metric("Calinski-Harabasz Index", f"{calinski:.2f}",
                     help="Higher values indicate better clustering")
        
        if silhouette < 0.2:
            st.warning("Low silhouette score - clusters may not be well separated")
        elif silhouette > 0.5:
            st.success("Good silhouette score - clusters are reasonably separated")
    except Exception as e:
        st.error(f"Could not calculate metrics: {str(e)}")

def interpret_clusters(cluster_profiles):
    """Provide automated interpretation of cluster characteristics"""
    st.subheader("Cluster Interpretation")
    
    normalized = cluster_profiles.set_index('Cluster').apply(
        lambda x: (x - x.mean()) / x.std(), axis=0)
    
    interpretations = []
    for cluster in normalized.index:
        top_features = normalized.loc[cluster].abs().nlargest(2).index.tolist()
        direction = []
        for feat in top_features:
            if normalized.loc[cluster, feat] > 0:
                direction.append(f"higher than average {feat}")
            else:
                direction.append(f"lower than average {feat}")
        interpretations.append(
            f"Cluster {cluster}: Characterized by {', '.join(direction)}"
        )
    
    st.write("Key characteristics:")
    for interp in interpretations:
        st.write(f"- {interp}")

# Main app logic
def main():
    # Load data
    if use_sample_data:
        try:
            # Generate better sample data with clear clusters
            X, y = make_blobs(
                n_samples=300,
                centers=4,
                n_features=5,
                cluster_std=0.8,
                random_state=42
            )
            
            # Create DataFrame with meaningful cluster-related features
            data = pd.DataFrame(X, columns=[
                'Spending_Power', 
                'Brand_Loyalty', 
                'Digital_Engagement',
                'Store_Activity',
                'Promo_Responsiveness'
            ])
            
            # Add derived features that correlate with clusters
            data['Income'] = (data['Spending_Power'] * 15000 + 40000).abs()
            data['Age'] = (data['Brand_Loyalty'] * 8 + 35).abs().astype(int)
            data['Total_Spent'] = (data['Spending_Power'] * 500 + 1000).abs()
            data['Children'] = (data['Store_Activity'] * 0.5 + 1).abs().astype(int)
            data['Is_Parent'] = np.where(data['Children'] > 0, 1, 0)
            data['Recency'] = (data['Promo_Responsiveness'] * 15 + 30).abs().astype(int)
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            st.stop()
    else:
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if data is None:
                st.stop()
            data = engineer_features(data)
        else:
            st.warning("Please upload a file or use sample data")
            st.stop()

    # Show raw data
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(data)

    # Data Exploration Section
    st.header("🔍 Data Exploration")
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Pairplot visualization
    st.subheader("Pairplot of Selected Features")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    # Handle default features safely
    default_pairplot_features = ["Income", "Total_Spent", "Age", "Children"]
    available_features = [f for f in default_pairplot_features if f in numeric_cols]
    
    # Fallback logic with validation
    if not available_features:
        available_features = numeric_cols[:min(4, len(numeric_cols))]
    if not available_features:  # If still empty after fallback
        st.error("No numeric columns available for pairplot visualization.")
        st.stop()

    selected_features = st.multiselect(
        "Select features for pairplot",
        options=numeric_cols,
        default=available_features
    )

    if selected_features:
        hue_col = "Is_Parent" if "Is_Parent" in data.columns else None
        fig = sns.pairplot(data[selected_features + ([hue_col] if hue_col else [])], 
                          hue=hue_col, palette=["#682F2F", "#F3AB60"])
        st.pyplot(fig)

    # Clustering Section
    st.header("📊 Customer Segmentation")
    st.subheader("Feature Selection")
    
    # Handle default cluster features safely
    default_cluster_features = ["Income", "Total_Spent", "Age", "Children"]
    available_cluster_features = [f for f in default_cluster_features if f in numeric_cols]
    
    # Fallback logic with validation
    if not available_cluster_features:
        available_cluster_features = numeric_cols[:min(4, len(numeric_cols))]
    if not available_cluster_features:  # If still empty after fallback
        st.error("No numeric columns available for clustering.")
        st.stop()

    cluster_features = st.multiselect(
        "Select features for clustering",
        options=numeric_cols,
        default=available_cluster_features
    )

    if not cluster_features:
        st.warning("Please select at least one feature for clustering")
        st.stop()

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[cluster_features])

    # Determine optimal clusters
    st.subheader("Determine Optimal Number of Clusters")
    k_range = st.slider("Select range for k", 2, 10, (2, 8))
    
    elbow_fig, elbow_value = perform_elbow_analysis(scaled_data, k_range)
    if elbow_fig is not None:
        with st.container():
            st.markdown('<div class="elbow-plot">', unsafe_allow_html=True)
            st.pyplot(elbow_fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # Perform clustering
    st.subheader("Perform Clustering")
    n_clusters = st.slider("Select number of clusters", 
                          min_value=2, max_value=10, 
                          value=elbow_value)

    algorithm = st.radio("Select clustering algorithm", 
                        ["K-Means", "Agglomerative"], index=0)

    if algorithm == "K-Means":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    
    clusters = clusterer.fit_predict(scaled_data)
    data["Cluster"] = clusters

    # Cluster validation metrics
    st.subheader("Cluster Validation Metrics")
    evaluate_clusters(scaled_data, clusters)

    # Visualize clusters
    st.subheader("Cluster Visualization")
    dim_red_method = st.radio("Dimensionality reduction method",
                             ["PCA", "t-SNE"], index=0)

    if dim_red_method == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    
    reduced_data = reducer.fit_transform(scaled_data)
    data["Dim1"] = reduced_data[:, 0]
    data["Dim2"] = reduced_data[:, 1]

    fig = px.scatter(
        data, x="Dim1", y="Dim2", color="Cluster",
        hover_data=cluster_features,
        title=f"Customer Clusters ({dim_red_method} Projection)",
        width=800, height=600,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    if algorithm == "K-Means" and dim_red_method == "PCA":
        centers = reducer.transform(clusterer.cluster_centers_)
        fig.add_trace(px.scatter(
            x=centers[:, 0], y=centers[:, 1],
            text=[f"Center {i}" for i in range(n_clusters)],
            size=[10]*n_clusters
        ).data[0])
        fig.update_traces(marker=dict(symbol='x', size=12, line=dict(width=2)))

    st.plotly_chart(fig)

    # Cluster profiles
    st.subheader("Cluster Profiles")
    cluster_profiles = data.groupby("Cluster")[cluster_features].mean().reset_index()
    st.dataframe(cluster_profiles.style.background_gradient(cmap="YlOrBr"))

    # Cluster interpretation
    interpret_clusters(cluster_profiles)

    # Download results
    st.subheader("Download Results")
    if st.button("Download Cluster Data"):
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This customer segmentation dashboard was created using:
    - Python 🐍
    - Streamlit 🎈
    - scikit-learn 🤖
    - Plotly 📊
    """)

if __name__ == "__main__":
    main()