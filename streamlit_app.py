import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Sample data: Replace this with your actual prompt embeddings
prompt_embeddings = np.random.rand(100, 10)  # 100 prompts with 10-dimensional embeddings
prompt_list = [f"Prompt {i}" for i in range(100)]  # Sample prompt list

# Clustering the embeddings
num_clusters = 5  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(prompt_embeddings)
clusters = kmeans.labels_

# Creating a dictionary to hold prompts by cluster
clustered_prompts = {i: [] for i in range(num_clusters)}
for idx, cluster in enumerate(clusters):
    clustered_prompts[cluster].append(prompt_list[idx])

# PCA for 2D and 3D visualization
pca = PCA(n_components=3)
pca_result = pca.fit_transform(prompt_embeddings)

# Streamlit app layout
st.sidebar.title("Navigation")
view = st.sidebar.radio("Select a view:", ["3D Visualization", "2D Visualization", "Prompt Browser"])

if view == "3D Visualization":
    # 3D Visualization
    fig_3d = px.scatter_3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        color=clusters,
        labels={'color': 'Cluster'},
        title='3D Cluster Visualization',
        hover_name=prompt_list
    )
    st.plotly_chart(fig_3d)

elif view == "2D Visualization":
    # 2D Visualization for selection
    pca_2d = PCA(n_components=2)
    pca_result_2d = pca_2d.fit_transform(prompt_embeddings)

    # 2D Scatter Plot
    fig_2d = px.scatter(
        x=pca_result_2d[:, 0],
        y=pca_result_2d[:, 1],
        color=clusters,
        labels={'color': 'Cluster'},
        title='2D Cluster Selection',
        hover_name=prompt_list
    )
    st.plotly_chart(fig_2d)

elif view == "Prompt Browser":
    # Streamlit app layout for browsing prompts
    st.title("Prompt Browser")
    selected_cluster = st.selectbox("Select a category:", list(clustered_prompts.keys()))

    # Display prompts in the selected cluster
    st.subheader(f"Prompts in Category {selected_cluster}:")
    for prompt in clustered_prompts[selected_cluster]:
        st.write(prompt)
