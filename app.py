import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----- CACHING THE MODEL -----
@st.cache_resource(show_spinner=False)
def load_model(model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
    """Load and return the SentenceTransformer model (cached)."""
    return SentenceTransformer(model_name)

# ----- AUXILIARY FUNCTIONS -----
def truncate(text, limit):
    """Keep only the first `limit` words, adding '…' if truncated."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "…"

@st.cache_data(show_spinner=False)
def compute_embeddings(items, model_name):
    """Return embeddings for a list of items, using a cached model."""
    model = load_model(model_name)
    return model.encode(items)

def compute_similarity_matrix(embeddings):
    """Return a cosine similarity matrix for the embeddings."""
    return cosine_similarity(embeddings)

def plot_heatmap(sim_mat, items, word_limit):
    """Plot a heatmap (with Ward linkage ordering) and return the Matplotlib figure."""
    dist = 1 - sim_mat
    linked = linkage(dist, method='ward')
    dendro = dendrogram(linked, no_plot=True)
    order = dendro['leaves']

    ordered_mat = sim_mat[np.ix_(order, order)]
    ordered_items = [items[i] for i in order]
    labels = [truncate(it, word_limit) for it in ordered_items]

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.imshow(ordered_mat, vmin=0, vmax=1)
    n = len(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{ordered_mat[i, j]:.2f}",
                    ha='center', va='center', fontsize=6)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(cax, ax=ax, shrink=0.7)
    plt.title("Similarity Heatmap (Ward Linkage)")
    plt.tight_layout()
    return fig

def plot_dendrogram(sim_mat, items, word_limit):
    """Return a Matplotlib figure of a hierarchical clustering dendrogram using Ward linkage."""
    dist = 1 - sim_mat
    linked = linkage(dist, method='ward')
    labels = [truncate(it, word_limit) for it in items]

    fig, ax = plt.subplots(figsize=(8, 8))
    dendrogram(linked, labels=labels, orientation='right', leaf_font_size=7)
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    plt.tight_layout()
    return fig

def plot_mds(sim_mat, items, word_limit):
    """Return a Matplotlib figure of an MDS projection."""
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(1 - sim_mat)
    labels = [truncate(it, word_limit) for it in items]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=40)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (coords[i, 0], coords[i, 1]), fontsize=7)
    plt.title("MDS Projection")
    plt.tight_layout()
    return fig

# ----- STREAMLIT LAYOUT -----
st.set_page_config(page_title="Similarity Explorer", layout="wide")
st.title("📊 Similarity Explorer with Sentence Transformers")

# Sidebar: parameter inputs
with st.sidebar:
    st.header("Settings")
    WORD_LIMIT = st.slider(
        "Number of words to display in plot labels", min_value=1, max_value=10, value=4
    )
    run_button = st.button("▶️ Run Analysis")

# Main: text area for user to paste items
st.subheader("Input Items")
st.write("Paste your items below, one per line (e.g., a list of abstracts).")
text_input = st.text_area("Paste items here", height=200)

if run_button:
    # Process pasted text into a list of items
    raw_lines = [line.strip() for line in text_input.split("\n") if line.strip()]
    if len(raw_lines) < 2:
        st.error("Please paste at least 2 non-empty lines.")
        st.stop()
    items = raw_lines

    # Compute embeddings (cached)
    with st.spinner("Computing embeddings…"):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = compute_embeddings(items, model_name)

    # Compute similarity matrix
    sim_mat = compute_similarity_matrix(embeddings)

    # Visualizations
    st.subheader("Visualizations")

    # Heatmap
    fig_heatmap = plot_heatmap(sim_mat, items, WORD_LIMIT)
    st.pyplot(fig_heatmap)

    # Dendrogram
    fig_dend = plot_dendrogram(sim_mat, items, WORD_LIMIT)
    st.pyplot(fig_dend)

    # MDS
    fig_mds = plot_mds(sim_mat, items, WORD_LIMIT)
    st.pyplot(fig_mds)

    # Download embeddings as CSV
    st.subheader("Download Embeddings")
    df_emb = pd.DataFrame(embeddings, index=items)
    csv_bytes = df_emb.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download embeddings as CSV",
        data=csv_bytes,
        file_name="embeddings.csv",
        mime="text/csv"
    )
