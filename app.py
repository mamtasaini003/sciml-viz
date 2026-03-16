import streamlit as st
from utils import fetch_mamta_repos, load_css

st.set_page_config(
    page_title="Mamta Saini | SciML Projects",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
load_css()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103259.png", width=80) 
    st.markdown("## Mamta Saini")
    st.markdown("Scientific Machine Learning Researcher based in Bengaluru.")
    st.markdown("[Google Scholar](#) | [GitHub](https://github.com/mamtasaini) | [LinkedIn](#)")
    st.markdown("---")
    st.markdown("### ⚛️ Neural Operators")
    st.markdown("### 🌊 Fluid Dynamics")
    st.markdown("### 🧪 Physics-Informed ML")

# Main Content
st.markdown("<h1 class='main-title'>Welcome to SciML Visualizer Hub</h1>", unsafe_allow_html=True)
st.markdown("Explore state-of-the-art physics-informed deep learning models, neural operators, and diffusion implementations. Automatically synced from [Mamta Saini's GitHub](https://github.com/mamtasaini).")
st.markdown("---")

st.markdown("## 📚 Project Overview")

# Fetch projects
with st.spinner("Fetching SciML repositories from GitHub..."):
    repos = fetch_mamta_repos()

if not repos:
    st.info("No SciML repositories found. Check GitHub API connection.")
else:
    # 3-column responsive grid
    cols = st.columns(3)
    for i, repo in enumerate(repos):
        col = cols[i % 3]
        with col:
            # Create tags HTML
            tags_html = "".join([f"<span class='tag'>{opt}</span>" for opt in repo['topics'][:3]])
            if not tags_html:
                tags_html = f"<span class='tag'>{repo['language']}</span>"
                
            card_html = f"""
            <div class="metric-card">
                <h3>{repo['name']}</h3>
                {tags_html}
                <br><br>
                <p>{repo['description'] if repo['description'] else 'No description available for this repository.'}</p>
                <p>⭐ {repo['stars']} | 📅 Updated: {repo['last_updated']}</p>
                <a class="custom-link" href="{repo['url']}" target="_blank">View on GitHub →</a>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Instructions")
st.markdown("""
- **Model Explorer:** Navigate to the `Model Explorer` page using the left sidebar. Select deployed `.pth` files and visualize Fourier modes, kernels, and network weights.
- **Live Demos:** Run inference interactively by generating initial conditions and predicting PDE solutions via the `Live Demos` page.
- **Paper Gallery:** Read connected research papers in the `Paper Gallery`.
""")
