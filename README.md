# SciML Visualizer Hub

A comprehensive Streamlit web platform that serves as a centralized dashboard to visualize and showcase **Scientific Machine Learning (SciML)** projects, neural operators, diffusion models, and physics-informed learning.

## Features Let's Explore!
1. **GitHub Auto-Discovery:** Fetches all public SciML repos using GitHub API.
2. **Model Explorer:** Select deployed `.pth` structural weights and visualize them directly in the browser! Supports Fourier Spectra, 3D operator kernels, layer-wise heatmaps, and weight distributions.
3. **Live Demos:** Run dummy inference pipelines for 1D Burgers', 2D Darcy Flow, and Navier-Stokes with real-time tensor shape simulation.
4. **Paper Gallery:** A collection of seminal papers and theories in SciML.

## Installation & Local Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mamtasaini003/sciml-viz.git
   cd sciml-viz
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Deployment to Streamlit Cloud in 5 Minutes!
1. Fork or push this repository to your GitHub account.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, select `mamtasaini003/sciml-viz`, and choose `app.py` as your Main file path.
4. Add your GitHub Classic Token to Streamlit Secrets (optional, helps avoid API rate limits):
   ```toml
   # In Streamlit Cloud Secrets
   GITHUB_TOKEN = "ghp_your_secret_token_here"
   ```
5. Click **Deploy!**

You're done. Your dazzling, highly interactive dark-themed app uses `.pth` parser out of the box and pulls your direct repositories gracefully.
