import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(page_title="SciML Interactive Architecture", layout="wide", initial_sidebar_state="collapsed")

# Custom UI for 3D/Visualization aesthetics
st.markdown("""
<style>
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Inter', sans-serif;
}
.header-box { 
    text-align: center; 
    padding: 3rem 1rem; 
    background: linear-gradient(180deg, rgba(88,166,255,0.1) 0%, rgba(13,17,23,1) 100%);
    border-bottom: 1px solid #30363d; 
    margin-bottom: 30px;
}
.header-box h1 { font-size: 3.5rem; color: #58a6ff; font-weight: 800; margin-bottom: 0px;}
.layer-box { 
    background: #161b22; 
    border-radius: 12px; 
    padding: 30px; 
    margin: 40px auto; 
    border: 1px solid #30363d;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    max-width: 1200px;
}
.layer-box h2 {
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 15px;
}
.layer-desc {
    color: #8b949e;
    font-size: 1.1rem;
    line-height: 1.6;
    margin-bottom: 25px;
}
.stat-tag {
    background: #21262d; 
    border: 1px solid #30363d; 
    padding: 5px 12px; 
    border-radius: 20px; 
    font-size: 0.85rem; 
    color: #c9d1d9;
    display: inline-block;
    margin-right: 10px;
    margin-bottom: 15px;
}
/* Allow containers to span fully */
.block-container {
    padding-top: 0px;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION --- 
st.markdown("""
<div class='header-box'>
    <h1>⚛️ Operator Visualization Explainer</h1>
    <p style="font-size: 1.2rem; color: #8b949e; max-width: 800px; margin: 15px auto;">
        An interactive, pixel-perfect anatomical walkthrough of the chosen Neural Operator architecture. 
        Inspired by <i>bbycroft.net</i>, this visualizer unfolds the actual weights directly from GitHub into a 
        linear, scrollable topology. Hover over individual pixels to reveal precise float values deployed inside the matrix.
    </p>
</div>
""", unsafe_allow_html=True)

# --- MODEL SELECTION ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    model_dir = "models"
    available_models = [m for m in os.listdir(model_dir) if m.endswith('.pth')] if os.path.exists(model_dir) else []
    
    if not available_models:
        st.error("No models found in the deployment directory!")
        st.stop()
        
    selected_model = st.selectbox("📂 Select a deployed model weight to parse:", available_models, index=available_models.index('navier_stokes_fno.pth') if 'navier_stokes_fno.pth' in available_models else 0)

model_path = os.path.join(model_dir, selected_model)
try:
    state_dict = torch.load(model_path, map_location='cpu')
except Exception as e:
    st.error(f"Failed to load weights: {e}")
    st.stop()

param_count = sum(p.numel() for p in state_dict.values())
st.markdown(f"<p style='text-align: center; color: #8b949e;'>Loaded <b>{selected_model}</b> successfully (Total Connectome: {param_count:,} Parameters)</p>", unsafe_allow_html=True)

# --- RENDER ENGINE ---
def render_matrix_heatmap(tensor, tensor_name):
    """
    Decodes highly dimensional neural arrays into flattened 2D visually appealing 
    heatmaps mapped exactly to their underlying real numerical values.
    """
    tensor_np = tensor.numpy()
    is_complex = False
    
    if np.iscomplexobj(tensor_np):
        tensor_np = np.abs(tensor_np) # Calculate magnitude for frequency filters
        is_complex = True
        
    original_shape = list(tensor_np.shape)
    
    if tensor_np.ndim == 1:
        tensor_np = tensor_np.reshape(1, -1)
        plot_title = "1D Vector Array (Bias / Normalization)"
        
    elif tensor_np.ndim == 2:
        plot_title = "2D Dense Weight Matrix"
        
    elif tensor_np.ndim == 4:
        # Complex FNO layers shape usually [out_c, in_c, modes1, modes2]
        c_out, c_in, m1, m2 = tensor_np.shape
        tensor_np = tensor_np.reshape(c_out * m1, c_in * m2)
        plot_title = "4D Convolutional Kernel (Flattened internally for visualization)"
        
    else:
        # Squeeze down any other shape safely
        try:
            tensor_np = tensor_np.reshape(-1, tensor_np.shape[-1])
            plot_title = "Multi-dimensional Tensor (Flattened)"
        except:
             tensor_np = tensor_np.flatten().reshape(1, -1)
             plot_title = "1D Flattened Array"

    fig = px.imshow(
        tensor_np, 
        color_continuous_scale="Plasma",
    )
    
    # Intricate Hover Tooltips showing what layman wants to know
    custom_hover = (
        "<b>Visual Pixel Inspected:</b><br>"
        "Grid Position (Y, X): (%{y}, %{x})<br>"
        "<b>Mathematical Parameter Value:</b> %{z:.6f}<br>"
    )
    if is_complex:
        custom_hover += "<i>(Value displayed is Complex Magnitude |x|)</i><br>"
        
    custom_hover += "<extra></extra>"
    
    fig.update_traces(hovertemplate=custom_hover)
    
    fig.update_layout(
        template="plotly_dark", 
        plot_bgcolor="#161b22", 
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=400 if tensor_np.shape[0] > 10 else 150 # Make biases slim
    )
    
    return fig, original_shape

# --- NETWORK TOPOLOGY MAPPER ---
# We define standard layman explanations dynamically matching module names

network_blueprint = []
keys = list(state_dict.keys())
joined_keys = "".join(keys).lower()

if "fc0" in joined_keys or "conv0" in joined_keys:
    # Detected FNO
    network_blueprint = [
        {"name": "1. The Featurization (Lift) Layer", "prefix": "fc0", 
         "desc": "Before performing continuous operator learning, the PDE's simple raw coordinates (x, y, time) are mathematically broadcasted up into a highly dense feature map. Every pixel mapped here is a parameter responsible for multiplying your raw grid into the embedding space."},
        {"name": "2. High-Frequency Spectral Convolution 0", "prefix": ["conv0", "w0"], 
         "desc": "The beating heart of the FNO. The `conv` portions denote Complex Numbers inside the Fourier Function block (`spectral_conv` weights multiply the waves by amplitude). The `w0` dictates the 1x1 Convolution skip-connection which adds standard deep-learning residuals back into the signal!"},
        {"name": "3. High-Frequency Spectral Convolution 1", "prefix": ["conv1", "w1"], 
         "desc": "Just like the layer before, passing through the Fast Fourier Transform (FFT) again to glean higher-level patterns globally rather than just close neighbors."},
        {"name": "4. Decoder Collapse Layer 1", "prefix": "fc1", 
         "desc": "We step out of the Operator domain. This massive fully-connected linear matrix translates hundreds of dimensional modes back into readable Euclidean spaces."},
        {"name": "5. Final Output Predictor", "prefix": "fc2", 
         "desc": "The exit gate. The last parameters multiply the processed data aggressively down into exactly 1 final variable per coordinate (for instance, exactly what the vorticity flow is at coordinate [X=10, Y=50])."}
    ]
elif "branch" in joined_keys:
    # Detected DeepONet
    network_blueprint = [
        {"name": "1. Branch Network Extractors", "prefix": "branch", 
         "desc": "The branch learns functions (the 'u' field input parameters). Watch how the weights transition as they analyze the initial environmental conditions."},
        {"name": "2. Trunk Network Coordinate Map", "prefix": "trunk", 
         "desc": "Simultaneously, the trunk learns the geometry of the PDE output query positions (the 'y' locations)."},
        {"name": "3. Output Combiner", "prefix": "output", 
         "desc": "The fusion mechanism that performs the ultimate inner dot-product between Branch and Trunk states."}
    ]
else:
    # Generic MLP / Unknown
    network_blueprint = [
        {"name": "Hidden Layers", "prefix": "", 
         "desc": "A fully connected pathway tracing layer-by-layer topologies."}
    ]


st.markdown("<h2 style='text-align: center; margin-top: 50px; color: #8b949e;'>↓ Scroll to trace the Forward Pass ↓</h2>", unsafe_allow_html=True)

# Parse through the blueprint and render!
for block in network_blueprint:
    # Allow prefix lists or single strings
    prefixes = block["prefix"] if isinstance(block["prefix"], list) else [block["prefix"]]
    
    # Find matching params
    matching_keys = []
    for p in prefixes:
        matching_keys.extend([k for k in keys if k.startswith(p)])
        
    if not matching_keys:
         # Skip if missing in some models
         if p == "": # Catch all generic
             matching_keys = keys
         else:
             continue
    
    st.markdown("<div class='layer-box'>", unsafe_allow_html=True)
    st.markdown(f"<h2>{block['name']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='layer-desc'>{block['desc']}</div>", unsafe_allow_html=True)
    
    for mk in matching_keys:
        tensor = state_dict[mk]
        param_vol = tensor.numel()
        
        # Draw Graphic
        fig, original_shape = render_matrix_heatmap(tensor, mk)
        
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown(f"**Target Array:**<br>`{mk}`", unsafe_allow_html=True)
            st.markdown(f"<span class='stat-tag'><b>Shape:</b> <span>{original_shape}</span></span>", unsafe_allow_html=True)
            st.markdown(f"<span class='stat-tag'><b>Parameters:</b> <span>{param_vol:,}</span></span>", unsafe_allow_html=True)
            st.markdown(f"<span class='stat-tag'><b>L1 Norm:</b> <span>{torch.norm(tensor.float(), p=1):.2f}</span></span>", unsafe_allow_html=True)
        
        with c2:
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("<hr style='border: 1px solid #30363d;'>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; margin-bottom: 50px;'>Output Field Generated! 🏁</h3>", unsafe_allow_html=True)
