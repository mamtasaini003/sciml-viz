import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os

st.set_page_config(page_title="FNO Visualization", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { background: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
.block-container { padding-top: 1rem; }
h1, h2, h3 { color: #c9d1d9 !important; }

.hero { text-align: center; padding: 2rem 1rem 1rem; }
.hero h1 { font-size: 2.8rem; color: #58a6ff !important; font-weight: 800; margin-bottom: 5px; }

.comp-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 10px;
    transition: border-color 0.2s;
}
.comp-card:hover { border-color: #58a6ff; }
.comp-card h3 { margin-top: 0; color: #58a6ff !important; font-size: 1.15rem; }
.comp-card p { color: #8b949e; font-size: 0.92rem; line-height: 1.5; margin-bottom: 8px; }

.eq-box {
    background: #21262d; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 18px; margin: 10px 0 15px; font-family: 'Courier New', monospace;
    color: #f0883e; font-size: 1rem; text-align: center;
}
.dim-tag {
    background: rgba(88,166,255,0.15); color: #58a6ff; padding: 3px 10px;
    border-radius: 12px; font-size: 0.8rem; display: inline-block; margin: 2px 4px;
    border: 1px solid rgba(88,166,255,0.3);
}
.op-tag {
    background: rgba(240,136,62,0.15); color: #f0883e; padding: 3px 10px;
    border-radius: 12px; font-size: 0.8rem; display: inline-block; margin: 2px 4px;
    border: 1px solid rgba(240,136,62,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
model_dir = "models"
available = [m for m in os.listdir(model_dir) if m.endswith('.pth')] if os.path.exists(model_dir) else []

with st.sidebar:
    st.markdown("## ⚛️ FNO Visualizer")
    if not available:
        st.error("No models found!")
        st.stop()
    selected = st.selectbox("Model", available, index=available.index('navier_stokes_fno.pth') if 'navier_stokes_fno.pth' in available else 0)
    sd = torch.load(os.path.join(model_dir, selected), map_location='cpu')
    n_params = sum(p.numel() for p in sd.values())
    st.metric("Parameters", f"{n_params:,}")
    st.metric("Layers", len(sd))
    
    st.markdown("---")
    st.markdown("### Table of Contents")
    sections = ["Introduction", "Input Lifting", "Spectral Conv", "Skip Connection", "Residual Add + GELU", "Projection / Decode", "Full Model Map"]
    chosen = st.radio("Navigate to:", sections, label_visibility="collapsed")

# ── Helper: compact heatmap ──
def mini_heatmap(tensor, title="", height=220):
    arr = tensor.numpy()
    if np.iscomplexobj(arr): arr = np.abs(arr)
    orig = list(arr.shape)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    elif arr.ndim == 4:
        a, b, c, d = arr.shape
        arr = arr.reshape(a*c, b*d)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    fig = go.Figure(go.Heatmap(z=arr, colorscale="Plasma",
        hovertemplate="(%{x}, %{y})<br><b>%{z:.5f}</b><extra></extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117", margin=dict(l=0,r=0,t=25,b=0), height=height,
        title=dict(text=f"{title}  <span style='color:#8b949e'>shape {orig}</span>", font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(showticklabels=False, showgrid=False), yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"))
    return fig

# ── HERO ──
st.markdown("""
<div class='hero'>
    <h1>Fourier Neural Operator</h1>
    <p style='color:#8b949e; font-size:1.1rem; max-width:700px; margin:0 auto;'>
        Interactive walkthrough of the FNO architecture for solving PDEs.
        <br>Based on <i>Li et al., "Fourier Neural Operator for Parametric PDEs" (ICLR 2021)</i>
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SECTIONS — each one is a clean, contained card
# ══════════════════════════════════════════════════════════════════════

if chosen == "Introduction":
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### 📖 What is the Fourier Neural Operator?")
    st.markdown("""<p>
    The <b>Fourier Neural Operator (FNO)</b> is a neural network that learns to solve 
    <b>partial differential equations</b> (PDEs) by operating in <b>Fourier (frequency) space</b> 
    rather than physical space.<br><br>
    Traditional neural networks process data point-by-point. The FNO instead transforms the 
    entire input field into its frequency representation using FFT, applies learnable filters 
    to specific frequency modes, and transforms back. This makes it:<br>
    • <b>Resolution-invariant</b> — train on 64×64, evaluate on 256×256<br>
    • <b>Globally aware</b> — every output point "sees" the entire input domain<br>
    • <b>Fast</b> — FFT is O(N log N) vs O(N²) for attention
    </p>""", unsafe_allow_html=True)
    st.markdown("<div class='eq-box'>v<sub>k+1</sub>(x) = σ( W<sub>k</sub> · v<sub>k</sub>(x)  +  𝓕⁻¹( R<sub>k</sub> · 𝓕(v<sub>k</sub>) )(x) )</div>", unsafe_allow_html=True)
    
    # Architecture overview diagram
    fig = go.Figure()
    blocks = [
        (0, "Input\na(x)", "#1f77b4", 1.5), (3, "Lift\n(Linear)", "#ff7f0e", 1.5),
        (6, "Fourier\nLayer ×4", "#2ca02c", 3), (11, "Decode\n(MLP)", "#d62728", 2),
        (15, "Output\nu(x)", "#9467bd", 1.5),
    ]
    for x, label, color, w in blocks:
        fig.add_shape(type="rect", x0=x, x1=x+w, y0=0, y1=2, fillcolor=color, opacity=0.7, line=dict(color="white", width=1))
        fig.add_annotation(x=x+w/2, y=1, text=label, showarrow=False, font=dict(color="white", size=13))
    for i in range(len(blocks)-1):
        x0 = blocks[i][0] + blocks[i][3]
        x1 = blocks[i+1][0]
        fig.add_annotation(x=(x0+x1)/2, y=1, text="→", showarrow=False, font=dict(size=22, color="#58a6ff"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=120, margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(visible=False, range=[-1,17.5]), yaxis=dict(visible=False, range=[-0.5,2.5]))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif chosen == "Input Lifting":
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### 🔼 Input Lifting Layer (P)")
    st.markdown("""<p>
    The raw PDE input <code>a(x)</code> (e.g. initial vorticity or permeability field) has only 
    1–3 channels. The lifting layer is a <b>pointwise linear transform</b> that projects each 
    spatial point into a higher-dimensional feature space of width <code>W</code> (typically 32).
    <br><br>
    Spatial coordinates <code>(x, y)</code> are concatenated to the input, so the linear layer maps 
    from <code>C_in + 2</code> → <code>W</code> channels. This is applied identically at every grid point.
    </p>""", unsafe_allow_html=True)
    st.markdown("<div class='eq-box'>v₀(x) = P · [a(x); x, y] + b &nbsp;&nbsp; where P ∈ ℝ<sup>W × (C+2)</sup></div>", unsafe_allow_html=True)
    st.markdown("<span class='dim-tag'>Input: [B, 64, 64, 3]</span> <span class='op-tag'>Linear (3→32)</span> <span class='dim-tag'>Output: [B, 64, 64, 32]</span>", unsafe_allow_html=True)
    
    # Show the actual weights
    fc0_w = [k for k in sd if k.startswith('fc0') and 'weight' in k]
    fc0_b = [k for k in sd if k.startswith('fc0') and 'bias' in k]
    c1, c2 = st.columns(2)
    if fc0_w:
        c1.plotly_chart(mini_heatmap(sd[fc0_w[0]], f"Weight: {fc0_w[0]}"), use_container_width=True)
    if fc0_b:
        c2.plotly_chart(mini_heatmap(sd[fc0_b[0]], f"Bias: {fc0_b[0]}", height=80), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif chosen == "Spectral Conv":
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### 🌊 Spectral Convolution (Fourier Layer Core)")
    st.markdown("""<p>
    This is the <b>key innovation</b> of the FNO. Instead of a spatial convolution kernel, the 
    network learns <b>complex-valued weights</b> <code>R<sub>k</sub></code> that multiply directly 
    against the Fourier modes of the input.<br><br>
    <b>Step by step:</b><br>
    1. <b>FFT</b> — transform the input field v<sub>k</sub> into frequency space: 𝓕(v<sub>k</sub>)<br>
    2. <b>Truncate</b> — keep only the lowest <code>modes</code> frequencies (ignore high-freq noise)<br>
    3. <b>Multiply</b> — element-wise complex multiply with learned weight tensor R<sub>k</sub><br>
    4. <b>IFFT</b> — transform back to physical space: 𝓕⁻¹(R<sub>k</sub> · 𝓕(v<sub>k</sub>))
    </p>""", unsafe_allow_html=True)
    st.markdown("<div class='eq-box'>𝓕⁻¹( R<sub>k</sub>(ξ) · 𝓕(v<sub>k</sub>)(ξ) )&nbsp;&nbsp; where R<sub>k</sub> ∈ ℂ<sup>W×W×modes₁×modes₂</sup></div>", unsafe_allow_html=True)
    
    # Find spectral conv weights
    conv_keys = sorted([k for k in sd if 'conv' in k and 'weight' in k])
    if conv_keys:
        layer_idx = st.selectbox("Select Fourier Layer", range(len(conv_keys)), format_func=lambda i: conv_keys[i])
        key = conv_keys[layer_idx]
        t = sd[key]
        st.markdown(f"<span class='dim-tag'>Shape: {list(t.shape)}</span> <span class='dim-tag'>Params: {t.numel():,}</span> <span class='op-tag'>Complex-valued</span>", unsafe_allow_html=True)
        
        arr = np.abs(t.numpy())
        # Show as a grid of mode magnitudes
        if arr.ndim == 4:
            c_out, c_in, m1, m2 = arr.shape
            # Show first output channel's view across input channels × modes
            fig = make_subplots(rows=1, cols=min(4, c_out), subplot_titles=[f"Out ch {i}" for i in range(min(4, c_out))])
            for i in range(min(4, c_out)):
                slice_data = arr[i].reshape(c_in, m1*m2)
                fig.add_trace(go.Heatmap(z=slice_data, colorscale="Viridis", showscale=(i==0),
                    hovertemplate="in_ch %{y}, mode %{x}<br><b>|R|=%{z:.5f}</b><extra></extra>"), row=1, col=i+1)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300,
                margin=dict(l=0,r=0,t=30,b=0))
            for ax in fig.layout:
                if ax.startswith('xaxis') or ax.startswith('yaxis'):
                    fig.layout[ax].showticklabels = False
                    fig.layout[ax].showgrid = False
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(mini_heatmap(t, key), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif chosen == "Skip Connection":
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### ⏭️ Skip Connection (1×1 Convolution)")
    st.markdown("""<p>
    In parallel with the spectral path, a simple <b>1×1 pointwise convolution</b> 
    <code>W<sub>k</sub></code> is applied to the input. This acts as a <b>local linear residual</b> 
    that preserves high-frequency spatial information the FFT branch might discard 
    (since we truncate to only <code>modes</code> frequencies).<br><br>
    Think of it as: the spectral branch captures <i>global patterns</i>, while the skip 
    captures <i>local detail</i>. They are added together before activation.
    </p>""", unsafe_allow_html=True)
    st.markdown("<div class='eq-box'>W<sub>k</sub> · v<sub>k</sub>(x) &nbsp;&nbsp; where W<sub>k</sub> ∈ ℝ<sup>W×W×1×1</sup> (pointwise)</div>", unsafe_allow_html=True)
    
    w_keys = sorted([k for k in sd if k.startswith('w') and 'weight' in k])
    w_bias = sorted([k for k in sd if k.startswith('w') and 'bias' in k])
    if w_keys:
        layer_idx = st.selectbox("Select Skip Layer", range(len(w_keys)), format_func=lambda i: w_keys[i])
        c1, c2 = st.columns([3, 1])
        c1.plotly_chart(mini_heatmap(sd[w_keys[layer_idx]], f"Weight: {w_keys[layer_idx]}"), use_container_width=True)
        if layer_idx < len(w_bias):
            c2.plotly_chart(mini_heatmap(sd[w_bias[layer_idx]], f"Bias", height=80), use_container_width=True)
        t = sd[w_keys[layer_idx]]
        st.markdown(f"<span class='dim-tag'>Shape: {list(t.shape)}</span> <span class='dim-tag'>Params: {t.numel():,}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif chosen == "Residual Add + GELU":
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### ➕ Residual Addition & GELU Activation")
    st.markdown("""<p>
    The outputs from the <b>spectral branch</b> (global Fourier filtering) and 
    <b>skip branch</b> (local 1×1 conv) are <b>summed element-wise</b>, then passed 
    through a <b>GELU</b> non-linearity.<br><br>
    This creates the non-linear expressiveness needed to approximate complex PDE solution operators. 
    The process repeats for <b>K layers</b> (typically 4), each refining the solution estimate 
    at progressively higher abstraction levels.
    </p>""", unsafe_allow_html=True)
    st.markdown("<div class='eq-box'>v<sub>k+1</sub> = GELU( 𝓕⁻¹(R<sub>k</sub> · 𝓕(v<sub>k</sub>)) + W<sub>k</sub> · v<sub>k</sub> )</div>", unsafe_allow_html=True)
    
    # Show GELU curve
    x_gelu = np.linspace(-4, 4, 200)
    y_gelu = x_gelu * 0.5 * (1 + np.vectorize(lambda x: float(torch.erf(torch.tensor(x/np.sqrt(2)))))(x_gelu))
    fig = go.Figure(go.Scatter(x=x_gelu, y=y_gelu, mode='lines', line=dict(color='#58a6ff', width=3)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=200,
        margin=dict(l=40,r=10,t=30,b=30), title=dict(text="GELU(x) Activation", font=dict(size=12)),
        xaxis=dict(title=dict(text="x"), showgrid=True, gridcolor="#21262d"),
        yaxis=dict(title=dict(text="GELU(x)"), showgrid=True, gridcolor="#21262d"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the iterative block diagram
    fig2 = go.Figure()
    for i in range(4):
        x0 = i * 4.5
        fig2.add_shape(type="rect", x0=x0, x1=x0+3.5, y0=0, y1=2, fillcolor="#2ca02c", opacity=0.5, line=dict(color="white"))
        fig2.add_annotation(x=x0+1.75, y=1, text=f"Fourier<br>Layer {i}", showarrow=False, font=dict(color="white", size=11))
        if i < 3:
            fig2.add_annotation(x=x0+3.5+0.5, y=1, text="→", showarrow=False, font=dict(size=20, color="#58a6ff"))
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=100, margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(visible=False, range=[-1,18]), yaxis=dict(visible=False, range=[-0.5,2.5]))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif chosen == "Projection / Decode":
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Output Projection (Decode)")
    st.markdown("""<p>
    After K Fourier layers, the feature map v<sub>K</sub> has width <code>W=32</code> channels. 
    The decode stage is a small <b>2-layer MLP</b> applied pointwise at every spatial location to 
    project back down to the desired output dimension (typically 1 for scalar fields like pressure 
    or vorticity).<br><br>
    Layer 1: W → 128 channels (with GELU)<br>
    Layer 2: 128 → C_out channels
    </p>""", unsafe_allow_html=True)
    st.markdown("<div class='eq-box'>u(x) = Q₂ · GELU( Q₁ · v<sub>K</sub>(x) + b₁ ) + b₂ &nbsp;&nbsp; where Q₁ ∈ ℝ<sup>128×W</sup>, Q₂ ∈ ℝ<sup>C_out×128</sup></div>", unsafe_allow_html=True)
    
    fc1_keys = sorted([k for k in sd if k.startswith('fc1')])
    fc2_keys = sorted([k for k in sd if k.startswith('fc2')])
    
    c1, c2 = st.columns(2)
    for k in fc1_keys:
        c1.plotly_chart(mini_heatmap(sd[k], k), use_container_width=True)
    for k in fc2_keys:
        c2.plotly_chart(mini_heatmap(sd[k], k, height=100), use_container_width=True)
    
    st.markdown("<span class='dim-tag'>[B, 64, 64, 32]</span> <span class='op-tag'>→ Linear 32→128</span> <span class='op-tag'>→ GELU</span> <span class='op-tag'>→ Linear 128→1</span> <span class='dim-tag'>[B, 64, 64, 1]</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif chosen == "Full Model Map":
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Full Model — All Parameters at a Glance")
    st.markdown("""<p>
    Every rectangle below is one parameter tensor from the checkpoint. The area of each block 
    is proportional to the number of parameters it contains. Hover to see exact values.
    </p>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Treemap of parameter sizes
    names, parents, values, colors = [], [], [], []
    color_map = {"fc0": "#1f77b4", "conv": "#2ca02c", "w": "#9467bd", "fc1": "#e377c2", "fc2": "#8c564b"}
    
    for key in sd:
        t = sd[key]
        n = t.numel()
        names.append(f"{key}\n{list(t.shape)}")
        parents.append("")
        values.append(n)
        c = "#58a6ff"
        for prefix, col in color_map.items():
            if key.startswith(prefix): c = col; break
        colors.append(c)
    
    fig_tree = go.Figure(go.Treemap(
        labels=names, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(width=2, color="#0d1117")),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Parameters: %{value:,}<extra></extra>",
    ))
    fig_tree.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=500)
    st.plotly_chart(fig_tree, use_container_width=True)
    
    # Compact overview strip
    st.markdown("<div class='comp-card'>", unsafe_allow_html=True)
    st.markdown("### All Weight Matrices (compact)")
    
    sel_key = st.selectbox("Inspect layer:", list(sd.keys()))
    t = sd[sel_key]
    st.markdown(f"<span class='dim-tag'>Shape: {list(t.shape)}</span> <span class='dim-tag'>Params: {t.numel():,}</span> <span class='dim-tag'>dtype: {t.dtype}</span>", unsafe_allow_html=True)
    st.plotly_chart(mini_heatmap(t, sel_key, height=350), use_container_width=True)
    
    mc1, mc2, mc3 = st.columns(3)
    arr_f = t.float()
    mc1.metric("Mean", f"{arr_f.mean():.6f}")
    mc2.metric("Std Dev", f"{arr_f.std():.6f}")
    mc3.metric("L2 Norm", f"{torch.norm(arr_f):.4f}")
    st.markdown("</div>", unsafe_allow_html=True)
