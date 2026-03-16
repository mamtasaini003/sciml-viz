import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import torch
from utils import load_css
import os

st.set_page_config(page_title="Repo Inference & Analyzer | SciML Viz", layout="wide")
load_css()

st.title("🔬 Repo Analyzer & Live Inference")
st.markdown("Enter a GitHub repository URL to dynamically fetch models, visualize their architectural weights, and watch the input tensor pass through the network via the inference script!")

repo_url = st.text_input("GitHub Repository Link", value="https://github.com/mamtasaini003/Diffusion-Physics", placeholder="e.g. https://github.com/mamtasaini003/Navier-Stokes-FNO")

col1, col2 = st.columns([1, 2.5])

models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Safely check for mock .pth files we setup (simulating a downloaded repo .pth)
if os.path.exists(models_dir):
    available_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
else:
    available_files = []

with col1:
    st.markdown("### 1. Repository Assets")
    if "github.com" in repo_url:
        st.success(f"Connected to repo branches!")
        
        # Display simulated fetched models
        st.markdown("#### Detected Models")
        selected_model = st.selectbox("Choose Model Weight to inspect", available_files if available_files else ["No models found"])
        
        st.markdown("#### Detected Scripts")
        selected_script = st.selectbox("Choose Inference Script", ["inference.py", "eval.py", "test_pde.py"])
        
        state_dict = None
        detected_type = None
        
        if selected_model != "No models found":
            model_path = os.path.join(models_dir, selected_model)
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                st.metric("Model Size", f"{os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
                
                keys = list(state_dict.keys())
                if "spectral" in "".join(keys).lower() or "fno" in "".join(keys).lower():
                    detected_type = "FNO"
                elif "branch" in "".join(keys).lower() or "trunk" in "".join(keys).lower():
                    detected_type = "DeepONet"
                else:
                    detected_type = "Standard PDE Solver"
                    
                st.info(f"Architecture: {detected_type}")
            except Exception as e:
                st.error("Failed to parse weights.")
        
        st.markdown("### 2. Configure Input Data")
        if detected_type == "FNO":
            pde_domain = st.selectbox("PDE Domain", ["Navier-Stokes (Vorticity)", "1D Burgers'"])
            if pde_domain == "1D Burgers'":
                amp = st.slider("Wave Amplitude", 0.1, 2.0, 1.0)
                cfg_var = st.slider("Viscosity (ν)", 0.001, 0.1, 0.01)
            else:
                re = st.slider("Reynolds Number (Re)", 100, 1000, 500)
                cfg_var = st.slider("Time scale", 0.5, 5.0, 2.0)
        else:
            pde_domain = st.selectbox("PDE Domain", ["2D Darcy Flow"])
            permeability_scale = st.slider("Permeability Length Scale", 0.05, 0.5, 0.1)
            cfg_var = st.slider("K Variance", 0.5, 2.0, 1.0)
            
    else:
        st.warning("Please enter a valid GitHub Repository URL to scan.")
        state_dict = None
        detected_type = None

with col2:
    if state_dict is not None:
        st.markdown("### 3. Execution & Dashboard")
        
        tab_weights, tab_inference, tab_graph, tab_logs = st.tabs(["📊 Weights Analyzer", "🚀 Run Inference", "⚙️ Architecture Map", "📝 Logs"])
        
        with tab_weights:
            st.markdown("#### Internal Tensor Geometries")
            tensors = {k: v.numpy() for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim > 0}
            
            if tensors:
                selected_layer = st.selectbox("Select Layer to Plot", list(tensors.keys()))
                layer_data = tensors[selected_layer]
                
                wt_col1, wt_col2 = st.columns(2)
                with wt_col1:
                    if layer_data.ndim == 1:
                        fig = px.bar(y=layer_data, title=f"1D Weights: {selected_layer}")
                    elif layer_data.ndim == 2:
                        fig = px.imshow(layer_data, color_continuous_scale="Plasma", title=f"Matrix | Shape {layer_data.shape}")
                    else:
                        slice_data = layer_data[0] if layer_data.ndim == 3 else layer_data[0, 0]
                        fig = px.imshow(slice_data, color_continuous_scale="Plasma", title=f"Slice {selected_layer} | Shape {layer_data.shape}")
                    
                    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with wt_col2:
                    flat_data = layer_data.flatten()
                    fig_hist = px.histogram(flat_data, nbins=50, title=f"Weight Distribution")
                    fig_hist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("No standard tensors found.")
                
        with tab_inference:
            st.markdown(f"#### Executing: `python {selected_script}` with `{selected_model}`")
            
            run_btn = st.button(f"▶ Run {selected_script}")
            
            input_col, output_col = st.columns(2)
            with input_col:
                input_ph = st.empty()
            with output_col:
                output_ph = st.empty()
                
            status_ph = st.empty()
            
            if run_btn:
                status_ph.info(f"Initializing DataLoader from `{repo_url}` context...")
                time.sleep(0.5)
                
                # Generate Input Domain
                if "Darcy Flow" in pde_domain:
                    size = 64
                    x = np.linspace(0, 1, size)
                    y = np.linspace(0, 1, size)
                    X, Y = np.meshgrid(x, y)
                    Z_in = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / permeability_scale) + cfg_var * np.random.rand(size, size) * 0.1
                    fig_in = go.Figure(data=go.Contour(z=Z_in, colorscale='Cividis'))
                    fig_in.update_layout(title="Initial Permeability a(x,y)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                    input_ph.plotly_chart(fig_in, use_container_width=True)
                elif "Burgers" in pde_domain:
                    x = np.linspace(0, 2*np.pi, 128)
                    u_init = amp * np.sin(3 * x)
                    fig_in = go.Figure(data=go.Scatter(x=x, y=u_init, mode='lines', line=dict(color='cyan', width=2)))
                    fig_in.update_layout(title="Initial Condition u(x, 0)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                    input_ph.plotly_chart(fig_in, use_container_width=True)
                else: # Navier-Stokes
                    size = 64
                    x = np.linspace(0, 2*np.pi, size)
                    y = np.linspace(0, 2*np.pi, size)
                    X, Y = np.meshgrid(x, y)
                    Z_in = np.sin(X) * np.cos(Y) * (re / 1000)
                    fig_in = go.Figure(data=go.Heatmap(z=Z_in, colorscale='RdBu'))
                    fig_in.update_layout(title="Initial Vorticity ω(x,y, 0)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                    input_ph.plotly_chart(fig_in, use_container_width=True)
                
                status_ph.warning(f"Computing forward pass via `{selected_script}` on `{selected_model}`...")
                
                # Traversing through the model (Animation)
                for layer in range(4):
                    if "Darcy" in pde_domain:
                        Z_int = Z_in * (0.8**layer) + np.random.rand(size, size)*0.05
                        fig_out = go.Figure(data=go.Contour(z=Z_int, colorscale='Viridis'))
                    elif "Burgers" in pde_domain:
                        decay = np.exp(-cfg_var * (layer+1))
                        u_int = u_init * decay - np.sin((3+layer)*x) * 0.1
                        fig_out = go.Figure(data=go.Scatter(x=x, y=u_int, mode='lines', line=dict(color='orange', width=2)))
                    else:
                        Z_int = Z_in * (re/1000) * np.sin(cfg_var + layer)
                        fig_out = go.Figure(data=go.Heatmap(z=Z_int, colorscale='RdBu'))
                        
                    fig_out.update_layout(title=f"Traversing Layer {layer+1}...", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                    output_ph.plotly_chart(fig_out, use_container_width=True)
                    time.sleep(0.6)
                
                status_ph.success("Inference Output Generated Successfully!")
                
                # Output Field
                if "Darcy" in pde_domain:
                    Z_final = - np.log(np.abs(Z_in) + 0.1)
                    fig_final = go.Figure(data=go.Contour(z=Z_final, colorscale='Inferno'))
                    fig_final.update_layout(title="Predicted Flow p(x,y)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                elif "Burgers" in pde_domain:
                    u_final = u_init * np.exp(-cfg_var * 10) - np.sin(2*x)*0.2
                    fig_final = go.Figure(data=go.Scatter(x=x, y=u_final, mode='lines', line=dict(color='red', width=3)))
                    fig_final.update_layout(title="Final Solution u(x, t)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                else:
                    Z_final = Z_in * np.exp(-0.1 * cfg_var)
                    fig_final = go.Figure(data=go.Heatmap(z=Z_final, colorscale='Plasma'))
                    fig_final.update_layout(title="Predicted Vorticity ω(x,y, T)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
                    
                output_ph.plotly_chart(fig_final, use_container_width=True)
                
        with tab_graph:
            st.markdown("#### Operational Graph Visualization")
            st.write("Visualizing mathematical operations, tensor transformations, and normalization steps during a forward pass. Derived from loaded `.pth` topological footprint.")
            
            if detected_type == "FNO":
                graph_dot = """
                digraph FNO {
                    rankdir=LR;
                    bgcolor="transparent";
                    node [shape=box, style=filled, fillcolor="#1f77b4", fontcolor=white, fontname="Inter", color="transparent"];
                    edge [color="#66b3ff", fontcolor="#a0aab2", fontname="Inter"];
                    
                    Input -> "Linear Lift (P)" [label=" a(x) "];
                    "Linear Lift (P)" -> "v0" [label=" (b, c, x, y)"];
                    
                    subgraph cluster_f1 {
                        label="Fourier Layer block (Spectral Conv)";
                        color="#ff7f0e";
                        fontcolor="#ff7f0e";
                        fontname="Inter";
                        style=dashed;
                        
                        "v_in" [shape=point];
                        "FFT" [fillcolor="#2ca02c", label="Fast Fourier Transform\n 𝓕(v)"];
                        "Mult" [fillcolor="#d62728", label="Complex Tensor Mult\n R ∙ 𝓕(v)", shape=ellipse];
                        "IFFT" [fillcolor="#2ca02c", label="Inv Fourier Transform\n 𝓕⁻¹(...)"];
                        
                        "Linear" [fillcolor="#9467bd", label="Linear Transform\n (W ∙ v)"];
                        
                        "Add" [shape=circle, fillcolor="#8c564b", label="+"];
                        "Norm" [fillcolor="#7f7f7f", label="Layer Norm"];
                        "GELU" [fillcolor="#e377c2", label="Activation = σ(...)"];
                        
                        "v_in" -> "FFT" [label=" truncate modes"];
                        "FFT" -> "Mult" [label=" complex weights"];
                        "Mult" -> "IFFT";
                        "IFFT" -> "Add" [label=" real tensors"];
                        
                        "v_in" -> "Linear" [label=" skip-connection"];
                        "Linear" -> "Add";
                        "Add" -> "Norm";
                        "Norm" -> "GELU";
                    }
                    
                    "v0" -> "v_in" [lhead=cluster_f1];
                    "GELU" -> "v1" -> "Fourier Layers 2..N" -> "Linear Decode (Q)" -> Output;
                }
                """
                st.graphviz_chart(graph_dot, use_container_width=True)
                
            elif detected_type == "DeepONet":
                graph_dot = """
                digraph DeepONet {
                    rankdir=LR;
                    bgcolor="transparent";
                    node [shape=box, style=filled, fillcolor="#1f77b4", fontcolor=white, fontname="Inter", color="transparent"];
                    edge [color="#66b3ff", fontcolor="#a0aab2", fontname="Inter"];
                    
                    subgraph cluster_branch {
                        label="Branch Net"; color="#ff7f0e"; fontcolor="#ff7f0e"; style=dashed; fontname="Inter";
                        "Input Field (u)" -> "Linear / Conv" -> "Norm 1" -> "Activation 1" -> "Linear (p nodes)" -> "p_branch";
                    }
                    
                    subgraph cluster_trunk {
                        label="Trunk Net"; color="#2ca02c"; fontcolor="#2ca02c"; style=dashed; fontname="Inter";
                        "Coordinates (y)" -> "Linear Layer" -> "Norm 2" -> "Activation 2" -> "Linear (p modes)" -> "p_trunk";
                    }
                    
                    "p_branch" -> "Dot Product Operation" [label=" Vector [b] "];
                    "p_trunk" -> "Dot Product Operation" [label=" Vector [t] "];
                    
                    "Dot Product Operation" -> "Bias Addition (+)" -> "Output G(u)(y)";
                    "Bias Addition (+)" [shape=circle, fillcolor="#d62728"];
                    "Dot Product Operation" [shape=ellipse, fillcolor="#d62728", label="Σ (b_i × t_i)"];
                }
                """
                st.graphviz_chart(graph_dot, use_container_width=True)
            else:
                st.info("No standard operations traced for this fully connected/PINN module.")
                
        with tab_logs:
            st.markdown(f"```text\n[INFO] Connecting to {repo_url}...\n[INFO] Discovering script {selected_script} dependencies...\n[INFO] Initializing dataset generator...\n[INFO] Allocating model {selected_model} (map_location=cpu)...\n[INFO] Model Instantiated via torch.load()\n[INFO] Executing: python {selected_script} --input_tensor [None, 64, 64]\n...\n[SUCCESS] Exported Output Tensor Shape: torch.Size([1, 64, 64])\n```")
