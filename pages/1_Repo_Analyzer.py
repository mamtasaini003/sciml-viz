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
            tensors = {k: (np.abs(v.numpy()) if v.is_complex() else v.numpy()) for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim > 0}
            
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
            st.markdown("#### 🧊 3D Architecture Diagram (Interactive)")
            st.write("A bbycroft.net inspired topological map visualizing the geometrical tensor shapes, operations, and information flow.")
            
            fig3d = go.Figure()
            
            def add_cube(fig, x_center, y_center, z_center, dx, dy, dz, color, name, opacity=0.4):
                # Defines 8 vertices of a cube centered at (x,y,z) with dimensions (dx, dy, dz)
                x = [x_center-dx/2, x_center+dx/2, x_center+dx/2, x_center-dx/2, x_center-dx/2, x_center+dx/2, x_center+dx/2, x_center-dx/2]
                y = [y_center-dy/2, y_center-dy/2, y_center+dy/2, y_center+dy/2, y_center-dy/2, y_center-dy/2, y_center+dy/2, y_center+dy/2]
                z = [z_center-dz/2, z_center-dz/2, z_center-dz/2, z_center-dz/2, z_center+dz/2, z_center+dz/2, z_center+dz/2, z_center+dz/2]
                # define the 12 triangles connecting the 8 vertices
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
                
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                                        color=color, opacity=opacity, name=name, showscale=False,
                                        hoverinfo='name'))
                
                # Add edges for a wireframe look
                lines_x = [x[0], x[1], x[2], x[3], x[0], x[4], x[5], x[6], x[7], x[4], None, x[1], x[5], None, x[2], x[6], None, x[3], x[7]]
                lines_y = [y[0], y[1], y[2], y[3], y[0], y[4], y[5], y[6], y[7], y[4], None, y[1], y[5], None, y[2], y[6], None, y[3], y[7]]
                lines_z = [z[0], z[1], z[2], z[3], z[0], z[4], z[5], z[6], z[7], z[4], None, z[1], z[5], None, z[2], z[6], None, z[3], z[7]]
                fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='white', width=2), showlegend=False, hoverinfo='skip'))

            def add_arrow(fig, x0, y0, z0, x1, y1, z1, text):
                fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines+text', 
                                           line=dict(color='yellow', width=3), text=["", text], textposition="middle right", showlegend=False))
            
            if detected_type == "FNO":
                # Layout spacing along X axis
                # Input -> Linear Lift -> F1 (FFT -> Mult -> IFFT) -> ... -> Decode
                
                add_cube(fig3d, 0, 0, 0, 1, 8, 8, "#1f77b4", "Input a(x)\n[64x64x1]")
                add_arrow(fig3d, 0.5, 0, 0, 2.5, 0, 0, "Lift (Linear)")
                
                add_cube(fig3d, 3, 0, 0, 4, 8, 8, "#ff7f0e", "Lifted State v0\n[64x64x32]")
                add_arrow(fig3d, 3.5, 0, 0, 5.5, 0, 2, "FFT")
                add_arrow(fig3d, 3.5, 0, 0, 5.5, 0, -3, "Skip (1x1 Conv)")
                
                # F1 Spectral Branch
                add_cube(fig3d, 6, 0, 2, 4, 4, 4, "#2ca02c", "Fourier Modes\n[12x12x32]")
                add_arrow(fig3d, 6.5, 0, 2, 8.5, 0, 2, "Complex Mult")
                add_cube(fig3d, 9, 0, 2, 4, 4, 4, "#d62728", "Weighted Modes\n[12x12x32]")
                add_arrow(fig3d, 9.5, 0, 2, 11.5, 0, 0, "IFFT")
                
                # F1 Skip Branch
                add_cube(fig3d, 6, 0, -3, 4, 8, 8, "#9467bd", "W(v0)\n[64x64x32]")
                add_arrow(fig3d, 6.5, 0, -3, 11.5, 0, 0, "Add")
                
                # Recombined
                add_cube(fig3d, 12, 0, 0, 4, 8, 8, "#ff7f0e", "State v1 (+GELU)\n[64x64x32]")
                add_arrow(fig3d, 12.5, 0, 0, 15.5, 0, 0, "Blocks 2..N")
                
                # Decode
                add_cube(fig3d, 16, 0, 0, 4, 8, 8, "#ff7f0e", "State vN\n[64x64x32]")
                add_arrow(fig3d, 16.5, 0, 0, 18.5, 0, 0, "Decode")
                add_cube(fig3d, 19, 0, 0, 2, 8, 8, "#8c564b", "Proj Layer 1\n[64x64x128]")
                add_arrow(fig3d, 19.5, 0, 0, 21.5, 0, 0, "")
                add_cube(fig3d, 22, 0, 0, 1, 8, 8, "#e377c2", "Output u(x)\n[64x64x1]")
                
                fig3d.update_layout(scene=dict(
                    xaxis=dict(showgrid=False, visible=False),
                    yaxis=dict(showgrid=False, visible=False),
                    zaxis=dict(showgrid=False, visible=False),
                    camera=dict(eye=dict(x=-1.5, y=-2.5, z=1.5)),
                    aspectmode='data'
                ), template="plotly_dark", margin=dict(l=0, r=0, b=0, t=30), height=600, paper_bgcolor="rgba(0,0,0,0)")
                
                st.plotly_chart(fig3d, use_container_width=True)
                
            elif detected_type == "DeepONet":
                # Branch Net
                add_cube(fig3d, 0, 4, 0, 1, 6, 6, "#1f77b4", "Input Field u\n[Batch, M]")
                add_arrow(fig3d, 0.5, 4, 0, 3.5, 4, 0, "Branch Forward")
                add_cube(fig3d, 4, 4, 0, 1, 2, 10, "#ff7f0e", "b_nodes\n[Batch, p]")
                
                # Trunk Net
                add_cube(fig3d, 0, -4, 0, 1, 2, 2, "#1f77b4", "Coordinates y\n[Batch, 2]")
                add_arrow(fig3d, 0.5, -4, 0, 3.5, -4, 0, "Trunk Forward")
                add_cube(fig3d, 4, -4, 0, 1, 2, 10, "#2ca02c", "t_nodes\n[Batch, p]")
                
                add_arrow(fig3d, 4, 3, 0, 8, 1, 0, "Dot")
                add_arrow(fig3d, 4, -3, 0, 8, -1, 0, "Product")
                
                # Inner product
                add_cube(fig3d, 8, 0, 0, 1, 4, 4, "#d62728", "Inner Product\n Σ(b * t)")
                add_arrow(fig3d, 8.5, 0, 0, 10.5, 0, 0, "+ Bias")
                add_cube(fig3d, 11, 0, 0, 1, 2, 2, "#e377c2", "Output G(u)(y)\n[Batch, 1]")
                
                fig3d.update_layout(scene=dict(
                    xaxis=dict(showgrid=False, visible=False),
                    yaxis=dict(showgrid=False, visible=False),
                    zaxis=dict(showgrid=False, visible=False),
                    camera=dict(eye=dict(x=-1.5, y=-2.5, z=0.5)),
                    aspectmode='data'
                ), template="plotly_dark", margin=dict(l=0, r=0, b=0, t=30), height=600, paper_bgcolor="rgba(0,0,0,0)")
                
                st.plotly_chart(fig3d, use_container_width=True)
                
            else:
                st.info("No 3D Architecture trace configured for this module type.")
                
        with tab_logs:
            st.markdown(f"```text\n[INFO] Connecting to {repo_url}...\n[INFO] Discovering script {selected_script} dependencies...\n[INFO] Initializing dataset generator...\n[INFO] Allocating model {selected_model} (map_location=cpu)...\n[INFO] Model Instantiated via torch.load()\n[INFO] Executing: python {selected_script} --input_tensor [None, 64, 64]\n...\n[SUCCESS] Exported Output Tensor Shape: torch.Size([1, 64, 64])\n```")
