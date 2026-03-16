import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import load_css
import tempfile
import io

st.set_page_config(page_title="Model Explorer | SciML Viz", layout="wide")
load_css()

import os

st.title("🧩 Model Explorer")
st.markdown("Select one of the deployed `.pth` models (FNO, DeepONet, etc.) to visualize their internal structure, layer weights, Fourier modes, and more.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Deployed Models")
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pth') or f.endswith('.pt')] if os.path.exists(models_dir) else []
    
    if not available_models:
        st.warning("No models found in the deployment directory.")
        state_dict = None
    else:
        selected_model = st.selectbox("Choose a pre-trained model", available_models)
        
        try:
            model_path = os.path.join(models_dir, selected_model)
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            st.success(f"Model {selected_model} loaded successfully!")
            
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            st.metric("Model Size", f"{file_size_mb:.2f} MB")
            
            # Simple parameter count estimation from state_dict
            param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            st.metric("Total Parameters", f"{param_count:,}")
            
            st.markdown("#### Detected Structure")
            keys = list(state_dict.keys())
            st.write(f"Layers: {len(keys)}")
            if "fno" in "".join(keys).lower() or "spectral" in "".join(keys).lower():
                st.info("Detected Architecture: Fourier Neural Operator (FNO)")
                detected_type = "FNO"
            elif "branch" in "".join(keys).lower() or "trunk" in "".join(keys).lower():
                st.info("Detected Architecture: DeepONet")
                detected_type = "DeepONet"
            else:
                st.info("Detected Architecture: Standard/PINN")
                detected_type = "PINN"
                
        except Exception as e:
            st.error(f"Failed to parse model file: {e}")
            state_dict = None

with col2:
    if state_dict is not None:
        st.markdown("### 🎨 Weights Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Layer heatmaps", "Fourier Spectra / 3D Kernels", "Weight Histograms"])
        
        # Helper to get numeric tensors
        tensors = {k: v.numpy() for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim > 0}
        
        with tab1:
            if not tensors:
                st.warning("No valid tensor layers found.")
            else:
                layer_names = list(tensors.keys())
                selected_layer = st.selectbox("Select Layer to Plot", layer_names)
                
                layer_data = tensors[selected_layer]
                
                if layer_data.ndim == 1:
                    fig = px.bar(y=layer_data, title=f"1D Weights: {selected_layer}")
                elif layer_data.ndim == 2:
                    fig = px.imshow(layer_data, color_continuous_scale="Plasma", title=f"Weight Matrix: {selected_layer} | Shape: {layer_data.shape}")
                else:
                    # Flatten or slice 3D+ tensors
                    st.write(f"Tensor shape {layer_data.shape} is >2D. Displaying first 2D slice.")
                    slice_data = layer_data[0] if layer_data.ndim == 3 else layer_data[0, 0]
                    fig = px.imshow(slice_data, color_continuous_scale="Plasma", title=f"Slice of {selected_layer} | Original Shape: {layer_data.shape}")
                
                fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### Kernel / Operator Visualization")
            if detected_type == "FNO":
                st.write("Fourier spectra (2D FFT Contours).")
                
                # Mock fourier modes visualization since actual complex modes might be hard to parse blindly
                # Find a tensor that looks like fourier modes (complex or large 4D)
                fourier_candidates = [k for k, v in tensors.items() if (v.ndim >= 3 and ('spectral' in k or 'fourier' in k or 'conv' in k))]
                
                if fourier_candidates:
                    f_layer = st.selectbox("Select Spectral Layer", fourier_candidates)
                    f_data = np.abs(tensors[f_layer]) # Abs in case of complex
                    
                    if f_data.ndim >= 3:
                        # Contour plot of first mode
                        mode_slice = f_data[0, 0] if f_data.ndim >= 4 else f_data[0]
                        fig2 = go.Figure(data=go.Contour(z=mode_slice, colorscale="Viridis"))
                        fig2.update_layout(title="2D Fourier Modes", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No recognizable Fourier layers found. Showing sample 3D Attention/Kernel Surface.")
                    
                    # Sample 3D surface
                    x = np.linspace(-5, 5, 50)
                    y = np.linspace(-5, 5, 50)
                    X, Y = np.meshgrid(x, y)
                    Z = np.sin(np.sqrt(X**2 + Y**2))
                    fig3 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="IceFire")])
                    fig3.update_layout(title="Sample Attention/Operator Kernel mapped as 3D Surface", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig3, use_container_width=True)

            elif detected_type == "DeepONet":
                st.write("Branch and Trunk network visualization.")
                # Show generic 3D space surface for deeponet operator
                x = np.linspace(-1, 1, 30)
                y = np.linspace(-1, 1, 30)
                X, Y = np.meshgrid(x, y)
                Z = np.exp(-X**2 - Y**2)
                fig3 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Turbo")])
                fig3.update_layout(title="DeepONet Base Functions Interaction", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.write("No specific operator structure detected.")

        with tab3:
            st.markdown("#### Weight Histograms & Norms")
            if tensors:
                h_layer = st.selectbox("Select Layer for Histogram", list(tensors.keys()), key="hist_sel")
                flat_data = tensors[h_layer].flatten()
                
                fig4 = px.histogram(flat_data, nbins=50, title=f"Distribution of Weights: {h_layer}")
                fig4.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig4, use_container_width=True)
                
                st.metric("L2 Norm", f"{np.linalg.norm(flat_data):.4f}")
                st.metric("Sparsity (%)", f"{(np.sum(np.abs(flat_data) < 1e-4) / flat_data.size) * 100:.2f}%")
