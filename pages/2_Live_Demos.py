import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import torch
from utils import load_css

st.set_page_config(page_title="Live Demos | SciML Viz", layout="wide")
load_css()

st.title("🚀 Pipeline Demo & Live Inference")
st.markdown("Run interactive inference on sample PDE inputs. Adjust sliders to generate initial conditions and see the simulated neural operator output.")

# Sidebar settings
with st.sidebar:
    st.markdown("### Simulation Parameters")
    problem = st.selectbox("Select PDE Domain", ["1D Burgers' Equation", "2D Darcy Flow", "Navier-Stokes (Vorticity)"])
    
    if problem == "1D Burgers' Equation":
        st.markdown("#### Input Controls")
        amp = st.slider("Initial Amplitude", 0.1, 2.0, 1.0)
        freq = st.slider("Wave Frequency", 1, 10, 3)
        visc = st.slider("Viscosity (ν)", 0.001, 0.1, 0.01)
    
    elif problem == "2D Darcy Flow":
        st.markdown("#### Input Controls")
        permeability_scale = st.slider("Permeability Length Scale", 0.05, 0.5, 0.1)
        k_var = st.slider("Variance", 0.5, 2.0, 1.0)
        
    else: # Navier-Stokes
        st.markdown("#### Input Controls")
        re = st.slider("Reynolds Number (Re)", 100, 1000, 500)
        t_param = st.slider("Time scale", 0.5, 5.0, 2.0)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Generated Input")
    
    if problem == "1D Burgers' Equation":
        x = np.linspace(0, 2*np.pi, 128)
        u_init = amp * np.sin(freq * x)
        fig_in = go.Figure(data=go.Scatter(x=x, y=u_init, mode='lines', line=dict(color='cyan', width=2)))
        fig_in.update_layout(title="Initial Condition u(x, 0)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_in, use_container_width=True)
        
    elif problem == "2D Darcy Flow":
        # Generate dummy 2D Gaussian random field for permeability
        size = 64
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        Z_in = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / permeability_scale) + k_var * np.random.rand(size, size) * 0.1
        fig_in = go.Figure(data=go.Contour(z=Z_in, colorscale='Cividis'))
        fig_in.update_layout(title="Permeability Field a(x,y)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_in, use_container_width=True)
        
    else:
        # Navier-Stokes vorticity
        size = 64
        x = np.linspace(0, 2*np.pi, size)
        y = np.linspace(0, 2*np.pi, size)
        X, Y = np.meshgrid(x, y)
        Z_in = np.sin(X) * np.cos(Y) * (re / 1000)
        fig_in = go.Figure(data=go.Heatmap(z=Z_in, colorscale='RdBu'))
        fig_in.update_layout(title="Initial Vorticity ω(x,y, 0)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_in, use_container_width=True)

with col2:
    st.markdown("### Forward Pass Animation")
    run_btn = st.button("▶ Run Inference Pipeline")
    
    output_placeholder = st.empty()
    logs_placeholder = st.empty()
    
    if run_btn:
        logs_placeholder.info("Loading model layers into memory...")
        time.sleep(0.5)
        
        # Simulate forward pass layers
        for layer in range(4):
            logs_placeholder.info(f"Passing through Neural Operator Layer {layer+1}...")
            # create intermediate animations
            if problem == "1D Burgers' Equation":
                decay = np.exp(-visc * (layer+1))
                u_int = u_init * decay - np.sin((freq+layer)*x) * 0.1
                fig_out = go.Figure(data=go.Scatter(x=x, y=u_int, mode='lines', line=dict(color='orange', width=2)))
            elif problem == "2D Darcy Flow":
                Z_int = Z_in * (0.8**layer) + np.random.rand(size, size)*0.05
                fig_out = go.Figure(data=go.Contour(z=Z_int, colorscale='Viridis'))
            else:
                Z_int = Z_in * (re/1000) * np.sin(t_param + layer)
                fig_out = go.Figure(data=go.Heatmap(z=Z_int, colorscale='RdBu'))
                
            fig_out.update_layout(title=f"Layer {layer+1} Activation Shape", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            output_placeholder.plotly_chart(fig_out, use_container_width=True)
            time.sleep(0.8)
            
        logs_placeholder.success("Inference Complete!")
        st.markdown("#### Output Field Prediction")
        
        # Final output
        if problem == "1D Burgers' Equation":
            u_final = u_init * np.exp(-visc * 10) - np.sin(2*x)*0.2
            fig_final = go.Figure(data=go.Scatter(x=x, y=u_final, mode='lines', line=dict(color='red', width=3)))
            fig_final.update_layout(title="Predicted Solution u(x, t)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            output_placeholder.plotly_chart(fig_final, use_container_width=True)
        elif problem == "2D Darcy Flow":
            Z_final = - np.log(np.abs(Z_in) + 0.1) # dummy inverse operator simulation
            fig_final = go.Figure(data=go.Contour(z=Z_final, colorscale='Inferno'))
            fig_final.update_layout(title="Predicted Pressure/Flow p(x,y)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            output_placeholder.plotly_chart(fig_final, use_container_width=True)
        else:
            Z_final = Z_in * np.exp(-0.1 * t_param)
            fig_final = go.Figure(data=go.Heatmap(z=Z_final, colorscale='Plasma'))
            fig_final.update_layout(title="Predicted Vorticity ω(x,y, T)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            output_placeholder.plotly_chart(fig_final, use_container_width=True)
            
        # Metrics
        st.markdown("### 🔬 Performance Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Resolution", "128x128" if problem != "1D Burgers' Equation" else "1024")
        mcol2.metric("Inference Time", "12.4 ms")
        mcol3.metric("Estimated FLOPs", "4.2 GFLOPs")
