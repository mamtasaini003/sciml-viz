import streamlit as st
from utils import load_css

st.set_page_config(page_title="Paper Gallery | SciML Viz", layout="wide")
load_css()

st.title("📄 Paper Gallery")
st.markdown("A collection of related research papers, implementation details, and theoretical foundations for the models showcased in this hub.")

papers = [
    {
        "title": "Fourier Neural Operator for Parametric Partial Differential Equations",
        "authors": "Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar",
        "venue": "ICLR 2021",
        "link": "https://arxiv.org/abs/2010.08895",
        "code": "https://github.com/neuraloperator/neuraloperator",
        "desc": "The seminal paper introducing Fourier Neural Operators (FNO), which learn mappings between infinite-dimensional spaces and are resolution-invariant."
    },
    {
        "title": "DeepONet: Learning nonlinear operators for identifying differential equations",
        "authors": "Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, George Em Karniadakis",
        "venue": "Nature Machine Intelligence 2021",
        "link": "https://arxiv.org/abs/1910.03193",
        "code": "https://github.com/lululxvi/deeponet",
        "desc": "Foundational architecture for learning operators via branch and trunk networks based on the universal approximation theorem for operators."
    },
    {
        "title": "Physics-Informed Neural Networks (PINNs)",
        "authors": "Maziar Raissi, Paris Perdikaris, George E. Karniadakis",
        "venue": "Journal of Computational Physics 2019",
        "link": "https://arxiv.org/abs/1711.10561",
        "code": "https://github.com/maziarraissi/PINNs",
        "desc": "A breakthrough framework linking data-driven machine learning with physics-based constraints from PDEs."
    }
]

for paper in papers:
    st.markdown(f"### {paper['title']}")
    st.markdown(f"**Authors:** {paper['authors']}")
    st.markdown(f"**Venue:** {paper['venue']}")
    st.markdown(f"{paper['desc']}")
    
    col1, col2 = st.columns([1, 10])
    with col1:
        st.markdown(f"[📄 arXiv]({paper['link']})")
    with col2:
        st.markdown(f"[💻 GitHub]({paper['code']})")
    st.markdown("---")
