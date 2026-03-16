import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import json
import os

st.set_page_config(page_title="FNO Architecture Visualizer", layout="wide", initial_sidebar_state="collapsed")

# Load model data
model_dir = "models"
available = [m for m in os.listdir(model_dir) if m.endswith('.pth')] if os.path.exists(model_dir) else []
if not available:
    st.error("No models found!")
    st.stop()

selected = available[0]
sd = torch.load(os.path.join(model_dir, selected), map_location='cpu', weights_only=False)

# Build layer data for JS
layers_js = []
for key in sd:
    t = sd[key]
    arr = t.numpy()
    if np.iscomplexobj(arr):
        arr = np.abs(arr).astype(np.float32)
    else:
        arr = arr.astype(np.float32)
    flat = arr.flatten()
    vmin, vmax = float(flat.min()), float(flat.max())
    rng = vmax - vmin if (vmax - vmin) > 1e-8 else 1.0
    normalized = ((flat - vmin) / rng)
    # Downsample for browser: max 2048 pixels per block face
    n = len(normalized)
    if n > 2048:
        step = max(1, n // 2048)
        normalized = normalized[::step]
        n = len(normalized)
    layers_js.append({
        'name': key,
        'shape': list(t.shape),
        'params': int(t.numel()),
        'dtype': str(t.dtype),
        'vmin': round(vmin, 5),
        'vmax': round(vmax, 5),
        'mean': round(float(t.float().mean()), 5),
        'pixels': [round(float(x), 3) for x in normalized],
        'n_pixels': n,
    })

layers_json = json.dumps(layers_js)

html_code = f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background: #0d1117; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
canvas {{ display: block; }}
#info-panel {{
    position: fixed; top: 12px; right: 12px;
    background: rgba(22,27,34,0.95); border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 22px; color: #c9d1d9;
    min-width: 280px; max-width: 340px; z-index: 100;
    backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}}
#info-panel h2 {{ color: #58a6ff; font-size: 15px; margin-bottom: 10px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
#info-panel .row {{ display: flex; justify-content: space-between; margin: 4px 0; font-size: 13px; }}
#info-panel .label {{ color: #8b949e; }}
#info-panel .val {{ color: #f0883e; font-weight: 600; }}
#title-bar {{
    position: fixed; top: 12px; left: 12px;
    background: rgba(22,27,34,0.95); border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 20px; z-index: 100; color: #c9d1d9;
    backdrop-filter: blur(12px);
}}
#title-bar h1 {{ font-size: 18px; color: #58a6ff; margin: 0; }}
#title-bar p {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
#controls-hint {{
    position: fixed; bottom: 12px; left: 50%; transform: translateX(-50%);
    background: rgba(22,27,34,0.9); border: 1px solid #30363d;
    border-radius: 20px; padding: 8px 20px; color: #8b949e;
    font-size: 12px; z-index: 100;
}}
</style>
</head>
<body>

<div id="title-bar">
    <h1>⚛️ FNO Architecture</h1>
    <p>Fourier Neural Operator — Interactive 3D View</p>
</div>

<div id="info-panel">
    <h2 id="panel-title">Hover over a block</h2>
    <div class="row"><span class="label">Layer:</span><span class="val" id="p-name">—</span></div>
    <div class="row"><span class="label">Shape:</span><span class="val" id="p-shape">—</span></div>
    <div class="row"><span class="label">Parameters:</span><span class="val" id="p-params">—</span></div>
    <div class="row"><span class="label">Type:</span><span class="val" id="p-dtype">—</span></div>
    <div class="row"><span class="label">Value Range:</span><span class="val" id="p-range">—</span></div>
    <div class="row"><span class="label">Mean:</span><span class="val" id="p-mean">—</span></div>
    <canvas id="mini-heatmap" width="256" height="40" style="margin-top:10px; border-radius:4px; width:100%;"></canvas>
</div>

<div id="controls-hint">
    🖱️ Drag to rotate &nbsp;|&nbsp; Scroll to zoom &nbsp;|&nbsp; Right-drag to pan &nbsp;|&nbsp; Hover blocks for details
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const LAYERS = {layers_json};

// Plasma colormap (approximate)
function plasma(t) {{
    t = Math.max(0, Math.min(1, t));
    const r = Math.min(1, 0.05 + t * 2.2 - t * t * 1.4);
    const g = Math.min(1, Math.max(0, -0.7 + t * 2.8 - t * t * 1.2));
    const b = Math.min(1, Math.max(0, 0.53 + t * 0.7 - t * t * 1.8));
    return [r, g, b];
}}

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.FogExp2(0x0d1117, 0.008);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(30, 25, 50);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, -15, 0);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dl = new THREE.DirectionalLight(0xffffff, 0.6);
dl.position.set(20, 30, 20);
scene.add(dl);

// Group categories
const categories = {{
    'fc0': {{ color: 0x1f77b4, label: 'Input Lifting', xOffset: 0 }},
    'conv0': {{ color: 0x2ca02c, label: 'Spectral Conv 0', xOffset: 0 }},
    'w0': {{ color: 0x9467bd, label: 'Skip Conv 0', xOffset: 12 }},
    'conv1': {{ color: 0x2ca02c, label: 'Spectral Conv 1', xOffset: 0 }},
    'w1': {{ color: 0x9467bd, label: 'Skip Conv 1', xOffset: 12 }},
    'conv2': {{ color: 0xff7f0e, label: 'Spectral Conv 2', xOffset: 0 }},
    'w2': {{ color: 0xd62728, label: 'Skip Conv 2', xOffset: 12 }},
    'conv3': {{ color: 0xff7f0e, label: 'Spectral Conv 3', xOffset: 0 }},
    'w3': {{ color: 0xd62728, label: 'Skip Conv 3', xOffset: 12 }},
    'fc1': {{ color: 0xe377c2, label: 'Decode Layer 1', xOffset: 0 }},
    'fc2': {{ color: 0x8c564b, label: 'Decode Layer 2', xOffset: 0 }},
}};

function getCat(name) {{
    for (const [prefix, cat] of Object.entries(categories)) {{
        if (name.startsWith(prefix)) return cat;
    }}
    return {{ color: 0x58a6ff, label: 'Other', xOffset: 0 }};
}}

// Create blocks
const blockGroup = new THREE.Group();
const blockMeshes = [];
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

let yPos = 0;
const SCALE = 0.0008; // scale params to visual size

LAYERS.forEach((layer, idx) => {{
    const cat = getCat(layer.name);
    const vol = Math.cbrt(layer.params * SCALE) * 3;
    const w = Math.max(1.5, vol);
    const h = Math.max(0.6, vol * 0.4);
    const d = Math.max(1.5, vol);
    
    // Create geometry with per-face vertex colors from actual weights
    const geo = new THREE.BoxGeometry(w, h, d, 
        Math.min(32, Math.ceil(Math.sqrt(layer.n_pixels))),
        1,
        Math.min(32, Math.ceil(Math.sqrt(layer.n_pixels)))
    );
    
    // Apply pixel colors to vertices
    const count = geo.attributes.position.count;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {{
        const pixIdx = i % layer.n_pixels;
        const [r, g, b] = plasma(layer.pixels[pixIdx]);
        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }}
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const mat = new THREE.MeshPhongMaterial({{
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
        shininess: 60,
    }});
    
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(cat.xOffset, yPos, 0);
    mesh.userData = {{ layerIdx: idx, layerData: layer, cat: cat }};
    
    blockGroup.add(mesh);
    blockMeshes.push(mesh);
    
    // Edge wireframe
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(w + 0.05, h + 0.05, d + 0.05));
    const lineMat = new THREE.LineBasicMaterial({{ color: cat.color, transparent: true, opacity: 0.7 }});
    const wireframe = new THREE.LineSegments(edges, lineMat);
    wireframe.position.copy(mesh.position);
    blockGroup.add(wireframe);
    
    // Label sprite
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#c9d1d9';
    ctx.font = 'bold 28px Segoe UI';
    ctx.fillText(layer.name, 10, 38);
    ctx.font = '22px Segoe UI';
    ctx.fillStyle = '#8b949e';
    ctx.fillText('[' + layer.shape.join('×') + ']', 10 + ctx.measureText(layer.name + '  ').width * 0.9, 38);
    
    const tex = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({{ map: tex, transparent: true, opacity: 0.85 }});
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(8, 1, 1);
    sprite.position.set(cat.xOffset - w/2 - 5, yPos, 0);
    blockGroup.add(sprite);
    
    // Connection lines to next block
    if (idx < LAYERS.length - 1) {{
        const nextCat = getCat(LAYERS[idx + 1].name);
        const nextVol = Math.cbrt(LAYERS[idx + 1].params * SCALE) * 3;
        const nextH = Math.max(0.6, nextVol * 0.4);
        const gap = h/2 + 0.3;
        const nextY = yPos - h/2 - nextH/2 - 1.8;
        
        const points = [
            new THREE.Vector3(cat.xOffset, yPos - h/2, 0),
            new THREE.Vector3(cat.xOffset, yPos - h/2 - 0.9, 0),
            new THREE.Vector3(nextCat.xOffset, nextY + nextH/2 + 0.9, 0),
            new THREE.Vector3(nextCat.xOffset, nextY + nextH/2, 0),
        ];
        const curve = new THREE.CatmullRomCurve3(points);
        const lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(20));
        const lineMat2 = new THREE.LineBasicMaterial({{ color: 0x30363d, linewidth: 1 }});
        blockGroup.add(new THREE.Line(lineGeo, lineMat2));
    }}
    
    yPos -= h + 1.8;
}});

scene.add(blockGroup);

// Hover logic
let hoveredMesh = null;

function onMouseMove(e) {{
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
}}
window.addEventListener('mousemove', onMouseMove);

function updateInfoPanel(data) {{
    if (!data) {{
        document.getElementById('panel-title').textContent = 'Hover over a block';
        document.getElementById('p-name').textContent = '—';
        document.getElementById('p-shape').textContent = '—';
        document.getElementById('p-params').textContent = '—';
        document.getElementById('p-dtype').textContent = '—';
        document.getElementById('p-range').textContent = '—';
        document.getElementById('p-mean').textContent = '—';
        return;
    }}
    const layer = data.layerData;
    document.getElementById('panel-title').textContent = data.cat.label;
    document.getElementById('p-name').textContent = layer.name;
    document.getElementById('p-shape').textContent = '[' + layer.shape.join(' × ') + ']';
    document.getElementById('p-params').textContent = layer.params.toLocaleString();
    document.getElementById('p-dtype').textContent = layer.dtype;
    document.getElementById('p-range').textContent = layer.vmin + ' → ' + layer.vmax;
    document.getElementById('p-mean').textContent = layer.mean.toString();
    
    // Mini heatmap
    const cv = document.getElementById('mini-heatmap');
    const ctx = cv.getContext('2d');
    const w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);
    const px = layer.pixels;
    const cols = w;
    const rows = Math.ceil(px.length / cols);
    const pw = w / cols, ph = h / Math.max(1, rows);
    for (let i = 0; i < px.length; i++) {{
        const [r, g, b] = plasma(px[i]);
        ctx.fillStyle = `rgb(${{Math.floor(r*255)}},${{Math.floor(g*255)}},${{Math.floor(b*255)}})`;
        const col = i % cols, row = Math.floor(i / cols);
        ctx.fillRect(col * pw, row * ph, Math.ceil(pw), Math.ceil(ph));
    }}
}}

// Animation loop
function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    
    // Raycast for hover
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(blockMeshes);
    
    if (hoveredMesh) {{
        hoveredMesh.material.opacity = 0.92;
        hoveredMesh.material.emissive = new THREE.Color(0x000000);
    }}
    
    if (intersects.length > 0) {{
        hoveredMesh = intersects[0].object;
        hoveredMesh.material.opacity = 1.0;
        hoveredMesh.material.emissive = new THREE.Color(hoveredMesh.userData.cat.color);
        hoveredMesh.material.emissiveIntensity = 0.3;
        updateInfoPanel(hoveredMesh.userData);
        document.body.style.cursor = 'pointer';
    }} else {{
        hoveredMesh = null;
        updateInfoPanel(null);
        document.body.style.cursor = 'default';
    }}
    
    renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>
"""

components.html(html_code, height=800)
