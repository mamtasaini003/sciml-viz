import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import json
import os

st.set_page_config(page_title="FNO Architecture Visualizer", layout="wide", initial_sidebar_state="collapsed")

# Load model
model_dir = "models"
available = [m for m in os.listdir(model_dir) if m.endswith('.pth')] if os.path.exists(model_dir) else []
if not available:
    st.error("No models found!")
    st.stop()

selected = available[0]
sd = torch.load(os.path.join(model_dir, selected), map_location='cpu', weights_only=False)

# Pre-compute layer metadata for JS
layers_meta = {}
for key in sd:
    t = sd[key]
    arr = t.numpy()
    if np.iscomplexobj(arr): arr = np.abs(arr).astype(np.float32)
    else: arr = arr.astype(np.float32)
    layers_meta[key] = {
        'name': key,
        'shape': list(t.shape),
        'params': int(t.numel()),
        'dtype': str(t.dtype),
        'vmin': round(float(arr.min()), 5),
        'vmax': round(float(arr.max()), 5),
        'mean': round(float(arr.mean()), 5),
    }

meta_json = json.dumps(layers_meta)

html_code = """
<!DOCTYPE html>
<html>
<head>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background: #1a1a2e; overflow: hidden; font-family: 'Segoe UI', sans-serif; }
canvas#c { display: block; }

#info {
    position: fixed; top: 12px; right: 12px;
    background: rgba(22,27,34,0.96); border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px; color: #c9d1d9;
    width: 300px; z-index: 100; font-size: 13px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
#info h2 { color: #58a6ff; font-size: 14px; margin-bottom: 8px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
#info .r { display:flex; justify-content:space-between; margin:3px 0; }
#info .l { color:#8b949e; }
#info .v { color:#f0883e; font-weight:600; }

#hdr {
    position: fixed; top: 12px; left: 12px;
    background: rgba(22,27,34,0.96); border: 1px solid #30363d;
    border-radius: 10px; padding: 12px 18px; z-index: 100; color: #c9d1d9;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
#hdr h1 { font-size: 16px; color: #58a6ff; margin:0; }
#hdr p { font-size: 11px; color: #8b949e; margin-top:3px; }

#legend {
    position: fixed; bottom: 12px; left: 12px;
    background: rgba(22,27,34,0.96); border: 1px solid #30363d;
    border-radius: 10px; padding: 12px 16px; z-index: 100; color: #c9d1d9;
    font-size: 11px; box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
.leg-item { display:flex; align-items:center; margin: 3px 0; }
.leg-box { width: 14px; height: 14px; border-radius: 3px; margin-right: 8px; border: 1px solid rgba(255,255,255,0.2); }

#hint {
    position: fixed; bottom: 12px; right: 12px;
    background: rgba(22,27,34,0.9); border: 1px solid #30363d;
    border-radius: 20px; padding: 7px 16px; color: #8b949e;
    font-size: 11px; z-index: 100;
}
</style>
</head>
<body>

<div id="hdr">
    <h1>⚛️ Fourier Neural Operator</h1>
    <p>2D FNO for Navier-Stokes — Interactive Architecture</p>
</div>

<div id="info">
    <h2 id="it">Hover a component</h2>
    <div class="r"><span class="l">Layer:</span><span class="v" id="in">—</span></div>
    <div class="r"><span class="l">Shape:</span><span class="v" id="is">—</span></div>
    <div class="r"><span class="l">Parameters:</span><span class="v" id="ip">—</span></div>
    <div class="r"><span class="l">Type:</span><span class="v" id="id">—</span></div>
    <div class="r"><span class="l">Range:</span><span class="v" id="ir">—</span></div>
    <div class="r"><span class="l">Operation:</span><span class="v" id="io">—</span></div>
</div>

<div id="legend">
    <div style="font-weight:600; margin-bottom:5px; color:#58a6ff;">Legend</div>
    <div class="leg-item"><div class="leg-box" style="background:#4a9eff;"></div>Intermediate (data flow)</div>
    <div class="leg-item"><div class="leg-box" style="background:#2ea043;"></div>Weight matrix</div>
    <div class="leg-item"><div class="leg-box" style="background:#f0883e;"></div>Spectral Conv weights (ℂ)</div>
    <div class="leg-item"><div class="leg-box" style="background:#a371f7;"></div>Bias vector</div>
    <div class="leg-item"><div class="leg-box" style="background:#da3633;"></div>Addition / GELU</div>
    <div class="leg-item"><div class="leg-box" style="background:#8b949e;"></div>FFT / IFFT operation</div>
</div>

<div id="hint">🖱 Drag to rotate · Scroll to zoom · Right-drag to pan</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const META = """ + meta_json + """;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 600);
camera.position.set(55, 30, 55);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, -40, 0);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const d1 = new THREE.DirectionalLight(0xffffff, 0.5);
d1.position.set(30, 50, 30);
scene.add(d1);
const d2 = new THREE.DirectionalLight(0x8888ff, 0.2);
d2.position.set(-20, 30, -20);
scene.add(d2);

// Colors — matching bbycroft style: solid, distinct, no gradients
const C = {
    inter:   0x4a9eff, // intermediate data (blue)
    weight:  0x2ea043, // weight matrices (green)
    spectral:0xf0883e, // spectral conv complex weights (orange)
    bias:    0xa371f7, // bias vectors (purple)
    op:      0xda3633, // operations like add, gelu (red)
    fft:     0x8b949e, // FFT/IFFT (grey)
};

const allMeshes = [];
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-999, -999);

// ─── Block builder ───
function mkBlock(x, y, z, w, h, d, color, name, meta_key, opDesc, opacity) {
    opacity = opacity || 0.85;
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshPhongMaterial({ color: color, transparent: true, opacity: opacity, shininess: 30 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.userData = { name: name, meta_key: meta_key, opDesc: opDesc, baseColor: color, baseOpacity: opacity };
    scene.add(mesh);
    allMeshes.push(mesh);

    // Thin wireframe
    const edges = new THREE.EdgesGeometry(geo);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.15 }));
    line.position.copy(mesh.position);
    scene.add(line);
    return mesh;
}

// ─── Label builder ───  
function mkLabel(x, y, z, text, size, color) {
    const cv = document.createElement('canvas');
    cv.width = 512; cv.height = 64;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = color || '#c9d1d9';
    ctx.font = (size || 24) + 'px Segoe UI';
    ctx.fillText(text, 4, 40);
    const tex = new THREE.CanvasTexture(cv);
    const sp = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.8 });
    const spr = new THREE.Sprite(sp);
    spr.scale.set(10, 1.2, 1);
    spr.position.set(x, y, z);
    scene.add(spr);
}

// ─── Connection lines ───
function mkLine(pts, color, dashed) {
    const geo = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    let mat;
    if (dashed) {
        mat = new THREE.LineDashedMaterial({ color: color, dashSize: 0.5, gapSize: 0.3, transparent: true, opacity: 0.5 });
    } else {
        mat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.4 });
    }
    const line = new THREE.Line(geo, mat);
    if (dashed) line.computeLineDistances();
    scene.add(line);
}

// ─── Section labels ───
function mkSection(y, text) {
    mkLabel(-25, y, 0, text, 28, '#58a6ff');
}

// ══════════════════════════════════════════════════════════════
// LAY OUT THE FNO ARCHITECTURE
// Center X=0 is the residual stream (intermediate values)
// Weights go to the LEFT (negative X)
// Biases are small blocks next to weights
// ══════════════════════════════════════════════════════════════

let Y = 0;
const CELL = 1.2; // unit cell size
const GAP = 4;    // gap between major sections
const W = 32;     // model width (channels)
const M = 12;     // fourier modes

// ───────── INPUT ─────────
mkSection(Y + 1, '── Input ──');
mkBlock(0, Y, 0, 8, 1, 8, C.inter, 'Input Field a(x)', null, 'Raw PDE input [B, 64, 64, 1]');
mkLabel(12, Y, 0, 'a(x)  [B, 64, 64, 1]', 20);
Y -= GAP;

// ───────── LIFTING (fc0) ─────────
mkSection(Y + 1, '── Lift ──');
// Weight to the left
mkBlock(-12, Y, 0, 4, 4, 1.5, C.weight, 'Lift Weight (P)', 'fc0.weight', 'Linear: [3, 32]  →  Multiply');
mkLabel(-20, Y, 0, 'fc0.weight [3×32]', 18);
// Bias
mkBlock(-6, Y, 0, 1, 4, 1.5, C.bias, 'Lift Bias', 'fc0.bias', 'Add bias [32]');
// Arrow from weight to intermediate
mkLine([[-6, Y, 0], [-3, Y, 0]], 0xffffff, false);
// Intermediate (result)
mkBlock(0, Y, 0, 8, 4, 8, C.inter, 'v₀ (Lifted)', null, 'Projected features [B, 64, 64, 32]');
mkLabel(12, Y, 0, 'v₀  [B, 64, 64, 32]', 20);

Y -= GAP + 2;

// ───────── FOURIER LAYERS ×4 ─────────
for (let layer = 0; layer < 4; layer++) {
    const ly = Y;
    const isLast = (layer === 3);
    mkSection(ly + 2, '── Fourier Layer ' + layer + ' ──');
    
    // ── SPECTRAL PATH (left branch) ──
    const specX = -18;
    
    // FFT block
    mkBlock(specX, ly, 0, 3, 2, 3, C.fft, 'FFT (Layer ' + layer + ')', null, '2D Fast Fourier Transform');
    mkLabel(specX - 8, ly, 0, 'FFT  𝓕(v)', 18);
    
    // Spectral weight R_k (complex) — weights1
    mkBlock(specX, ly - 4, -4, 3, 3, 3, C.spectral, 'R₊ weights (L' + layer + ')', 'conv' + layer + '.weights1', 'Complex multiply [' + W + ',' + W + ',' + M + ',' + M + ']');
    mkLabel(specX - 10, ly - 4, -4, 'conv' + layer + '.weights1', 16, '#f0883e');
    
    // Spectral weight R_k — weights2
    mkBlock(specX, ly - 4, 4, 3, 3, 3, C.spectral, 'R₋ weights (L' + layer + ')', 'conv' + layer + '.weights2', 'Complex multiply (neg freq)');
    mkLabel(specX - 10, ly - 4, 4, 'conv' + layer + '.weights2', 16, '#f0883e');
    
    // IFFT
    mkBlock(specX, ly - 8, 0, 3, 2, 3, C.fft, 'IFFT (Layer ' + layer + ')', null, 'Inverse FFT  𝓕⁻¹');
    mkLabel(specX - 8, ly - 8, 0, 'IFFT  𝓕⁻¹', 18);
    
    // Spectral path connections
    mkLine([[0, ly + 2, 0], [specX, ly + 2, 0], [specX, ly + 1, 0]], 0x8b949e, true); // from residual to FFT
    mkLine([[specX, ly - 1, 0], [specX, ly - 2.5, -4]], 0xf0883e, false); // FFT to weights1
    mkLine([[specX, ly - 1, 0], [specX, ly - 2.5, 4]], 0xf0883e, false);  // FFT to weights2
    mkLine([[specX, ly - 5.5, -4], [specX, ly - 7, 0]], 0xf0883e, false); // weights1 to IFFT
    mkLine([[specX, ly - 5.5, 4], [specX, ly - 7, 0]], 0xf0883e, false);  // weights2 to IFFT
    mkLine([[specX, ly - 9, 0], [specX, ly - 10, 0], [-3, ly - 10, 0]], 0x8b949e, true); // IFFT back to add
    
    // ── SKIP PATH (right branch) ──
    const skipX = 14;
    
    // 1×1 Conv weight
    mkBlock(skipX, ly - 3, 0, 4, 4, 1.5, C.weight, 'W skip weight (L' + layer + ')', 'w' + layer + '.weight', '1×1 Conv [' + W + ',' + W + ',1,1]');
    mkLabel(skipX + 6, ly - 3, 0, 'w' + layer + '.weight', 18);
    
    // 1×1 Conv bias
    mkBlock(skipX + 6, ly - 3, 0, 1, 4, 1.5, C.bias, 'W skip bias (L' + layer + ')', 'w' + layer + '.bias', 'Bias [' + W + ']');
    
    // Skip connection line from residual
    mkLine([[3, ly + 1, 0], [skipX, ly + 1, 0], [skipX, ly - 1, 0]], 0x2ea043, true);
    // Skip back to add
    mkLine([[skipX, ly - 5, 0], [skipX, ly - 10, 0], [3, ly - 10, 0]], 0x2ea043, true);
    
    // ── ADD node ──
    mkBlock(0, ly - 10, 0, 4, 2, 4, C.op, '⊕ Add (L' + layer + ')', null, 'Spectral + Skip residual');
    mkLabel(6, ly - 10, 0, '⊕  Add', 20, '#da3633');
    
    if (!isLast) {
        // GELU
        mkBlock(0, ly - 13, 0, 4, 1.5, 4, C.op, 'GELU (L' + layer + ')', null, 'σ(x) = x·Φ(x)  non-linearity');
        mkLabel(6, ly - 13, 0, 'GELU  σ', 20, '#da3633');
        
        // Output intermediate
        mkBlock(0, ly - 16, 0, 8, 4, 8, C.inter, 'v' + (layer + 1), null, 'Intermediate [B, 64, 64, 32]');
        mkLabel(12, ly - 16, 0, 'v' + (layer+1) + '  [B, 64, 64, 32]', 20);
        
        // Vertical residual line
        mkLine([[0, ly - 18, 0], [0, ly - 20, 0]], 0x4a9eff, false);
        
        Y = ly - 21;
    } else {
        // Last layer has no GELU — output goes straight to decode
        mkBlock(0, ly - 13, 0, 8, 4, 8, C.inter, 'v_K (final features)', null, 'Final features [B, 64, 64, 32]');
        mkLabel(12, ly - 13, 0, 'v_K  [B, 64, 64, 32]', 20);
        Y = ly - 18;
    }
}

Y -= GAP;

// ───────── DECODE ─────────
mkSection(Y + 1, '── Decode ──');

// fc1 weight
mkBlock(-12, Y, 0, 5, 5, 1.5, C.weight, 'Decode W₁', 'fc1.weight', 'Linear: [32 → 128]');
mkLabel(-20, Y, 0, 'fc1.weight [32×128]', 18);
mkBlock(-6, Y, 0, 1, 5, 1.5, C.bias, 'Decode b₁', 'fc1.bias', 'Bias [128]');
mkLine([[-6, Y, 0], [-3, Y, 0]], 0xffffff, false);
mkBlock(0, Y, 0, 8, 5, 8, C.inter, 'Decoded (128)', null, '[B, 64, 64, 128]');
mkLabel(12, Y, 0, '[B, 64, 64, 128]', 20);

Y -= GAP;

// GELU
mkBlock(0, Y, 0, 4, 1.5, 4, C.op, 'GELU', null, 'σ(x) activation');
mkLabel(6, Y, 0, 'GELU', 18, '#da3633');

Y -= GAP;

// fc2 weight
mkBlock(-12, Y, 0, 4, 2, 1.5, C.weight, 'Output W₂', 'fc2.weight', 'Linear: [128 → 1]');
mkLabel(-20, Y, 0, 'fc2.weight [128×1]', 18);
mkBlock(-6, Y, 0, 1, 2, 1.5, C.bias, 'Output b₂', 'fc2.bias', 'Bias [1]');
mkLine([[-6, Y, 0], [-3, Y, 0]], 0xffffff, false);
mkBlock(0, Y, 0, 8, 1, 8, C.inter, 'Output u(x)', null, 'Predicted solution [B, 64, 64, 1]');
mkLabel(12, Y, 0, 'u(x)  [B, 64, 64, 1]', 20);

mkSection(Y - 3, '── Output ──');


// ══════════════════════════════════════════════════════════════
// INTERACTION
// ══════════════════════════════════════════════════════════════

let hovered = null;

window.addEventListener('mousemove', e => {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
});

function updatePanel(ud) {
    if (!ud) {
        document.getElementById('it').textContent = 'Hover a component';
        ['in','is','ip','id','ir','io'].forEach(i => document.getElementById(i).textContent = '—');
        return;
    }
    document.getElementById('it').textContent = ud.name;
    document.getElementById('io').textContent = ud.opDesc || '—';
    
    if (ud.meta_key && META[ud.meta_key]) {
        const m = META[ud.meta_key];
        document.getElementById('in').textContent = m.name;
        document.getElementById('is').textContent = '[' + m.shape.join(' × ') + ']';
        document.getElementById('ip').textContent = m.params.toLocaleString();
        document.getElementById('id').textContent = m.dtype;
        document.getElementById('ir').textContent = m.vmin + ' → ' + m.vmax;
    } else {
        document.getElementById('in').textContent = ud.name;
        document.getElementById('is').textContent = '—';
        document.getElementById('ip').textContent = '—';
        document.getElementById('id').textContent = '—';
        document.getElementById('ir').textContent = '—';
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(allMeshes);
    
    // Reset previous
    if (hovered) {
        hovered.material.opacity = hovered.userData.baseOpacity;
        hovered.material.emissive.setHex(0x000000);
    }
    
    if (hits.length > 0) {
        hovered = hits[0].object;
        hovered.material.opacity = 1.0;
        hovered.material.emissive.setHex(hovered.userData.baseColor);
        hovered.material.emissiveIntensity = 0.25;
        updatePanel(hovered.userData);
        renderer.domElement.style.cursor = 'pointer';
    } else {
        hovered = null;
        updatePanel(null);
        renderer.domElement.style.cursor = 'default';
    }
    
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""

components.html(html_code, height=800)
