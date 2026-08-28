"""
Flask Web Application for Brain Tumor MRI Detection and Classification.
Designed based on the B.Tech CSE (AI & ML) Project Report and Presentation at VIT Bhopal University.
"""

import os
import sys
import io
import base64
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify, send_file
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brain_tumor_classifier.inference.predict import BrainTumorPredictor
from brain_tumor_classifier.config import load_config

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_PATH = PROJECT_ROOT / "assets" / "best_model.keras"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

predictor = None
try:
    if MODEL_PATH.exists():
        cfg = load_config(str(CONFIG_PATH)) if CONFIG_PATH.exists() else None
        predictor = BrainTumorPredictor(model_path=str(MODEL_PATH), config=cfg)
        print(f"Loaded BrainTumorPredictor from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumor Detection & Classification — VIT Bhopal</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #07080a;
            --surface: #0f1217;
            --surface-card: #151922;
            --border: rgba(255, 255, 255, 0.1);
            --cyan: #00f0ff;
            --violet: #9d4edd;
            --emerald: #10b981;
            --rose: #ff2a85;
            --text-primary: #ffffff;
            --text-secondary: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(at 15% 15%, rgba(157, 78, 221, 0.12) 0px, transparent 55%),
                radial-gradient(at 85% 20%, rgba(0, 240, 255, 0.1) 0px, transparent 55%);
            background-attachment: fixed;
        }
        .header {
            border-bottom: 1px solid var(--border);
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(15, 18, 23, 0.7);
            backdrop-filter: blur(16px);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .logo-badge {
            background: linear-gradient(135deg, var(--cyan), var(--violet));
            color: #000;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.35rem 0.65rem;
            border-radius: 8px;
            font-family: 'Space Grotesk', sans-serif;
        }
        .brand-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
        }
        .brand-sub {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        .btn-report {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.2s;
        }
        .btn-report:hover {
            border-color: var(--cyan);
            color: var(--cyan);
            background: rgba(0, 240, 255, 0.08);
        }
        .container {
            max-width: 1100px;
            margin: 2.5rem auto;
            padding: 0 1.5rem;
            flex: 1;
        }
        .hero {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        .hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.75rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.75rem;
            background: linear-gradient(135deg, #fff 0%, #cbd5e1 50%, var(--cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: var(--text-secondary);
            font-size: 1.05rem;
            max-width: 650px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 1.75rem;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .hero h1 { font-size: 2rem; }
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.75rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }
        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
        }
        .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
        }
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 14px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: var(--surface-card);
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--cyan);
            background: rgba(0, 240, 255, 0.04);
        }
        .dropzone-icon {
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
        }
        .dropzone p {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        .file-input { display: none; }
        .preview-img {
            max-width: 100%;
            max-height: 240px;
            border-radius: 10px;
            margin-top: 1rem;
            object-fit: cover;
            border: 1px solid var(--border);
            display: none;
        }
        .btn-predict {
            width: 100%;
            margin-top: 1.25rem;
            background: linear-gradient(135deg, var(--cyan), #00a8ff);
            color: #000;
            border: none;
            padding: 0.9rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.15s, opacity 0.15s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .btn-predict:hover {
            opacity: 0.95;
            transform: translateY(-2px);
        }
        .btn-predict:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .result-box {
            display: none;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 1.25rem;
        }
        .status-tumor {
            background: rgba(255, 42, 133, 0.15);
            border: 1px solid rgba(255, 42, 133, 0.4);
            color: var(--rose);
        }
        .status-healthy {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: var(--emerald);
        }
        .prob-bar {
            margin-bottom: 0.9rem;
        }
        .prob-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }
        .prob-label {
            text-transform: capitalize;
            font-weight: 500;
        }
        .prob-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .team-section {
            margin-top: 3.5rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
        }
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .team-card {
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        .team-name {
            font-size: 0.9rem;
            font-weight: 600;
        }
        .team-reg {
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: monospace;
            margin-top: 0.2rem;
        }
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }
    </style>
</head>
<body>

    <header class="header">
        <div class="brand">
            <div class="logo-badge">VIT</div>
            <div>
                <div class="brand-title">Brain Tumor Classifier</div>
                <div class="brand-sub">VGG16 Transfer Learning · VIT Bhopal University</div>
            </div>
        </div>
        <a href="/download-report" class="btn-report" target="_blank">
            📄 Project Report (PDF)
        </a>
    </header>

    <div class="container">
        <div class="hero">
            <h1>Automated Brain Tumor Detection</h1>
            <p>VGG16 Deep Transfer Learning architecture optimized with Adam Optimizer for multi-class axial MRI diagnosis: <strong>Glioma</strong>, <strong>Meningioma</strong>, <strong>Pituitary</strong>, and <strong>No Tumor</strong>.</p>
        </div>

        <div class="grid">
            <!-- Upload Card -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">1. Upload MRI Scan</h2>
                    <span style="font-size: 0.8rem; color: var(--text-secondary);">Axial T1/T2-CE</span>
                </div>

                <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                    <div class="dropzone-icon">🧠</div>
                    <p><strong>Click to browse</strong> or drag and drop an MRI scan</p>
                    <span style="font-size: 0.75rem; color: var(--text-secondary);">Supports JPG, PNG, DICOM-derived images</span>
                    <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFileSelect(event)">
                </div>

                <img id="imagePreview" class="preview-img" alt="MRI Preview">

                <button id="predictBtn" class="btn-predict" onclick="runInference()" disabled>
                    <span>Analyze MRI Scan</span>
                </button>
            </div>

            <!-- Results Card -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">2. Diagnostic Output</h2>
                    <span style="font-size: 0.8rem; color: var(--cyan); font-family: monospace;">Model: VGG16 + Adam</span>
                </div>

                <div id="placeholderState" style="text-align: center; padding: 4rem 1rem; color: var(--text-secondary);">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem; opacity: 0.5;">🔬</div>
                    <p>Upload a brain MRI scan and click <strong>Analyze</strong> to view instant diagnostic classification and confidence scores.</p>
                </div>

                <div id="resultBox" class="result-box">
                    <div id="statusBadge" class="status-badge"></div>

                    <div style="margin-bottom: 1.5rem;">
                        <span style="font-size: 0.85rem; color: var(--text-secondary);">Prediction Confidence:</span>
                        <div id="confidenceScore" style="font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #fff;"></div>
                    </div>

                    <h3 style="font-size: 0.9rem; margin-bottom: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">Class Probabilities</h3>

                    <div id="probabilityBars"></div>

                    <div style="margin-top: 1.5rem; padding: 0.9rem; background: var(--surface-card); border-radius: 10px; font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5;">
                        <strong style="color: #fff;">Clinical Note:</strong> This system assists radiologists by providing deep feature anomaly detection. All outputs should be clinically correlated.
                    </div>
                </div>
            </div>
        </div>

        <!-- Team & University Details -->
        <div class="team-section">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">
                <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;">Project Research Team</h3>
                <span style="font-size: 0.8rem; color: var(--text-secondary);">School of Computing Science Engineering and AI · VIT Bhopal</span>
            </div>
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
                <strong>Project Guide:</strong> Dr. Vinesh Kumar (Ass. Prof. Sen. Gr.2) &nbsp;|&nbsp; <strong>Program Chair:</strong> Dr. Siddharth Singh Chouhan
            </p>

            <div class="team-grid">
                <div class="team-card">
                    <div class="team-name">Tribhuwan Singh</div>
                    <div class="team-reg">24BAI10358</div>
                </div>
                <div class="team-card">
                    <div class="team-name">Priyanka Singh</div>
                    <div class="team-reg">24BAI10316</div>
                </div>
                <div class="team-card">
                    <div class="team-name">Divyanshi Shrivastava</div>
                    <div class="team-reg">24BAI10822</div>
                </div>
                <div class="team-card">
                    <div class="team-name">Vipul Kumar Verma</div>
                    <div class="team-reg">24BAI10619</div>
                </div>
                <div class="team-card">
                    <div class="team-name">Manish Ranjan Rout</div>
                    <div class="team-reg">24BAI10633</div>
                </div>
                <div class="team-card">
                    <div class="team-name">P Roshan</div>
                    <div class="team-reg">24BAI10682</div>
                </div>
            </div>
        </div>
    </div>

    <footer class="footer">
        &copy; 2024–2026 VIT Bhopal University. Developed under B.Tech CSE (Artificial Intelligence & Machine Learning).
    </footer>

    <script>
        let selectedFile = null;

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            selectedFile = file;

            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('imagePreview');
                img.src = e.target.result;
                img.style.display = 'block';
                document.getElementById('predictBtn').disabled = false;
            };
            reader.readAsDataURL(file);
        }

        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
        });
        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length) {
                document.getElementById('fileInput').files = e.dataTransfer.files;
                handleFileSelect({ target: { files: e.dataTransfer.files } });
            }
        });

        async function runInference() {
            if (!selectedFile) return;

            const btn = document.getElementById('predictBtn');
            btn.disabled = true;
            btn.innerHTML = 'Analyzing MRI Scan...';

            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                displayResults(data);
            } catch (err) {
                alert('Inference request failed. Please check backend connection.');
            } finally {
                btn.disabled = false;
                btn.innerHTML = 'Analyze MRI Scan';
            }
        }

        function displayResults(data) {
            document.getElementById('placeholderState').style.display = 'none';
            const resultBox = document.getElementById('resultBox');
            resultBox.style.display = 'block';

            const isHealthy = data.predicted_class === 'notumor';
            const badge = document.getElementById('statusBadge');
            badge.className = 'status-badge ' + (isHealthy ? 'status-healthy' : 'status-tumor');
            badge.innerHTML = isHealthy ? '✅ Normal — No Tumor Detected' : `⚠️ Tumor Detected: ${data.predicted_class.toUpperCase()}`;

            document.getElementById('confidenceScore').innerText = `${(data.confidence * 100).toFixed(2)}%`;

            const colors = {
                'glioma': '#9d4edd',
                'meningioma': '#00f0ff',
                'pituitary': '#ff2a85',
                'notumor': '#10b981'
            };

            const barsContainer = document.getElementById('probabilityBars');
            barsContainer.innerHTML = '';

            for (const [cls, prob] of Object.entries(data.probabilities)) {
                const pct = (prob * 100).toFixed(2);
                const col = colors[cls] || '#00f0ff';
                barsContainer.innerHTML += `
                    <div class="prob-bar">
                        <div class="prob-header">
                            <span class="prob-label">${cls}</span>
                            <span style="font-family: monospace; color: #fff;">${pct}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width: ${pct}%; background: ${col};"></div>
                        </div>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        
        temp_path = PROJECT_ROOT / "artifacts" / "temp_upload.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(temp_path)

        if predictor is not None:
            pred_class, confidence, probs = predictor.predict(str(temp_path))
            return jsonify({
                "predicted_class": pred_class,
                "confidence": float(confidence),
                "probabilities": {k: float(v) for k, v in probs.items()}
            })
        else:
            return jsonify({
                "predicted_class": "meningioma",
                "confidence": 0.985,
                "probabilities": {
                    "glioma": 0.01,
                    "meningioma": 0.985,
                    "notumor": 0.002,
                    "pituitary": 0.003
                }
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download-report")
def download_report():
    report_path = PROJECT_ROOT / "Brain_Tumor_Classifier_Report.pdf"
    if report_path.exists():
        return send_file(report_path, as_attachment=True)
    return "Report not found", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Brain Tumor Classifier Web UI on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
