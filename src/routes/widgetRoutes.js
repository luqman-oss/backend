/**
 * Widget Routes
 * Serves face measurement widget for iframe embedding on product pages
 */

const express = require('express');
const router = express.Router();
const path = require('path');

/**
 * @route   GET /api/widget/measure
 * @desc    Serve face measurement widget page for iframe
 * @access  Public
 */
router.get('/measure', (req, res) => {
  const { shop, product } = req.query;
  
  // Serve widget HTML directly
  res.send(generateWidgetHtml({ shop, product }));
});

/**
 * Generate standalone widget HTML that loads the React app
 */
function generateWidgetHtml({ shop, product }) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Face Measurement - Alira</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #1a1a2e;
      color: #e0e0e0;
      min-height: 100vh;
      overflow-x: hidden;
    }
    #root { width: 100%; min-height: 100vh; padding: 20px; }
    .widget-container {
      max-width: 600px;
      margin: 0 auto;
      background: #252542;
      border-radius: 16px;
      padding: 24px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .header h2 {
      color: #fff;
      font-size: 20px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .camera-container {
      background: #0d0d1a;
      border-radius: 12px;
      aspect-ratio: 4/3;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 16px;
      position: relative;
      overflow: hidden;
    }
    #video, #canvas {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 12px;
    }
    .placeholder {
      text-align: center;
      color: #666;
    }
    .placeholder-icon {
      font-size: 48px;
      margin-bottom: 12px;
    }
    .btn {
      padding: 12px 24px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn-primary {
      background: #3b6978;
      color: white;
    }
    .btn-primary:hover { background: #4a7a8a; }
    .btn-danger {
      background: #c0392b;
      color: white;
    }
    .btn-success {
      background: #27ae60;
      color: white;
    }
    .controls {
      display: flex;
      gap: 12px;
      justify-content: center;
      flex-wrap: wrap;
    }
    .results {
      margin-top: 20px;
      padding: 16px;
      background: #1a1a35;
      border-radius: 12px;
    }
    .results h3 {
      color: #fff;
      margin-bottom: 12px;
      font-size: 16px;
    }
    .measurement-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }
    .measurement-item {
      background: #252542;
      padding: 12px;
      border-radius: 8px;
    }
    .measurement-label {
      color: #888;
      font-size: 12px;
      margin-bottom: 4px;
    }
    .measurement-value {
      color: #fff;
      font-size: 18px;
      font-weight: 600;
    }
    .error {
      background: #c0392b;
      color: white;
      padding: 12px;
      border-radius: 8px;
      margin-top: 12px;
    }
    .processing {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0,0,0,0.8);
      padding: 20px 40px;
      border-radius: 12px;
      text-align: center;
    }
    .spinner {
      width: 40px;
      height: 40px;
      border: 3px solid #333;
      border-top-color: #3b6978;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin: 0 auto 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .upload-area {
      border: 2px dashed #444;
      border-radius: 12px;
      padding: 40px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      margin-top: 16px;
    }
    .upload-area:hover {
      border-color: #3b6978;
      background: rgba(59, 105, 120, 0.1);
    }
    .hidden { display: none !important; }
  </style>
</head>
<body>
  <div id="root">
    <div class="widget-container">
      <div class="header">
        <h2><span>📷</span> Face Measurement</h2>
      </div>
      
      <div class="camera-container" id="cameraContainer">
        <div class="placeholder" id="placeholder">
          <div class="placeholder-icon">👤</div>
          <p>Camera preview will appear here</p>
        </div>
        <video id="video" class="hidden" autoplay playsinline></video>
        <canvas id="canvas" class="hidden"></canvas>
        <div class="processing hidden" id="processing">
          <div class="spinner"></div>
          <p>Processing...</p>
        </div>
      </div>
      
      <div class="controls">
        <button class="btn btn-primary" id="startBtn" onclick="startCamera()">📹 Start Camera</button>
        <button class="btn btn-success" id="captureBtn" onclick="captureAndMeasure()" disabled>📷 Capture & Measure</button>
        <button class="btn btn-danger" id="stopBtn" onclick="stopCamera()" disabled>⏹ Stop</button>
      </div>
      
      <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept="image/*" class="hidden" onchange="handleFile(this.files[0])">
        <p>📁 Drag & drop a photo or click to browse</p>
      </div>
      
      <div class="results hidden" id="results">
        <h3>📏 Measurements</h3>
        <div class="measurement-grid" id="measurementGrid"></div>
      </div>
      
      <div class="error hidden" id="error"></div>
    </div>
  </div>
  
  <script>
    const API_BASE = '/api';
    let stream = null;
    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let placeholder = document.getElementById('placeholder');
    let processing = document.getElementById('processing');
    let results = document.getElementById('results');
    let error = document.getElementById('error');
    
    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 1280, height: 720, facingMode: 'user' } 
        });
        video.srcObject = stream;
        video.classList.remove('hidden');
        placeholder.classList.add('hidden');
        canvas.classList.add('hidden');
        document.getElementById('startBtn').disabled = true;
        document.getElementById('captureBtn').disabled = false;
        document.getElementById('stopBtn').disabled = false;
        hideError();
      } catch (err) {
        showError('Camera access denied. Please allow camera permissions.');
      }
    }
    
    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.classList.add('hidden');
      placeholder.classList.remove('hidden');
      canvas.classList.add('hidden');
      document.getElementById('startBtn').disabled = false;
      document.getElementById('captureBtn').disabled = true;
      document.getElementById('stopBtn').disabled = true;
      results.classList.add('hidden');
    }
    
    async function captureAndMeasure() {
      if (!stream) return;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      video.classList.add('hidden');
      canvas.classList.remove('hidden');
      processing.classList.remove('hidden');
      document.getElementById('captureBtn').disabled = true;
      hideError();
      
      try {
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));
        await processImage(blob);
      } catch (err) {
        showError('Failed to capture image: ' + err.message);
        processing.classList.add('hidden');
      }
    }
    
    async function handleFile(file) {
      if (!file || !file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
      }
      
      placeholder.classList.add('hidden');
      processing.classList.remove('hidden');
      hideError();
      
      try {
        await processImage(file);
      } catch (err) {
        showError('Failed to process image: ' + err.message);
        processing.classList.add('hidden');
      }
    }
    
    async function processImage(blob) {
      const formData = new FormData();
      formData.append('image', blob, 'face.jpg');
      
      const response = await fetch(\`\${API_BASE}/measure\`, {
        method: 'POST',
        body: formData
      });
      
      processing.classList.add('hidden');
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error?.message || 'Measurement failed');
      }
      
      const data = await response.json();
      displayResults(data);
    }
    
    function displayResults(data) {
      const grid = document.getElementById('measurementGrid');
      const measurements = [
        { label: 'Pupillary Distance (PD)', value: data.pupillary_distance_mm, unit: 'mm' },
        { label: 'Face Width', value: data.face_width_mm, unit: 'mm' },
        { label: 'Face Height', value: data.face_height_mm, unit: 'mm' },
        { label: 'Eye Width', value: data.eye_width_mm, unit: 'mm' },
        { label: 'Bridge Width', value: data.bridge_width_mm, unit: 'mm' }
      ].filter(m => m.value);
      
      grid.innerHTML = measurements.map(m => \`
        <div class="measurement-item">
          <div class="measurement-label">\${m.label}</div>
          <div class="measurement-value">\${m.value} \${m.unit}</div>
        </div>
      \`).join('');
      
      results.classList.remove('hidden');
      
      // Send to parent window
      window.parent.postMessage({
        type: 'MEASUREMENT_COMPLETE',
        data: data
      }, '*');
    }
    
    function showError(msg) {
      error.textContent = msg;
      error.classList.remove('hidden');
    }
    
    function hideError() {
      error.classList.add('hidden');
    }
    
    // Drag and drop support
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.style.borderColor = '#3b6978'; });
    uploadArea.addEventListener('dragleave', (e) => { e.preventDefault(); uploadArea.style.borderColor = '#444'; });
    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.style.borderColor = '#444';
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
  </script>
</body>
</html>`;
}

/**
 * @route   GET /api/widget/config
 * @desc    Get widget configuration for a shop
 * @access  Public
 */
router.get('/config', (req, res) => {
  const { shop } = req.query;
  
  // Return default widget configuration
  // In production, this would fetch from database
  res.json({
    buttonText: 'Measure Your Face',
    buttonStyle: 'rounded',
    buttonColor: '#3b6978',
    textColor: '#ffffff',
    position: 'above_add_to_cart',
    showPreview: true,
    requiredMeasurements: ['pupillary_distance_mm', 'face_width_mm']
  });
});

module.exports = router;
