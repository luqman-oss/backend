/**
 * ═══════════════════════════════════════════════════════════════
 *  FaceScan AI — Frontend Application
 *  
 *  Connects to the Facial Measurement API and provides:
 *  - Drag & drop image upload
 *  - Live camera capture
 *  - Real-time measurement display
 *  - System health monitoring
 *  - JSON export
 * ═══════════════════════════════════════════════════════════════
 */

// ─── Configuration ───────────────────────────────────────────
const ENDPOINTS = {
    measure: '/api/measure',
    measureDetailed: '/api/measure/detailed',
    measureMulti: '/api/measure/multi',
    health: '/api/health',
    healthDetailed: '/api/health/detailed',
};

// ─── State ───────────────────────────────────────────────────
let currentFile = null;
let lastResults = null;
let cameraStream = null;
let isProcessing = false;

// ─── Measurement Definitions ─────────────────────────────────
const MEASUREMENTS = [
    {
        key: 'pupillary_distance_mm',
        label: 'Pupillary Distance',
        icon: '👁️',
        cssClass: 'result-card--pd',
        description: 'Center-to-center distance between pupils',
    },
    {
        key: 'face_width_mm',
        label: 'Face Width',
        icon: '↔️',
        cssClass: 'result-card--fw',
        description: 'Cheekbone to cheekbone width',
    },
    {
        key: 'face_height_mm',
        label: 'Face Height',
        icon: '↕️',
        cssClass: 'result-card--fh',
        description: 'Forehead to chin distance',
    },
    {
        key: 'eye_width_mm',
        label: 'Eye Width',
        icon: '👀',
        cssClass: 'result-card--ew',
        description: 'Average eye width (inner to outer corner)',
    },
    {
        key: 'eye_height_mm',
        label: 'Eye Height',
        icon: '🔲',
        cssClass: 'result-card--eh',
        description: 'Average eye opening height',
    },
    {
        key: 'bridge_width_mm',
        label: 'Bridge Width',
        icon: '🔸',
        cssClass: 'result-card--bw',
        description: 'Nose bridge width',
    },
    {
        key: 'forehead_width_mm',
        label: 'Forehead Width',
        icon: '📏',
        cssClass: 'result-card--fow',
        description: 'Forehead boundary width',
    },
    {
        key: 'side_length_mm',
        label: 'Side / Arm Length',
        icon: '📐',
        cssClass: 'result-card--sl',
        description: 'Estimated glasses arm length',
    },
];


// ═══════════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    console.log('[FaceScan AI] Initializing frontend...');

    // Attach event listeners to replace inline onclicks (more reliable with CSP)
    document.getElementById('tabUpload')?.addEventListener('click', () => switchTab('upload'));
    document.getElementById('tabCamera')?.addEventListener('click', () => switchTab('camera'));
    document.getElementById('btnStartCamera')?.addEventListener('click', () => startCamera());
    document.getElementById('btnCapture')?.addEventListener('click', () => capturePhoto());
    document.getElementById('btnStopCamera')?.addEventListener('click', () => stopCamera());
    document.getElementById('btnMeasure')?.addEventListener('click', () => measureFace());
    document.getElementById('btnHealth')?.addEventListener('click', () => checkHealth());
    document.getElementById('btnExport')?.addEventListener('click', () => exportResults());

    // Select all elements that should trigger resetUpload
    document.querySelectorAll('[data-action="reset"]').forEach(el => {
        el.addEventListener('click', () => resetUpload());
    });

    initUploadZone();
    checkSystemStatus();
});


// ═══════════════════════════════════════════════════════════════
//  UPLOAD HANDLING
// ═══════════════════════════════════════════════════════════════

function initUploadZone() {
    const zone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    // Click to browse
    zone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag & drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('upload-zone--dragover');
    });

    zone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        zone.classList.remove('upload-zone--dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('upload-zone--dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

function handleFileSelect(file) {
    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showToast('Invalid file type. Please upload JPG, PNG, WebP, or BMP.', 'error');
        return;
    }

    // Validate size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showToast('File too large. Maximum size is 10MB.', 'error');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        showPreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

function showPreview(dataUrl) {
    const zone = document.getElementById('uploadZone');
    const preview = document.getElementById('previewSection');
    const previewImg = document.getElementById('previewImage');
    const badge = document.getElementById('previewBadge');

    zone.classList.add('hidden');
    preview.classList.remove('hidden');
    previewImg.src = dataUrl;
    badge.textContent = 'Ready';
    badge.className = 'preview__badge';

    // Reset results
    hideResults();
    showEmptyState();
}

function resetUpload() {
    const zone = document.getElementById('uploadZone');
    const cameraZone = document.getElementById('cameraZone');
    const preview = document.getElementById('previewSection');
    const fileInput = document.getElementById('fileInput');

    currentFile = null;
    lastResults = null;
    fileInput.value = '';

    preview.classList.add('hidden');

    // Show the correct zone based on active tab
    const isUploadTab = document.getElementById('tabUpload').classList.contains('tab-btn--active');
    if (isUploadTab) {
        zone.classList.remove('hidden');
    }

    hideResults();
    showEmptyState();

    document.getElementById('btnExport').disabled = true;
}


// ═══════════════════════════════════════════════════════════════
//  CAMERA HANDLING
// ═══════════════════════════════════════════════════════════════

function switchTab(tab) {
    const tabUpload = document.getElementById('tabUpload');
    const tabCamera = document.getElementById('tabCamera');
    const uploadZone = document.getElementById('uploadZone');
    const cameraZone = document.getElementById('cameraZone');
    const preview = document.getElementById('previewSection');

    if (tab === 'upload') {
        tabUpload.classList.add('tab-btn--active');
        tabCamera.classList.remove('tab-btn--active');
        cameraZone.classList.add('hidden');
        if (!currentFile) {
            uploadZone.classList.remove('hidden');
        }
        stopCamera();
    } else {
        tabCamera.classList.add('tab-btn--active');
        tabUpload.classList.remove('tab-btn--active');
        uploadZone.classList.add('hidden');
        preview.classList.add('hidden');
        cameraZone.classList.remove('hidden');
    }
}

async function startCamera() {
    try {
        const video = document.getElementById('cameraVideo');
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 960 },
                facingMode: 'user',
            },
        });
        video.srcObject = cameraStream;

        document.getElementById('btnStartCamera').classList.add('hidden');
        document.getElementById('btnCapture').classList.remove('hidden');
        document.getElementById('btnStopCamera').classList.remove('hidden');

        showToast('Camera started. Position your face in the oval guide.', 'info');
    } catch (err) {
        showToast(`Camera access denied: ${err.message}`, 'error');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
        cameraStream = null;
    }

    const video = document.getElementById('cameraVideo');
    video.srcObject = null;

    document.getElementById('btnStartCamera').classList.remove('hidden');
    document.getElementById('btnCapture').classList.add('hidden');
    document.getElementById('btnStopCamera').classList.add('hidden');
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Mirror the image (since video is mirrored with scaleX(-1))
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Convert to blob
    canvas.toBlob((blob) => {
        currentFile = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
        const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

        // Switch to upload tab with preview
        switchTab('upload');
        showPreview(dataUrl);
        showToast('Photo captured! Click "Analyze Face" to measure.', 'success');
    }, 'image/jpeg', 0.92);
}


// ═══════════════════════════════════════════════════════════════
//  MEASUREMENT API CALL
// ═══════════════════════════════════════════════════════════════

async function measureFace() {
    console.log('[FaceScan AI] measureFace called');
    if (!currentFile || isProcessing) {
        console.warn('[FaceScan AI] measureFace aborted:', { currentFile: !!currentFile, isProcessing });
        return;
    }

    isProcessing = true;
    const badge = document.getElementById('previewBadge');
    const btnMeasure = document.getElementById('btnMeasure');

    // UI: Processing state
    badge.textContent = 'Processing...';
    badge.className = 'preview__badge preview__badge--processing';
    btnMeasure.disabled = true;
    btnMeasure.innerHTML = '<span class="btn__icon">⏳</span><span>Analyzing...</span>';

    showLoading();
    hideResults();
    hideError();

    try {
        // Build form data
        const formData = new FormData();
        formData.append('image', currentFile);

        // Add calibration if provided
        const refPixels = document.getElementById('refPixels').value;
        if (refPixels && parseFloat(refPixels) > 0) {
            formData.append('reference_pixels', refPixels);
        }

        // API call
        const startTime = performance.now();
        const response = await fetch(ENDPOINTS.measure, {
            method: 'POST',
            body: formData,
        });

        const elapsed = Math.round(performance.now() - startTime);
        const data = await response.json();

        if (!response.ok) {
            // Error from API
            const errorMsg = data.error?.message || data.error || 'Measurement failed';
            throw new Error(errorMsg);
        }

        // Success!
        lastResults = data;
        badge.textContent = 'Complete';
        badge.className = 'preview__badge preview__badge--done';

        hideLoading();
        displayResults(data, elapsed);
        showToast(`Measurement complete in ${elapsed}ms`, 'success');

        document.getElementById('btnExport').disabled = false;

    } catch (err) {
        badge.textContent = 'Error';
        badge.className = 'preview__badge preview__badge--error';

        hideLoading();
        showError(err.message);
        showToast(err.message, 'error');

    } finally {
        isProcessing = false;
        btnMeasure.disabled = false;
        btnMeasure.innerHTML = '<span class="btn__icon">🧠</span><span>Analyze Face</span>';
    }
}


// ═══════════════════════════════════════════════════════════════
//  RESULTS DISPLAY
// ═══════════════════════════════════════════════════════════════

function displayResults(data, elapsed) {
    const grid = document.getElementById('resultsGrid');
    const metadataSection = document.getElementById('metadataSection');
    const metadataGrid = document.getElementById('metadataGrid');
    const empty = document.getElementById('resultsEmpty');

    empty.classList.add('hidden');

    // Build measurement cards
    let cardsHTML = '';
    MEASUREMENTS.forEach((m, idx) => {
        const value = data[m.key];
        if (value !== undefined && value !== null) {
            cardsHTML += `
        <div class="result-card ${m.cssClass}" style="animation-delay: ${idx * 0.06}s" title="${m.description}">
          <div class="result-card__label">
            <span class="result-card__icon">${m.icon}</span>
            ${m.label}
          </div>
          <div class="result-card__value">
            ${value}<span class="result-card__unit">mm</span>
          </div>
        </div>
      `;
        }
    });

    grid.innerHTML = cardsHTML;
    grid.classList.remove('hidden');

    // Animate cards in
    const cards = grid.querySelectorAll('.result-card');
    cards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(12px)';
        setTimeout(() => {
            card.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, i * 80);
    });

    // Display metadata
    if (data._metadata) {
        const md = data._metadata;
        const items = [
            { key: 'Landmarks', value: md.landmarks_detected || '478' },
            { key: 'Head Tilt', value: `${md.head_tilt_degrees || 0}°` },
            { key: 'Calibration', value: md.calibration_method || 'iris' },
            { key: 'px/mm', value: md.pixels_per_mm || '—' },
            { key: 'Confidence', value: md.confidence || '—' },
            { key: 'Processing', value: `${md.processing_time_ms || elapsed}ms` },
            { key: 'API Time', value: `${md.api_response_time_ms || elapsed}ms` },
            { key: 'Head Yaw', value: `${md.head_yaw_degrees || 0}°` },
        ];

        metadataGrid.innerHTML = items.map((item) => `
      <div class="metadata__item">
        <span class="metadata__key">${item.key}</span>
        <span class="metadata__value">${item.value}</span>
      </div>
    `).join('');

        metadataSection.classList.remove('hidden');
    }
}

function hideResults() {
    document.getElementById('resultsGrid').classList.add('hidden');
    document.getElementById('metadataSection').classList.add('hidden');
}

function showEmptyState() {
    document.getElementById('resultsEmpty').classList.remove('hidden');
}

function showLoading() {
    document.getElementById('resultsEmpty').classList.add('hidden');
    document.getElementById('resultsLoading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('resultsLoading').classList.add('hidden');
}

function showError(message) {
    const errorDisplay = document.getElementById('errorDisplay');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorDisplay.classList.remove('hidden');
    document.getElementById('resultsEmpty').classList.add('hidden');
}

function hideError() {
    document.getElementById('errorDisplay').classList.add('hidden');
}


// ═══════════════════════════════════════════════════════════════
//  SYSTEM HEALTH
// ═══════════════════════════════════════════════════════════════

async function checkSystemStatus() {
    const statusEl = document.getElementById('systemStatus');

    try {
        const res = await fetch(ENDPOINTS.health);
        const data = await res.json();

        if (data.status === 'healthy') {
            statusEl.innerHTML = `
        <span class="status-dot status-dot--healthy"></span>
        <span class="status-text">System Online</span>
      `;
        } else {
            statusEl.innerHTML = `
        <span class="status-dot status-dot--error"></span>
        <span class="status-text">Degraded</span>
      `;
        }
    } catch (err) {
        console.error('[FaceScan AI] Health check failed:', err);
        statusEl.innerHTML = `
      <span class="status-dot status-dot--error"></span>
      <span class="status-text">Offline</span>
    `;
    }
}

async function checkHealth() {
    const modal = document.getElementById('healthModal');
    const body = document.getElementById('healthModalBody');

    modal.classList.remove('hidden');
    body.innerHTML = '<div class="spinner"></div><p style="text-align:center;color:var(--text-secondary)">Running diagnostics...</p>';

    try {
        const res = await fetch(ENDPOINTS.healthDetailed);
        const data = await res.json();

        let html = '';

        // Overall status
        const statusClass = data.status === 'healthy' ? 'health-item__badge--ok' : 'health-item__badge--error';
        html += `
      <div class="health-item">
        <span class="health-item__label">🏥 Overall Status</span>
        <span class="health-item__badge ${statusClass}">${data.status === 'healthy' ? '✓ Healthy' : '✗ ' + data.status}</span>
      </div>
    `;

        // Node.js
        if (data.checks?.node) {
            html += `
        <div class="health-item">
          <span class="health-item__label">🟢 Node.js</span>
          <span class="health-item__value">${data.checks.node.version}</span>
        </div>
      `;
        }

        // Python
        if (data.checks?.python) {
            const py = data.checks.python;
            const pyClass = py.status === 'ok' ? 'health-item__badge--ok' : 'health-item__badge--error';
            html += `
        <div class="health-item">
          <span class="health-item__label">🐍 Python</span>
          <span class="health-item__badge ${pyClass}">${py.status === 'ok' ? '✓ OK' : '✗ Error'}</span>
        </div>
      `;

            // Dependencies
            if (py.dependencies) {
                for (const [name, info] of Object.entries(py.dependencies)) {
                    const depClass = info.installed ? 'health-item__badge--ok' : 'health-item__badge--error';
                    html += `
            <div class="health-item" style="margin-left: 20px;">
              <span class="health-item__label">📦 ${name}</span>
              <div style="display:flex;align-items:center;gap:8px;">
                <span class="health-item__value">${info.version || '—'}</span>
                <span class="health-item__badge ${depClass}">${info.installed ? '✓' : '✗'}</span>
              </div>
            </div>
          `;
                }
            }
        }

        // Filesystem
        if (data.checks?.filesystem) {
            const fs = data.checks.filesystem;
            html += `
        <div class="health-item">
          <span class="health-item__label">📁 Upload Directory</span>
          <span class="health-item__badge ${fs.upload_directory === 'ok' ? 'health-item__badge--ok' : 'health-item__badge--error'}">${fs.upload_directory === 'ok' ? '✓ OK' : '✗ Missing'}</span>
        </div>
        <div class="health-item">
          <span class="health-item__label">🧠 AI Engine Script</span>
          <span class="health-item__badge ${fs.python_script === 'ok' ? 'health-item__badge--ok' : 'health-item__badge--error'}">${fs.python_script === 'ok' ? '✓ OK' : '✗ Missing'}</span>
        </div>
      `;
        }

        // Uptime and version
        html += `
      <div class="health-item" style="margin-top:12px; border-top: 1px solid var(--border-color); padding-top: 16px;">
        <span class="health-item__label">⏱️ Uptime</span>
        <span class="health-item__value">${formatUptime(data.uptime)}</span>
      </div>
      <div class="health-item">
        <span class="health-item__label">📌 Version</span>
        <span class="health-item__value">${data.version || '1.0.0'}</span>
      </div>
    `;

        body.innerHTML = html;

    } catch (err) {
        body.innerHTML = `
      <div style="text-align:center; padding:24px;">
        <p style="font-size:2rem; margin-bottom:12px;">❌</p>
        <p style="color:var(--accent-rose); font-weight:600;">Connection Failed</p>
        <p style="color:var(--text-secondary); font-size:0.85rem; margin-top:8px;">${err.message}</p>
      </div>
    `;
    }
}

function closeHealthModal() {
    document.getElementById('healthModal').classList.add('hidden');
}

// Close modal on overlay click
document.addEventListener('click', (e) => {
    if (e.target.id === 'healthModal') {
        closeHealthModal();
    }
});

// Close modal on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeHealthModal();
    }
});


// ═══════════════════════════════════════════════════════════════
//  EXPORT
// ═══════════════════════════════════════════════════════════════

function exportResults() {
    if (!lastResults) {
        showToast('No results to export.', 'error');
        return;
    }

    const jsonStr = JSON.stringify(lastResults, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `facial-measurements-${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(url);
    showToast('Measurements exported as JSON', 'success');
}


// ═══════════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };

    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `
    <span class="toast__icon">${icons[type]}</span>
    <span class="toast__message">${message}</span>
  `;

    container.appendChild(toast);

    // Auto-dismiss
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}


// ═══════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════

function formatUptime(seconds) {
    if (!seconds) return '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}
