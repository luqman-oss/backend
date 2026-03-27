/**
 * ═══════════════════════════════════════════════════════════════
 *  FaceScan AI — Frontend Application
 * 
 *  Camera capture, drag-and-drop upload, and measurement display
 *  Communicates with POST /api/measure endpoint
 * ═══════════════════════════════════════════════════════════════
 */

(function () {
    'use strict';

    // ─────────────────────────────────────────────────────────
    //  DOM Elements
    // ─────────────────────────────────────────────────────────

    const $ = (sel) => document.querySelector(sel);
    const cameraPreview    = $('#cameraPreview');
    const cameraCanvas     = $('#cameraCanvas');
    const faceDetectionCanvas = $('#faceDetectionCanvas');
    const cameraOverlay    = $('#cameraOverlay');
    const cameraPlaceholder = $('#cameraPlaceholder');
    const btnStartCamera   = $('#btnStartCamera');
    const btnCapture       = $('#btnCapture');
    const btnStopCamera    = $('#btnStopCamera');
    const dropZone         = $('#dropZone');
    const fileInput        = $('#fileInput');
    const previewContainer = $('#previewContainer');
    const previewImage     = $('#previewImage');
    const btnMeasure       = $('#btnMeasure');
    const btnClear         = $('#btnClear');
    const btnRetry         = $('#btnRetry');
    const loadingState     = $('#loadingState');
    const emptyState       = $('#emptyState');
    const errorState       = $('#errorState');
    const errorMessage     = $('#errorMessage');
    const resultsState     = $('#resultsState');
    const confidenceValue  = $('#confidenceValue');
    const confidenceBadge  = $('#confidenceBadge');
    const measurementGrid  = $('#measurementGrid');
    const metadataGrid     = $('#metadataGrid');

    // ─────────────────────────────────────────────────────────
    //  State
    // ─────────────────────────────────────────────────────────

    let cameraStream = null;
    let capturedBlob = null;   // Blob from camera capture
    let selectedFile = null;   // File from drag-drop / browse
    let faceDetection = null;
    let faceDetectionCtx = null;
    let isDetectingFace = false;

    const API_BASE = window.location.origin;

    // Measurement display configuration
    const MEASUREMENTS = [
        { key: 'pupillary_distance_mm', label: 'Pupillary Distance',  icon: '👁️', unit: 'mm', color: '#6C63FF' },
        { key: 'face_width_mm',         label: 'Face Width',          icon: '↔️', unit: 'mm', color: '#00D2FF' },
        { key: 'face_height_mm',        label: 'Face Height',         icon: '↕️', unit: 'mm', color: '#FF6B6B' },
        { key: 'eye_width_mm',          label: 'Eye Width',           icon: '👁',  unit: 'mm', color: '#4ECDC4' },
        { key: 'eye_height_mm',         label: 'Eye Height',          icon: '🔘', unit: 'mm', color: '#FFE66D' },
        { key: 'bridge_width_mm',       label: 'Bridge Width',        icon: '🔺', unit: 'mm', color: '#A8E6CF' },
        { key: 'forehead_width_mm',     label: 'Forehead Width',      icon: '📏', unit: 'mm', color: '#FF8A5C' },
        { key: 'side_length_mm',        label: 'Side/Arm Length',     icon: '📐', unit: 'mm', color: '#B8B8FF' },
    ];

    // ─────────────────────────────────────────────────────────
    //  Face Detection
    // ─────────────────────────────────────────────────────────

    function initializeFaceDetection() {
        faceDetection = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
            }
        });

        faceDetection.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceDetection.onResults(onFaceDetectionResults);
    }

    function onFaceDetectionResults(results) {
        if (!faceDetectionCtx) return;

        // Clear canvas
        faceDetectionCtx.clearRect(0, 0, faceDetectionCanvas.width, faceDetectionCanvas.height);

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];
            
            // Draw face mesh connections
            drawFaceMesh(landmarks);
            
            // Draw key measurements
            drawKeyMeasurements(landmarks);
        }
    }

    function drawFaceMesh(landmarks) {
        const canvas = faceDetectionCanvas;
        const ctx = faceDetectionCtx;
        
        // Draw selected face connections
        const connections = [
            // Face oval
            [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 340], [340, 346], [346, 347], [347, 348], [348, 349], [349, 350], [350, 451], [451, 452], [452, 453], [453, 464], [464, 435], [435, 410], [410, 287], [287, 273], [273, 335], [335, 406], [406, 313], [313, 18], [18, 83], [83, 182], [182, 106], [106, 43], [43, 57], [57, 186], [186, 92], [92, 165], [165, 167], [167, 164], [164, 393], [393, 391], [391, 322], [322, 410], [410, 287], [287, 273], [273, 335], [335, 406], [406, 313], [313, 18], [18, 83], [83, 182],
            // Eyes
            [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133],
            [362, 398], [398, 384], [384, 385], [385, 386], [386, 387], [387, 388], [388, 466],
            // Eyebrows
            [70, 63], [63, 105], [105, 66], [66, 107], [107, 55], [55, 65], [65, 52], [52, 53],
            [300, 293], [293, 334], [334, 296], [296, 336], [336, 285], [285, 295], [295, 282], [282, 283],
            // Nose
            [1, 2], [2, 5], [5, 4], [4, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 308], [308, 324], [324, 318], [318, 402], [402, 317], [317, 14], [14, 87], [87, 178], [178, 88], [88, 95],
            // Mouth
            [61, 84], [84, 17], [17, 314], [314, 405], [405, 320], [320, 307], [307, 375], [375, 321], [321, 308], [308, 324], [324, 318], [318, 402], [402, 317], [317, 14], [14, 87], [87, 178], [178, 88], [88, 95],
            [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 308]
        ];

        ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
        ctx.lineWidth = 1;

        connections.forEach(([start, end]) => {
            if (landmarks[start] && landmarks[end]) {
                const startPoint = landmarks[start];
                const endPoint = landmarks[end];
                
                ctx.beginPath();
                ctx.moveTo(startPoint.x * canvas.width, startPoint.y * canvas.height);
                ctx.lineTo(endPoint.x * canvas.width, endPoint.y * canvas.height);
                ctx.stroke();
            }
        });
    }

    function drawKeyMeasurements(landmarks) {
        const canvas = faceDetectionCanvas;
        const ctx = faceDetectionCtx;
        
        // Key landmark indices for measurements
        const keyPoints = {
            leftIris: landmarks[468],
            rightIris: landmarks[473],
            leftEyeInner: landmarks[133],
            leftEyeOuter: landmarks[33],
            rightEyeInner: landmarks[362],
            rightEyeOuter: landmarks[263],
            leftCheek: landmarks[234],
            rightCheek: landmarks[454],
            forehead: landmarks[10],
            chin: landmarks[152]
        };

        // Draw measurement lines
        ctx.strokeStyle = '#6C63FF';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        // Pupillary Distance
        if (keyPoints.leftIris && keyPoints.rightIris) {
            ctx.beginPath();
            ctx.moveTo(keyPoints.leftIris.x * canvas.width, keyPoints.leftIris.y * canvas.height);
            ctx.lineTo(keyPoints.rightIris.x * canvas.width, keyPoints.rightIris.y * canvas.height);
            ctx.stroke();
            
            // Label
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Arial';
            ctx.fillText('PD', 
                (keyPoints.leftIris.x * canvas.width + keyPoints.rightIris.x * canvas.width) / 2,
                (keyPoints.leftIris.y * canvas.height + keyPoints.rightIris.y * canvas.height) / 2 - 10
            );
        }
        
        // Face Width
        ctx.strokeStyle = '#00D2FF';
        if (keyPoints.leftCheek && keyPoints.rightCheek) {
            ctx.beginPath();
            ctx.moveTo(keyPoints.leftCheek.x * canvas.width, keyPoints.leftCheek.y * canvas.height);
            ctx.lineTo(keyPoints.rightCheek.x * canvas.width, keyPoints.rightCheek.y * canvas.height);
            ctx.stroke();
            
            // Label
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Arial';
            ctx.fillText('Width', 
                (keyPoints.leftCheek.x * canvas.width + keyPoints.rightCheek.x * canvas.width) / 2,
                (keyPoints.leftCheek.y * canvas.height + keyPoints.rightCheek.y * canvas.height) / 2 - 10
            );
        }
        
        ctx.setLineDash([]);
        
        // Draw key landmarks
        ctx.fillStyle = '#ff00ff';
        Object.values(keyPoints).forEach(point => {
            if (point) {
                ctx.beginPath();
                ctx.arc(point.x * canvas.width, point.y * canvas.height, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    }

    function setupFaceDetectionCanvas() {
        const rect = cameraPreview.getBoundingClientRect();
        faceDetectionCanvas.width = rect.width;
        faceDetectionCanvas.height = rect.height;
        faceDetectionCtx = faceDetectionCanvas.getContext('2d');
    }

    // ─────────────────────────────────────────────────────────
    //  Camera
    // ─────────────────────────────────────────────────────────

    async function startCamera() {
        try {
            // Initialize face detection if not already done
            if (!faceDetection) {
                initializeFaceDetection();
            }

            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
                audio: false,
            });

            cameraPreview.srcObject = cameraStream;
            cameraPreview.style.display = 'block';
            cameraPlaceholder.style.display = 'none';
            cameraOverlay.style.display = 'flex';

            // Setup face detection canvas
            setupFaceDetectionCanvas();

            // Start face detection
            if (faceDetection) {
                const camera = new Camera(cameraPreview, {
                    onFrame: async () => {
                        if (faceDetection && isDetectingFace) {
                            await faceDetection.send({image: cameraPreview});
                        }
                    },
                    width: 1280,
                    height: 720
                });
                camera.start();
                isDetectingFace = true;
            }

            btnStartCamera.disabled = true;
            btnCapture.disabled = false;
            btnStopCamera.disabled = false;
        } catch (err) {
            console.error('Camera error:', err);
            let msg = 'Could not access camera.';
            if (err.name === 'NotAllowedError') {
                msg = 'Camera permission denied. Please allow camera access and try again.';
            } else if (err.name === 'NotFoundError') {
                msg = 'No camera found on this device.';
            } else if (err.name === 'NotReadableError') {
                msg = 'Camera is in use by another application.';
            }
            showError(msg);
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach((t) => t.stop());
            cameraStream = null;
        }
        
        // Stop face detection
        isDetectingFace = false;
        
        // Clear face detection canvas
        if (faceDetectionCtx) {
            faceDetectionCtx.clearRect(0, 0, faceDetectionCanvas.width, faceDetectionCanvas.height);
        }

        cameraPreview.srcObject = null;
        cameraPreview.style.display = 'none';
        cameraPlaceholder.style.display = 'flex';
        cameraOverlay.style.display = 'none';

        btnStartCamera.disabled = false;
        btnCapture.disabled = true;
        btnStopCamera.disabled = true;
    }

    function captureFrame() {
        if (!cameraStream) return;

        const vw = cameraPreview.videoWidth;
        const vh = cameraPreview.videoHeight;
        cameraCanvas.width = vw;
        cameraCanvas.height = vh;

        const ctx = cameraCanvas.getContext('2d');
        ctx.drawImage(cameraPreview, 0, 0, vw, vh);

        cameraCanvas.toBlob((blob) => {
            capturedBlob = blob;
            selectedFile = null;

            // Show preview
            const url = URL.createObjectURL(blob);
            previewImage.src = url;
            previewContainer.style.display = 'block';

            // Auto-measure
            sendMeasurement(blob, 'camera_capture.jpg');
        }, 'image/jpeg', 0.95);
    }

    // ─────────────────────────────────────────────────────────
    //  Drag & Drop / File Browse
    // ─────────────────────────────────────────────────────────

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            showError('Please select a valid image file (JPEG, PNG, etc.).');
            return;
        }

        selectedFile = file;
        capturedBlob = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // Auto-measure
        sendMeasurement(file, file.name);
    }

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // ─────────────────────────────────────────────────────────
    //  API Communication
    // ─────────────────────────────────────────────────────────

    async function sendMeasurement(fileOrBlob, filename) {
        showLoading();

        const formData = new FormData();
        formData.append('image', fileOrBlob, filename || 'image.jpg');

        try {
            const response = await fetch(`${API_BASE}/api/measure`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle different error response formats
                let errorMsg = `Server error (${response.status})`;
                
                if (data.error) {
                    if (typeof data.error === 'string') {
                        errorMsg = data.error;
                    } else if (data.error.message) {
                        errorMsg = data.error.message;
                    } else {
                        errorMsg = JSON.stringify(data.error);
                    }
                } else if (data.message) {
                    errorMsg = data.message;
                }
                
                throw new Error(errorMsg);
            }

            showResults(data);
        } catch (err) {
            console.error('Measurement error:', err);
            showError(err.message || 'Failed to process image. Please try again.');
        }
    }

    // ─────────────────────────────────────────────────────────
    //  UI State Management
    // ─────────────────────────────────────────────────────────

    function hideAllStates() {
        loadingState.style.display = 'none';
        emptyState.style.display   = 'none';
        errorState.style.display   = 'none';
        resultsState.style.display = 'none';
    }

    function showLoading() {
        hideAllStates();
        loadingState.style.display = 'flex';
    }

    function showEmpty() {
        hideAllStates();
        emptyState.style.display = 'flex';
    }

    function showError(msg) {
        hideAllStates();
        errorMessage.textContent = msg;
        errorState.style.display = 'flex';
    }

    function showResults(data) {
        hideAllStates();
        resultsState.style.display = 'block';

        // Confidence badge
        const conf = data._metadata?.confidence_score || 'unknown';
        confidenceValue.textContent = conf;
        
        // Set CSS class based on confidence level
        let confClass = 'unknown';
        if (typeof conf === 'string' && conf.includes('%')) {
            const confValue = parseFloat(conf);
            if (confValue >= 85) confClass = 'high';
            else if (confValue >= 70) confClass = 'medium';
            else confClass = 'low';
        }
        confidenceBadge.className = 'confidence-badge confidence-' + confClass;

        // Measurement cards
        measurementGrid.innerHTML = '';
        MEASUREMENTS.forEach((m) => {
            const value = data[m.key];
            if (value === undefined || value === null) return;

            const card = document.createElement('div');
            card.className = 'measurement-card glass-inner';
            card.style.borderLeft = `4px solid ${m.color}`;
            card.innerHTML = `
                <div class="card-icon">${m.icon}</div>
                <div class="card-body">
                    <div class="card-label">${m.label}</div>
                    <div class="card-value">
                        <span class="value-number">${value.toFixed(2)}</span>
                        <span class="value-unit">${m.unit}</span>
                    </div>
                </div>
            `;
            measurementGrid.appendChild(card);
        });

        // Metadata
        metadataGrid.innerHTML = '';
        const meta = data._metadata || {};
        const metaItems = [
            { label: 'Landmarks',        value: meta.landmarks_detected || '—' },
            { label: 'Calibration',       value: meta.calibration_method || '—' },
            { label: 'Head Tilt',         value: meta.head_tilt_degrees != null ? meta.head_tilt_degrees.toFixed(1) + '°' : '—' },
            { label: 'Pixels/mm',         value: meta.pixels_per_mm != null ? meta.pixels_per_mm.toFixed(2) : '—' },
            { label: 'Processing Time',   value: meta.processing_time_ms != null ? meta.processing_time_ms + 'ms' : '—' },
            { label: 'Image Size',        value: meta.image_dimensions ? `${meta.image_dimensions.width}×${meta.image_dimensions.height}` : '—' },
        ];

        metaItems.forEach((item) => {
            const div = document.createElement('div');
            div.className = 'meta-item';
            div.innerHTML = `<span class="meta-label">${item.label}</span><span class="meta-value">${item.value}</span>`;
            metadataGrid.appendChild(div);
        });
    }

    function clearAll() {
        capturedBlob = null;
        selectedFile = null;
        previewContainer.style.display = 'none';
        previewImage.src = '';
        fileInput.value = '';
        showEmpty();
    }

    // ─────────────────────────────────────────────────────────
    //  Button Handlers
    // ─────────────────────────────────────────────────────────

    btnStartCamera.addEventListener('click', startCamera);
    btnStopCamera.addEventListener('click', stopCamera);
    btnCapture.addEventListener('click', captureFrame);

    btnMeasure.addEventListener('click', () => {
        if (capturedBlob) {
            sendMeasurement(capturedBlob, 'camera_capture.jpg');
        } else if (selectedFile) {
            sendMeasurement(selectedFile, selectedFile.name);
        }
    });

    btnClear.addEventListener('click', clearAll);
    btnRetry.addEventListener('click', clearAll);

    // ─────────────────────────────────────────────────────────
    //  Initialization
    // ─────────────────────────────────────────────────────────

    // Check API health on load
    (async function init() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            if (!res.ok) {
                console.warn('API health check failed:', res.status);
            } else {
                const data = await res.json();
                console.log('API Health:', data);
            }
        } catch (err) {
            console.warn('API not reachable:', err.message);
        }
    })();

})();
