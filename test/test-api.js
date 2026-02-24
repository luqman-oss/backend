/**
 * ═══════════════════════════════════════════════════════════════
 *  API Test Script
 *  
 *  Tests the facial measurement API endpoints.
 *  Usage: node test/test-api.js [image_path]
 * ═══════════════════════════════════════════════════════════════
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://127.0.0.1:3000';

// ─── Utility Functions ───────────────────────────────────────

function makeRequest(method, urlPath, options = {}) {
    return new Promise((resolve, reject) => {
        const url = new URL(urlPath, BASE_URL);

        const reqOptions = {
            hostname: url.hostname,
            port: url.port,
            path: url.pathname,
            method,
            headers: options.headers || {},
        };

        const req = http.request(reqOptions, (res) => {
            let data = '';
            res.on('data', (chunk) => (data += chunk));
            res.on('end', () => {
                try {
                    resolve({
                        status: res.statusCode,
                        data: JSON.parse(data),
                    });
                } catch {
                    resolve({ status: res.statusCode, data });
                }
            });
        });

        req.on('error', reject);

        if (options.body) {
            req.write(options.body);
        }

        req.end();
    });
}

function createMultipartBody(filePath, fieldName = 'image') {
    const boundary = '----FormBoundary' + Math.random().toString(36).substr(2);
    const fileName = path.basename(filePath);
    const fileContent = fs.readFileSync(filePath);

    const header = Buffer.from(
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="${fieldName}"; filename="${fileName}"\r\n` +
        `Content-Type: image/jpeg\r\n\r\n`
    );

    const footer = Buffer.from(`\r\n--${boundary}--\r\n`);

    return {
        body: Buffer.concat([header, fileContent, footer]),
        contentType: `multipart/form-data; boundary=${boundary}`,
    };
}

// ─── Tests ───────────────────────────────────────────────────

async function runTests() {
    console.log('');
    console.log('╔══════════════════════════════════════════════════╗');
    console.log('║    FACIAL MEASUREMENT API - TEST SUITE          ║');
    console.log('╚══════════════════════════════════════════════════╝');
    console.log('');

    let passed = 0;
    let failed = 0;

    // Test 1: Root endpoint
    try {
        console.log('📋 Test 1: GET /api (API Info)');
        const res = await makeRequest('GET', '/api');
        if (res.status === 200 && res.data.name === 'Facial Measurement API') {
            console.log('   ✅ PASSED\n');
            passed++;
        } else {
            console.log(`   ❌ FAILED: Unexpected response (${res.status})\n`);
            failed++;
        }
    } catch (err) {
        console.log(`   ❌ FAILED: ${err.message}\n`);
        failed++;
    }

    // Test 2: Health check
    try {
        console.log('📋 Test 2: GET /api/health');
        const res = await makeRequest('GET', '/api/health');
        if (res.status === 200 && res.data.status === 'healthy') {
            console.log('   ✅ PASSED\n');
            passed++;
        } else {
            console.log(`   ❌ FAILED: Unexpected response\n`);
            failed++;
        }
    } catch (err) {
        console.log(`   ❌ FAILED: ${err.message}\n`);
        failed++;
    }

    // Test 3: Detailed health check
    try {
        console.log('📋 Test 3: GET /api/health/detailed');
        const res = await makeRequest('GET', '/api/health/detailed');
        if (res.status === 200 || res.status === 503) {
            console.log(`   ✅ PASSED (status: ${res.data.status})`);
            if (res.data.checks && res.data.checks.python) {
                const py = res.data.checks.python;
                console.log(`   Python: ${py.version}`);
                console.log(`   MediaPipe: ${py.dependencies?.mediapipe?.installed ? '✅' : '❌'} ${py.dependencies?.mediapipe?.version || ''}`);
                console.log(`   OpenCV: ${py.dependencies?.opencv?.installed ? '✅' : '❌'} ${py.dependencies?.opencv?.version || ''}`);
                console.log(`   NumPy: ${py.dependencies?.numpy?.installed ? '✅' : '❌'} ${py.dependencies?.numpy?.version || ''}`);
            }
            console.log('');
            passed++;
        } else {
            console.log(`   ❌ FAILED\n`);
            failed++;
        }
    } catch (err) {
        console.log(`   ❌ FAILED: ${err.message}\n`);
        failed++;
    }

    // Test 4: POST /api/measure without image
    try {
        console.log('📋 Test 4: POST /api/measure (no image - should fail)');
        const res = await makeRequest('POST', '/api/measure');
        if (res.status === 400 || res.status === 500) {
            console.log(`   ✅ PASSED (correctly rejected with ${res.status})\n`);
            passed++;
        } else {
            console.log(`   ❌ FAILED: Expected 400, got ${res.status}\n`);
            failed++;
        }
    } catch (err) {
        console.log(`   ❌ FAILED: ${err.message}\n`);
        failed++;
    }

    // Test 5: POST /api/measure with image (if provided)
    const testImagePath = process.argv[2];
    if (testImagePath && fs.existsSync(testImagePath)) {
        try {
            console.log(`📋 Test 5: POST /api/measure (with image: ${path.basename(testImagePath)})`);
            const { body, contentType } = createMultipartBody(testImagePath);

            const res = await makeRequest('POST', '/api/measure', {
                headers: {
                    'Content-Type': contentType,
                    'Content-Length': body.length,
                },
                body,
            });

            if (res.status === 200 && res.data.pupillary_distance_mm) {
                console.log('   ✅ PASSED');
                console.log('   📏 Measurements:');
                console.log(`      PD:          ${res.data.pupillary_distance_mm} mm`);
                console.log(`      Face Width:  ${res.data.face_width_mm} mm`);
                console.log(`      Face Height: ${res.data.face_height_mm} mm`);
                console.log(`      Eye Width:   ${res.data.eye_width_mm} mm`);
                console.log(`      Eye Height:  ${res.data.eye_height_mm} mm`);
                console.log(`      Bridge:      ${res.data.bridge_width_mm} mm`);
                console.log(`      Forehead:    ${res.data.forehead_width_mm} mm`);
                console.log(`      Side Length: ${res.data.side_length_mm} mm`);
                if (res.data._metadata) {
                    console.log(`   ⏱️  Processing time: ${res.data._metadata.processing_time_ms}ms`);
                    console.log(`   📐 Calibration: ${res.data._metadata.calibration_method}`);
                }
                console.log('');
                passed++;
            } else {
                console.log(`   ❌ FAILED: ${JSON.stringify(res.data)}\n`);
                failed++;
            }
        } catch (err) {
            console.log(`   ❌ FAILED: ${err.message}\n`);
            failed++;
        }
    } else {
        console.log('📋 Test 5: SKIPPED (no test image provided)');
        console.log('   Usage: node test/test-api.js <path-to-face-image>\n');
    }

    // Test 6: 404 handler
    try {
        console.log('📋 Test 6: GET /nonexistent (404 handler)');
        const res = await makeRequest('GET', '/nonexistent');
        if (res.status === 404) {
            console.log('   ✅ PASSED\n');
            passed++;
        } else {
            console.log(`   ❌ FAILED: Expected 404, got ${res.status}\n`);
            failed++;
        }
    } catch (err) {
        console.log(`   ❌ FAILED: ${err.message}\n`);
        failed++;
    }

    // Summary
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log(`Results: ${passed} passed, ${failed} failed`);
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

    process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(console.error);
