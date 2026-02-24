/**
 * ═══════════════════════════════════════════════════════════════
 *  Python Bridge Service
 *  
 *  Handles communication between Node.js and the Python AI engine
 *  via child_process spawning.
 * ═══════════════════════════════════════════════════════════════
 */

const { spawn } = require('child_process');
const path = require('path');
const config = require('../config');

class PythonBridgeService {
    constructor() {
        this.pythonPath = config.PYTHON_EXECUTABLE;
        this.scriptPath = config.PYTHON_SCRIPT;
        this.timeout = config.PYTHON_TIMEOUT;
    }

    /**
     * Execute the Python face measurement engine.
     * 
     * @param {string} imagePath - Absolute path to the image file
     * @param {number|null} referencePixels - Optional reference object width in pixels
     * @returns {Promise<Object>} - Parsed JSON result from Python
     */
    async execute(imagePath, referencePixels = null) {
        return new Promise((resolve, reject) => {
            const startTime = Date.now();

            // Build arguments
            const args = [this.scriptPath, imagePath];
            if (referencePixels) {
                args.push(String(referencePixels));
            }

            console.log(`[PythonBridge] Executing: ${this.pythonPath} ${args.join(' ')}`);

            // Spawn Python process
            const pythonProcess = spawn(this.pythonPath, args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env },
                timeout: this.timeout,
            });

            let stdoutData = '';
            let stderrData = '';

            // Collect stdout
            pythonProcess.stdout.on('data', (data) => {
                stdoutData += data.toString();
            });

            // Collect stderr
            pythonProcess.stderr.on('data', (data) => {
                stderrData += data.toString();
            });

            // Timeout handler
            const timeoutId = setTimeout(() => {
                pythonProcess.kill('SIGKILL');
                reject(new PythonBridgeError(
                    'Python process timed out',
                    'PYTHON_TIMEOUT',
                    408
                ));
            }, this.timeout);

            // Process exit handler
            pythonProcess.on('close', (code) => {
                clearTimeout(timeoutId);
                const elapsed = Date.now() - startTime;
                console.log(`[PythonBridge] Process exited with code ${code} in ${elapsed}ms`);

                if (stderrData) {
                    console.warn(`[PythonBridge] stderr: ${stderrData.substring(0, 500)}`);
                }

                // Parse stdout JSON
                try {
                    if (!stdoutData.trim()) {
                        reject(new PythonBridgeError(
                            'No output from Python engine',
                            'PYTHON_NO_OUTPUT',
                            500
                        ));
                        return;
                    }

                    // Find the JSON object in stdout (skip any warnings/logs)
                    const jsonMatch = stdoutData.match(/\{[\s\S]*\}/);
                    if (!jsonMatch) {
                        reject(new PythonBridgeError(
                            `Invalid output from Python engine: ${stdoutData.substring(0, 200)}`,
                            'PYTHON_INVALID_OUTPUT',
                            500
                        ));
                        return;
                    }

                    const result = JSON.parse(jsonMatch[0]);
                    result._processingTime = elapsed;
                    resolve(result);

                } catch (parseError) {
                    reject(new PythonBridgeError(
                        `Failed to parse Python output: ${parseError.message}. Raw output: ${stdoutData.substring(0, 200)}`,
                        'PYTHON_PARSE_ERROR',
                        500
                    ));
                }
            });

            // Process error handler
            pythonProcess.on('error', (err) => {
                clearTimeout(timeoutId);

                if (err.code === 'ENOENT') {
                    reject(new PythonBridgeError(
                        `Python executable not found at "${this.pythonPath}". Please install Python and ensure it's in your PATH.`,
                        'PYTHON_NOT_FOUND',
                        500
                    ));
                } else {
                    reject(new PythonBridgeError(
                        `Failed to start Python process: ${err.message}`,
                        'PYTHON_SPAWN_ERROR',
                        500
                    ));
                }
            });
        });
    }

    /**
     * Check if the Python engine and dependencies are properly installed.
     * @returns {Promise<Object>} - Status check result
     */
    async healthCheck() {
        return new Promise((resolve, reject) => {
            const checkScript = `
import sys
import json
status = {"python_version": sys.version}
try:
    import mediapipe
    status["mediapipe_version"] = mediapipe.__version__
    status["mediapipe"] = True
except ImportError:
    status["mediapipe"] = False
try:
    import cv2
    status["opencv_version"] = cv2.__version__
    status["opencv"] = True
except ImportError:
    status["opencv"] = False
try:
    import numpy
    status["numpy_version"] = numpy.__version__
    status["numpy"] = True
except ImportError:
    status["numpy"] = False
status["all_ok"] = all([status.get("mediapipe"), status.get("opencv"), status.get("numpy")])
print(json.dumps(status))
`;

            const proc = spawn(this.pythonPath, ['-c', checkScript], {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: 15000,
            });

            let output = '';
            proc.stdout.on('data', (d) => output += d.toString());
            proc.stderr.on('data', () => { }); // Ignore warnings

            proc.on('close', () => {
                try {
                    const jsonMatch = output.match(/\{[\s\S]*\}/);
                    resolve(jsonMatch ? JSON.parse(jsonMatch[0]) : { all_ok: false, error: 'No output' });
                } catch {
                    resolve({ all_ok: false, error: 'Parse failed' });
                }
            });

            proc.on('error', (err) => {
                resolve({ all_ok: false, error: err.message });
            });
        });
    }
}


/**
 * Custom error class for Python bridge errors
 */
class PythonBridgeError extends Error {
    constructor(message, code = 'PYTHON_ERROR', statusCode = 500) {
        super(message);
        this.name = 'PythonBridgeError';
        this.code = code;
        this.statusCode = statusCode;
    }
}

module.exports = new PythonBridgeService();
