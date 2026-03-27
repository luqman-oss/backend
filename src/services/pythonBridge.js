/**
 * ═══════════════════════════════════════════════════════════════
 *  Python Bridge Service — Persistent Worker Mode
 *  
 *  Keeps a single Python process alive for all requests.
 *  First request: cold start (~10-20s for imports + model load).
 *  Subsequent requests: ~2-5s (model already loaded in memory).
 *  
 *  Falls back to one-shot spawn if the worker fails.
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

        // Persistent worker state
        this.worker = null;
        this.workerReady = false;
        this.workerStarting = false;
        this.pendingRequests = [];   // Queue of { resolve, reject, timer }
        this.stdoutBuffer = '';
    }

    /**
     * Start the persistent Python worker process.
     * Returns a promise that resolves when the worker signals "ready".
     */
    async _startWorker() {
        if (this.workerReady && this.worker) return;
        if (this.workerStarting) {
            // Wait for the in-progress startup
            return new Promise((resolve, reject) => {
                const check = setInterval(() => {
                    if (this.workerReady) { clearInterval(check); resolve(); }
                    if (!this.workerStarting && !this.workerReady) { clearInterval(check); reject(new Error('Worker failed to start')); }
                }, 200);
                setTimeout(() => { clearInterval(check); reject(new Error('Worker startup timeout')); }, this.timeout);
            });
        }

        this.workerStarting = true;
        console.log(`[PythonBridge] Starting persistent worker: ${this.pythonPath} ${this.scriptPath} --worker`);

        return new Promise((resolve, reject) => {
            this.worker = spawn(this.pythonPath, [this.scriptPath, '--worker'], {
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env },
            });

            this.stdoutBuffer = '';

            // Handle stdout — each line is a JSON response
            this.worker.stdout.on('data', (data) => {
                this.stdoutBuffer += data.toString();
                
                // Process complete lines
                let newlineIdx;
                while ((newlineIdx = this.stdoutBuffer.indexOf('\n')) !== -1) {
                    const line = this.stdoutBuffer.substring(0, newlineIdx).trim();
                    this.stdoutBuffer = this.stdoutBuffer.substring(newlineIdx + 1);
                    
                    if (!line) continue;
                    
                    try {
                        const msg = JSON.parse(line);
                        
                        // Ready signal from worker
                        if (msg.ready) {
                            this.workerReady = true;
                            this.workerStarting = false;
                            console.log('[PythonBridge] Worker ready.');
                            resolve();
                            continue;
                        }
                        
                        // Response for a pending request
                        if (this.pendingRequests.length > 0) {
                            const req = this.pendingRequests.shift();
                            clearTimeout(req.timer);
                            req.resolve(msg);
                        }
                    } catch (e) {
                        console.warn('[PythonBridge] Failed to parse worker output:', line.substring(0, 200));
                    }
                }
            });

            // Log stderr (Python warnings, diagnostics)
            this.worker.stderr.on('data', (data) => {
                const msg = data.toString().trim();
                if (msg) console.log(`[PythonWorker] ${msg}`);
            });

            // Handle worker crash
            this.worker.on('close', (code) => {
                console.warn(`[PythonBridge] Worker exited with code ${code}`);
                this.worker = null;
                this.workerReady = false;
                this.workerStarting = false;
                
                // Reject all pending requests
                while (this.pendingRequests.length > 0) {
                    const req = this.pendingRequests.shift();
                    clearTimeout(req.timer);
                    req.reject(new PythonBridgeError('Worker process died', 'PYTHON_WORKER_DIED', 500));
                }
            });

            this.worker.on('error', (err) => {
                console.error(`[PythonBridge] Worker spawn error:`, err.message);
                this.worker = null;
                this.workerReady = false;
                this.workerStarting = false;
                reject(new PythonBridgeError(
                    `Failed to start Python worker: ${err.message}`,
                    err.code === 'ENOENT' ? 'PYTHON_NOT_FOUND' : 'PYTHON_SPAWN_ERROR',
                    500
                ));
            });

            // Startup timeout
            setTimeout(() => {
                if (!this.workerReady) {
                    this.workerStarting = false;
                    if (this.worker) { this.worker.kill(); this.worker = null; }
                    reject(new PythonBridgeError('Worker startup timed out', 'PYTHON_TIMEOUT', 408));
                }
            }, this.timeout);
        });
    }

    /**
     * Execute the Python face measurement engine via persistent worker.
     */
    async execute(imagePath, referencePixels = null) {
        const startTime = Date.now();

        // Ensure worker is running
        try {
            await this._startWorker();
        } catch (err) {
            console.warn('[PythonBridge] Worker unavailable, falling back to one-shot:', err.message);
            return this._executeOneShot(imagePath, referencePixels);
        }

        // Send request to worker
        return new Promise((resolve, reject) => {
            const command = JSON.stringify({
                image_path: imagePath,
                reference_pixels: referencePixels,
            }) + '\n';

            const timer = setTimeout(() => {
                // Remove from pending queue
                const idx = this.pendingRequests.findIndex(r => r.timer === timer);
                if (idx !== -1) this.pendingRequests.splice(idx, 1);
                reject(new PythonBridgeError('Worker request timed out', 'PYTHON_TIMEOUT', 408));
            }, this.timeout);

            this.pendingRequests.push({ resolve: (result) => {
                result._processingTime = Date.now() - startTime;
                console.log(`[PythonBridge] Worker responded in ${result._processingTime}ms`);
                resolve(result);
            }, reject, timer });

            this.worker.stdin.write(command);
        });
    }

    /**
     * Fallback: one-shot spawn (original behavior).
     */
    async _executeOneShot(imagePath, referencePixels = null) {
        return new Promise((resolve, reject) => {
            const startTime = Date.now();
            const args = [this.scriptPath, imagePath];
            if (referencePixels) args.push(String(referencePixels));

            console.log(`[PythonBridge] One-shot: ${this.pythonPath} ${args.join(' ')}`);

            const proc = spawn(this.pythonPath, args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env },
                timeout: this.timeout,
            });

            let stdoutData = '';
            let stderrData = '';

            proc.stdout.on('data', (d) => stdoutData += d.toString());
            proc.stderr.on('data', (d) => stderrData += d.toString());

            const timeoutId = setTimeout(() => {
                proc.kill('SIGKILL');
                reject(new PythonBridgeError('Python process timed out', 'PYTHON_TIMEOUT', 408));
            }, this.timeout);

            proc.on('close', (code) => {
                clearTimeout(timeoutId);
                const elapsed = Date.now() - startTime;
                console.log(`[PythonBridge] One-shot exited code ${code} in ${elapsed}ms`);
                if (stderrData) console.warn(`[PythonBridge] stderr: ${stderrData.substring(0, 500)}`);

                try {
                    if (!stdoutData.trim()) {
                        reject(new PythonBridgeError('No output from Python engine', 'PYTHON_NO_OUTPUT', 500));
                        return;
                    }
                    const jsonMatch = stdoutData.match(/\{[\s\S]*\}/);
                    if (!jsonMatch) {
                        reject(new PythonBridgeError(`Invalid output: ${stdoutData.substring(0, 200)}`, 'PYTHON_INVALID_OUTPUT', 500));
                        return;
                    }
                    const result = JSON.parse(jsonMatch[0]);
                    result._processingTime = elapsed;
                    resolve(result);
                } catch (parseError) {
                    reject(new PythonBridgeError(`Parse failed: ${parseError.message}`, 'PYTHON_PARSE_ERROR', 500));
                }
            });

            proc.on('error', (err) => {
                clearTimeout(timeoutId);
                reject(new PythonBridgeError(
                    err.code === 'ENOENT'
                        ? `Python not found at "${this.pythonPath}"`
                        : `Spawn error: ${err.message}`,
                    err.code === 'ENOENT' ? 'PYTHON_NOT_FOUND' : 'PYTHON_SPAWN_ERROR',
                    500
                ));
            });
        });
    }

    /**
     * Check if the Python engine and dependencies are properly installed.
     */
    async healthCheck() {
        // Quick check: ask the worker if it's alive
        if (this.workerReady && this.worker) {
            return new Promise((resolve) => {
                const timer = setTimeout(() => {
                    resolve({ all_ok: false, error: 'Worker health check timed out' });
                }, 5000);

                this.pendingRequests.push({
                    resolve: (result) => {
                        clearTimeout(timer);
                        resolve({ all_ok: true, worker_mode: true, ...result });
                    },
                    reject: () => {
                        clearTimeout(timer);
                        resolve({ all_ok: false, error: 'Worker health check failed' });
                    },
                    timer,
                });

                this.worker.stdin.write(JSON.stringify({ command: 'health' }) + '\n');
            });
        }

        // Full check via one-shot
        return new Promise((resolve) => {
            const checkScript = `
import sys, json
status = {"python_version": sys.version}
try:
    import mediapipe; status["mediapipe_version"] = mediapipe.__version__; status["mediapipe"] = True
except ImportError: status["mediapipe"] = False
try:
    import cv2; status["opencv_version"] = cv2.__version__; status["opencv"] = True
except ImportError: status["opencv"] = False
try:
    import numpy; status["numpy_version"] = numpy.__version__; status["numpy"] = True
except ImportError: status["numpy"] = False
status["all_ok"] = all([status.get("mediapipe"), status.get("opencv"), status.get("numpy")])
print(json.dumps(status))
`;
            const proc = spawn(this.pythonPath, ['-c', checkScript], {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: 15000,
            });

            let output = '';
            proc.stdout.on('data', (d) => output += d.toString());
            proc.stderr.on('data', () => {});

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

    /**
     * Gracefully shut down the worker process.
     */
    async shutdown() {
        if (this.worker) {
            console.log('[PythonBridge] Shutting down worker...');
            try {
                this.worker.stdin.write(JSON.stringify({ command: 'exit' }) + '\n');
            } catch (e) { /* ignore */ }
            setTimeout(() => {
                if (this.worker) { this.worker.kill(); this.worker = null; }
            }, 3000);
        }
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
