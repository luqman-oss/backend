/**
 * ═══════════════════════════════════════════════════════════════
 *  Health Controller
 * ═══════════════════════════════════════════════════════════════
 */

const pythonBridge = require('../services/pythonBridge');
const config = require('../config');
const fs = require('fs');
const path = require('path');

class HealthController {
    /**
     * GET /api/health
     * 
     * Basic health check endpoint.
     */
    async basicHealth(req, res) {
        res.status(200).json({
            status: 'healthy',
            service: 'facial-measurement-api',
            version: '1.0.0',
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
        });
    }

    /**
     * GET /api/health/detailed
     * 
     * Detailed health check including Python dependencies.
     */
    async detailedHealth(req, res) {
        try {
            // Check Python environment
            const pythonStatus = await pythonBridge.healthCheck();

            // Check upload directory
            const uploadDirExists = fs.existsSync(config.UPLOAD_DIR);

            // Check Python script exists
            const scriptExists = fs.existsSync(config.PYTHON_SCRIPT);

            const allHealthy = pythonStatus.all_ok && uploadDirExists && scriptExists;

            res.status(allHealthy ? 200 : 503).json({
                status: allHealthy ? 'healthy' : 'degraded',
                service: 'facial-measurement-api',
                version: '1.0.0',
                timestamp: new Date().toISOString(),
                uptime: process.uptime(),
                checks: {
                    node: {
                        status: 'ok',
                        version: process.version,
                    },
                    python: {
                        status: pythonStatus.all_ok ? 'ok' : 'error',
                        version: pythonStatus.python_version || 'unknown',
                        dependencies: {
                            mediapipe: {
                                installed: pythonStatus.mediapipe || false,
                                version: pythonStatus.mediapipe_version || 'not installed',
                            },
                            opencv: {
                                installed: pythonStatus.opencv || false,
                                version: pythonStatus.opencv_version || 'not installed',
                            },
                            numpy: {
                                installed: pythonStatus.numpy || false,
                                version: pythonStatus.numpy_version || 'not installed',
                            },
                        },
                    },
                    filesystem: {
                        upload_directory: uploadDirExists ? 'ok' : 'missing',
                        python_script: scriptExists ? 'ok' : 'missing',
                    },
                },
            });

        } catch (err) {
            res.status(503).json({
                status: 'unhealthy',
                error: err.message,
            });
        }
    }
}

module.exports = new HealthController();
