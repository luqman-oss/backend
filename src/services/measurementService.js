/**
 * ═══════════════════════════════════════════════════════════════
 *  Measurement Service
 * 
 *  Orchestrates the facial measurement pipeline:
 *  Image → Python Engine → Results
 * ═══════════════════════════════════════════════════════════════
 */

const fs = require('fs').promises;
const path = require('path');
const pythonBridge = require('./pythonBridge');
const { AppError } = require('../middleware/errorHandler');

class MeasurementService {
    /**
     * Process a single image for facial measurements.
     * 
     * @param {string} imagePath - Path to the uploaded image
     * @param {number|null} referencePixels - Optional calibration reference
     * @returns {Promise<Object>} - Measurement results
     */
    async measureSingle(imagePath, referencePixels = null) {
        try {
            // Call Python engine
            const result = await pythonBridge.execute(imagePath, referencePixels);

            if (!result.success) {
                const error = result.error || {};
                throw new AppError(
                    error.message || 'Measurement failed',
                    this._mapErrorCode(error.code),
                    error.code || 'MEASUREMENT_FAILED'
                );
            }

            return {
                success: true,
                measurements: result.measurements,
                metadata: {
                    ...result.metadata,
                    processing_time_ms: result._processingTime,
                },
            };

        } catch (err) {
            // Re-throw AppErrors
            if (err instanceof AppError) throw err;

            // Wrap unexpected errors
            throw new AppError(
                err.message || 'An unexpected error occurred during measurement',
                err.statusCode || 500,
                err.code || 'MEASUREMENT_ERROR'
            );
        }
    }

    /**
     * Process multiple images and average results.
     * 
     * @param {string[]} imagePaths - Array of image paths
     * @param {number|null} referencePixels - Optional calibration reference
     * @returns {Promise<Object>} - Averaged measurement results
     */
    async measureMultiFrame(imagePaths, referencePixels = null) {
        const validResults = [];
        const errors = [];

        for (const imgPath of imagePaths) {
            try {
                const result = await this.measureSingle(imgPath, referencePixels);
                validResults.push(result.measurements);
            } catch (err) {
                errors.push(err.message);
            }
        }

        if (validResults.length === 0) {
            throw new AppError(
                `No valid measurements from ${imagePaths.length} images. Errors: ${errors.slice(0, 3).join('; ')}`,
                422,
                'NO_VALID_FRAMES'
            );
        }

        // Average all measurements
        const averaged = {};
        const keys = Object.keys(validResults[0]);
        for (const key of keys) {
            const values = validResults.map((m) => m[key]);
            averaged[key] = Math.round(
                (values.reduce((a, b) => a + b, 0) / values.length) * 100
            ) / 100;
        }

        return {
            success: true,
            measurements: averaged,
            metadata: {
                frames_processed: validResults.length,
                frames_rejected: errors.length,
                total_frames: imagePaths.length,
                averaging_method: 'mean',
            },
        };
    }

    /**
     * Cleanup uploaded file after processing.
     * @param {string} filePath - Path to the file to delete
     */
    async cleanup(filePath) {
        try {
            await fs.unlink(filePath);
        } catch (err) {
            console.warn(`[Cleanup] Failed to delete ${filePath}: ${err.message}`);
        }
    }

    /**
     * Map Python error codes to HTTP status codes.
     */
    _mapErrorCode(code) {
        const mapping = {
            'NO_FACE_DETECTED': 422,
            'MULTIPLE_FACES_DETECTED': 422,
            'HEAD_TILT_EXCEEDED': 422,
            'HEAD_TURNED': 422,
            'IMAGE_QUALITY_ERROR': 400,
            'INSUFFICIENT_LANDMARKS': 422,
            'CALIBRATION_MISSING': 400,
            'PROCESSING_ERROR': 500,
            'PYTHON_TIMEOUT': 408,
            'PYTHON_NOT_FOUND': 500,
        };
        return mapping[code] || 500;
    }
}

module.exports = new MeasurementService();
