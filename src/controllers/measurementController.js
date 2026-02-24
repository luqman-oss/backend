/**
 * ═══════════════════════════════════════════════════════════════
 *  Measurement Controller
 * ═══════════════════════════════════════════════════════════════
 */

const measurementService = require('../services/measurementService');

class MeasurementController {
    /**
     * POST /api/measure
     * 
     * Process a single facial image and return measurements.
     */
    async measureSingle(req, res, next) {
        const startTime = Date.now();

        try {
            const imagePath = req.file.path;
            const referencePixels = req.calibrationPixels || null;

            const result = await measurementService.measureSingle(
                imagePath,
                referencePixels
            );

            // Add timing info
            result.metadata = {
                ...result.metadata,
                api_response_time_ms: Date.now() - startTime,
            };

            // Return measurements in the exact format specified
            res.status(200).json({
                pupillary_distance_mm: result.measurements.pupillary_distance_mm,
                face_width_mm: result.measurements.face_width_mm,
                face_height_mm: result.measurements.face_height_mm,
                eye_width_mm: result.measurements.eye_width_mm,
                eye_height_mm: result.measurements.eye_height_mm,
                bridge_width_mm: result.measurements.bridge_width_mm,
                forehead_width_mm: result.measurements.forehead_width_mm,
                side_length_mm: result.measurements.side_length_mm,
                _metadata: result.metadata,
            });

            // Cleanup uploaded file
            await measurementService.cleanup(imagePath);

        } catch (err) {
            // Cleanup on error too
            if (req.file && req.file.path) {
                await measurementService.cleanup(req.file.path);
            }
            next(err);
        }
    }

    /**
     * POST /api/measure/detailed
     * 
     * Process a single facial image and return detailed measurements
     * with full metadata.
     */
    async measureDetailed(req, res, next) {
        const startTime = Date.now();

        try {
            const imagePath = req.file.path;
            const referencePixels = req.calibrationPixels || null;

            const result = await measurementService.measureSingle(
                imagePath,
                referencePixels
            );

            result.metadata.api_response_time_ms = Date.now() - startTime;

            res.status(200).json(result);

            await measurementService.cleanup(imagePath);

        } catch (err) {
            if (req.file && req.file.path) {
                await measurementService.cleanup(req.file.path);
            }
            next(err);
        }
    }

    /**
     * POST /api/measure/multi
     * 
     * Process multiple frames for averaged measurements.
     * Expects multiple files uploaded as 'images'.
     */
    async measureMultiFrame(req, res, next) {
        const startTime = Date.now();

        try {
            if (!req.files || req.files.length === 0) {
                return res.status(400).json({
                    error: 'No image files provided. Upload multiple images using the "images" field.',
                });
            }

            const imagePaths = req.files.map((f) => f.path);
            const referencePixels = req.calibrationPixels || null;

            const result = await measurementService.measureMultiFrame(
                imagePaths,
                referencePixels
            );

            result.metadata.api_response_time_ms = Date.now() - startTime;

            res.status(200).json(result);

            // Cleanup all uploaded files
            for (const imgPath of imagePaths) {
                await measurementService.cleanup(imgPath);
            }

        } catch (err) {
            // Cleanup all files on error
            if (req.files) {
                for (const file of req.files) {
                    await measurementService.cleanup(file.path);
                }
            }
            next(err);
        }
    }
}

module.exports = new MeasurementController();
