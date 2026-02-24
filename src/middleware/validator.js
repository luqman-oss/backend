/**
 * ═══════════════════════════════════════════════════════════════
 *  Request Validation Middleware
 * ═══════════════════════════════════════════════════════════════
 */

const { AppError } = require('./errorHandler');

// ─── Validate Image Upload ───────────────────────────────────
const validateImageUpload = (req, res, next) => {
    if (!req.file) {
        return next(
            new AppError(
                'No image file provided. Please upload an image using the "image" field.',
                400,
                'NO_IMAGE'
            )
        );
    }

    // Verify file exists on disk
    const fs = require('fs');
    if (!fs.existsSync(req.file.path)) {
        return next(
            new AppError('Uploaded file could not be saved.', 500, 'UPLOAD_FAILED')
        );
    }

    next();
};

// ─── Validate Calibration Parameters ─────────────────────────
const validateCalibration = (req, res, next) => {
    // Optional calibration: if reference_pixels is provided, validate it
    if (req.body && req.body.reference_pixels) {
        const refPixels = parseFloat(req.body.reference_pixels);
        if (isNaN(refPixels) || refPixels <= 0) {
            return next(
                new AppError(
                    'Invalid reference_pixels value. Must be a positive number.',
                    400,
                    'INVALID_CALIBRATION'
                )
            );
        }
        req.calibrationPixels = refPixels;
    }

    next();
};

module.exports = { validateImageUpload, validateCalibration };
