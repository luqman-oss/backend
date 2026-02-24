/**
 * ═══════════════════════════════════════════════════════════════
 *  Measurement Routes
 * ═══════════════════════════════════════════════════════════════
 */

const express = require('express');
const router = express.Router();
const upload = require('../middleware/upload');
const { validateImageUpload, validateCalibration } = require('../middleware/validator');
const measurementController = require('../controllers/measurementController');

// ─── POST /api/measure ─────────────────────────────────────────
// Single image measurement (primary endpoint)
router.post(
    '/',
    upload.single('image'),
    validateImageUpload,
    validateCalibration,
    (req, res, next) => measurementController.measureSingle(req, res, next)
);

// ─── POST /api/measure/detailed ─────────────────────────────────
// Single image with detailed metadata
router.post(
    '/detailed',
    upload.single('image'),
    validateImageUpload,
    validateCalibration,
    (req, res, next) => measurementController.measureDetailed(req, res, next)
);

// ─── POST /api/measure/multi ────────────────────────────────────
// Multi-frame averaged measurement
router.post(
    '/multi',
    upload.array('images', 30),
    validateCalibration,
    (req, res, next) => measurementController.measureMultiFrame(req, res, next)
);

module.exports = router;
