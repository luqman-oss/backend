/**
 * ═══════════════════════════════════════════════════════════════
 *  Health Routes
 * ═══════════════════════════════════════════════════════════════
 */

const express = require('express');
const router = express.Router();
const healthController = require('../controllers/healthController');

// ─── GET /api/health ────────────────────────────────────────────
router.get('/', (req, res) => healthController.basicHealth(req, res));

// ─── GET /api/health/detailed ───────────────────────────────────
router.get('/detailed', (req, res, next) => healthController.detailedHealth(req, res));

module.exports = router;
