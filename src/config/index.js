/**
 * ═══════════════════════════════════════════════════════════════
 *  FACIAL MEASUREMENT API - Configuration
 * ═══════════════════════════════════════════════════════════════
 */

const path = require('path');

const config = {
  // ─── Server ───────────────────────────────────────────────
  PORT: process.env.PORT || 3000,
  HOST: process.env.HOST || '0.0.0.0',
  NODE_ENV: process.env.NODE_ENV || 'development',

  // ─── Upload Settings ─────────────────────────────────────
  UPLOAD_DIR: path.join(__dirname, '..', '..', 'uploads'),
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  ALLOWED_MIMETYPES: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'],
  ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],

  // ─── Python Engine ───────────────────────────────────────
  PYTHON_SCRIPT: path.join(__dirname, '..', '..', 'python', 'face_measurement_engine.py'),
  PYTHON_EXECUTABLE: process.env.PYTHON_PATH ? path.resolve(process.env.PYTHON_PATH) : 'python',
  PYTHON_TIMEOUT: 120000, // 120 seconds

  // ─── Calibration ─────────────────────────────────────────
  // Credit card width in mm (ISO/IEC 7810 standard)
  REFERENCE_OBJECT_MM: 85.6,
  // Default average interpupillary distance for calibration fallback (mm)
  DEFAULT_IPD_MM: 63.0,

  // ─── Accuracy Thresholds ─────────────────────────────────
  MAX_HEAD_TILT_DEGREES: 5,
  MIN_FACE_CONFIDENCE: 0.85,
  MIN_LANDMARK_CONFIDENCE: 0.80,
  MULTI_FRAME_COUNT: 20,

  // ─── API Rate Limiting ───────────────────────────────────
  RATE_LIMIT_WINDOW_MS: 15 * 60 * 1000, // 15 minutes
  RATE_LIMIT_MAX_REQUESTS: 100,

  // ─── CORS ────────────────────────────────────────────────
  CORS_ORIGINS: process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : ['*'],
};

module.exports = config;
