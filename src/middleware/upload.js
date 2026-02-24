/**
 * ═══════════════════════════════════════════════════════════════
 *  Image Upload Middleware (Multer)
 * ═══════════════════════════════════════════════════════════════
 */

const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const config = require('../config');

// ─── Storage Configuration ────────────────────────────────────
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, config.UPLOAD_DIR);
    },
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        const uniqueName = `${uuidv4()}${ext}`;
        cb(null, uniqueName);
    },
});

// ─── File Filter ──────────────────────────────────────────────
const fileFilter = (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const mimeOk = config.ALLOWED_MIMETYPES.includes(file.mimetype);
    const extOk = config.ALLOWED_EXTENSIONS.includes(ext);

    if (mimeOk && extOk) {
        cb(null, true);
    } else {
        cb(
            new Error(
                `Invalid file type. Allowed types: ${config.ALLOWED_EXTENSIONS.join(', ')}`
            ),
            false
        );
    }
};

// ─── Multer Instance ──────────────────────────────────────────
const upload = multer({
    storage,
    fileFilter,
    limits: {
        fileSize: config.MAX_FILE_SIZE,
        files: 1,
    },
});

module.exports = upload;
