/**
 * ═══════════════════════════════════════════════════════════════
 *  Global Error Handler Middleware
 * ═══════════════════════════════════════════════════════════════
 */

const config = require('../config');

class AppError extends Error {
    constructor(message, statusCode = 500, code = 'INTERNAL_ERROR') {
        super(message);
        this.statusCode = statusCode;
        this.code = code;
        this.isOperational = true;
        Error.captureStackTrace(this, this.constructor);
    }
}

// ─── Error Handler Middleware ─────────────────────────────────
const errorHandler = (err, req, res, next) => {
    let statusCode = err.statusCode || 500;
    let message = err.message || 'Internal Server Error';
    let code = err.code || 'INTERNAL_ERROR';

    // Multer-specific errors
    if (err.code === 'LIMIT_FILE_SIZE') {
        statusCode = 413;
        message = `File too large. Maximum size is ${config.MAX_FILE_SIZE / (1024 * 1024)}MB`;
        code = 'FILE_TOO_LARGE';
    }

    if (err.code === 'LIMIT_UNEXPECTED_FILE') {
        statusCode = 400;
        message = 'Unexpected file field. Use "image" as the field name.';
        code = 'INVALID_FIELD';
    }

    // Log error in development
    if (config.NODE_ENV === 'development') {
        console.error('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        console.error(`[ERROR] ${new Date().toISOString()}`);
        console.error(`Status: ${statusCode}`);
        console.error(`Message: ${message}`);
        console.error(`Stack: ${err.stack}`);
        console.error('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    }

    res.status(statusCode).json({
        success: false,
        error: {
            code,
            message,
            ...(config.NODE_ENV === 'development' && { stack: err.stack }),
        },
    });
};

module.exports = { AppError, errorHandler };
