/**
 * ═══════════════════════════════════════════════════════════════
 *  FACIAL MEASUREMENT API SERVER
 * 
 *  AI-Based Real-Time 3D Facial Measurement Backend System
 *  
 *  Architecture:
 *  ┌──────────────────┐
 *  │   Express.js     │  ← API Layer (Node.js)
 *  │   REST API       │
 *  └───────┬──────────┘
 *          │  spawn
 *  ┌───────▼──────────┐
 *  │   Python AI      │  ← Processing Layer
 *  │   MediaPipe      │
 *  │   Face Mesh      │
 *  └───────┬──────────┘
 *          │
 *  ┌───────▼──────────┐
 *  │   JSON Response  │  ← Output
 *  │   Measurements   │
 *  └──────────────────┘
 * 
 * ═══════════════════════════════════════════════════════════════
 */

require('dotenv').config();
const express = require('express');
const path = require('path');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const fs = require('fs');

const config = require('./config');
const { errorHandler } = require('./middleware/errorHandler');
const measureRoutes = require('./routes/measureRoutes');
const healthRoutes = require('./routes/healthRoutes');
const cartRoutes = require('./routes/cartRoutes');
const widgetRoutes = require('./routes/widgetRoutes');

// ═══════════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════════

const app = express();

// Ensure upload directory exists
if (!fs.existsSync(config.UPLOAD_DIR)) {
    fs.mkdirSync(config.UPLOAD_DIR, { recursive: true });
    console.log(`[Init] Created upload directory: ${config.UPLOAD_DIR}`);
}

// ═══════════════════════════════════════════════════════════════
//  MIDDLEWARE
// ═══════════════════════════════════════════════════════════════

// Security headers
app.use(helmet({
    crossOriginResourcePolicy: { policy: 'cross-origin' },
    contentSecurityPolicy: false,
}));

// CORS
app.use(cors({
    origin: config.CORS_ORIGINS,
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'Authorization'],
}));

// Request logging (detailed)
app.use((req, reqRes, next) => {
    console.log(`[Request] ${req.method} ${req.url}`);
    next();
});

// JSON parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// ═══════════════════════════════════════════════════════════════
//  ROUTES
// ═══════════════════════════════════════════════════════════════

// Mount API routes BEFORE static files to ensure precedence
app.use('/api/measure', measureRoutes);
app.use('/api/health', healthRoutes);
app.use('/api/cart', cartRoutes);
app.use('/api/widget', widgetRoutes);

// API info
app.get('/api', (req, res) => {
    res.json({
        name: 'Facial Measurement API',
        version: '1.0.0',
        description: 'AI-Based Real-Time 3D Facial Measurement System',
    });
});

// Static files — serve frontend UI from public/
app.use(express.static(path.join(__dirname, '..', 'public')));

// SPA fallback — serve index.html for non-API routes
app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api')) return next();
    res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

// 404 handler
app.use((req, res) => {
    console.warn(`[404] ${req.method} ${req.originalUrl}`);
    res.status(404).json({
        error: {
            code: 'NOT_FOUND',
            message: `Route ${req.method} ${req.originalUrl} not found`,
        },
    });
});

// Global error handler
app.use(errorHandler);

// ═══════════════════════════════════════════════════════════════
//  START SERVER
// ═══════════════════════════════════════════════════════════════

const server = app.listen(config.PORT, config.HOST, () => {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════╗');
    console.log('║                                                      ║');
    console.log('║   🧠 FACIAL MEASUREMENT API  —  Backend Only         ║');
    console.log('║                                                      ║');
    console.log(`║   🌐 http://${config.HOST}:${config.PORT}                         ║`);
    console.log(`║   🐍 Python: ${config.PYTHON_EXECUTABLE.slice(-40).padEnd(40)} ║`);
    console.log('║                                                      ║');
    console.log('║   POST  /api/measure           Single measurement    ║');
    console.log('║   POST  /api/measure/detailed  Full details          ║');
    console.log('║   POST  /api/measure/multi     Multi-frame average   ║');
    console.log('║   POST  /api/cart/add-measurement  Cart integration  ║');
    console.log('║   GET   /api/widget/measure    Widget iframe        ║');
    console.log('║   GET   /api/health            Health check          ║');
    console.log('║   GET   /api/health/detailed   System status         ║');
    console.log('║                                                      ║');
    console.log('╚══════════════════════════════════════════════════════╝');
    console.log('');
});

// Graceful shutdown
const pythonBridge = require('./services/pythonBridge');

process.on('SIGTERM', () => {
    console.log('[Server] SIGTERM received. Shutting down gracefully...');
    pythonBridge.shutdown();
    server.close(() => {
        console.log('[Server] Server closed.');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('[Server] SIGINT received. Shutting down...');
    pythonBridge.shutdown();
    server.close(() => {
        process.exit(0);
    });
});

module.exports = app;
