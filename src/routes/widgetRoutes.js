/**
 * Widget Routes
 * Serves face measurement widget for iframe embedding on product pages
 */

const express = require('express');
const router = express.Router();
const path = require('path');

/**
 * @route   GET /api/widget/measure
 * @desc    Serve face measurement widget page for iframe
 * @access  Public
 */
router.get('/measure', (req, res) => {
  const { shop, product } = req.query;
  
  // Redirect to the working widget URL
  res.redirect('https://mirian-ecospecific-ann.ngrok-free.dev');
});

/**
 * Generate standalone widget HTML that loads the React app
 */
function generateWidgetHtml({ shop, product }) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Face Measurement - Alira</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #1a1a2e;
      color: #e0e0e0;
      min-height: 100vh;
      overflow-x: hidden;
    }
    #root { width: 100%; min-height: 100vh; }
    .loading {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
      color: #888;
    }
  </style>
</head>
<body>
  <div id="root">
    <div class="loading">Loading Face Measurement...</div>
  </div>
  
  <script>
    window.ALIRA_WIDGET_CONFIG = {
      shop: '${shop || ''}',
      product: '${product || ''}',
      apiUrl: '/api',
      isWidget: true
    };
    
    // Notify parent window when measurement is complete
    window.addEventListener('message', function(event) {
      if (event.data && event.data.type === 'MEASUREMENT_COMPLETE') {
        window.parent.postMessage({
          type: 'MEASUREMENT_COMPLETE',
          data: event.data.data
        }, '*');
      }
      if (event.data && event.data.type === 'CLOSE_MODAL') {
        window.parent.postMessage({
          type: 'CLOSE_MODAL'
        }, '*');
      }
    });
  </script>
  
  <!-- In production, this would load the built widget bundle -->
  <script>
    // Fallback: redirect to measurement API if widget not built
    if (!window.ALIRA_WIDGET_LOADED) {
      console.log('[Alira Widget] Standalone mode - using API directly');
    }
  </script>
</body>
</html>`;
}

/**
 * @route   GET /api/widget/config
 * @desc    Get widget configuration for a shop
 * @access  Public
 */
router.get('/config', (req, res) => {
  const { shop } = req.query;
  
  // Return default widget configuration
  // In production, this would fetch from database
  res.json({
    buttonText: 'Measure Your Face',
    buttonStyle: 'rounded',
    buttonColor: '#3b6978',
    textColor: '#ffffff',
    position: 'above_add_to_cart',
    showPreview: true,
    requiredMeasurements: ['pupillary_distance_mm', 'face_width_mm']
  });
});

module.exports = router;
