/**
 * Cart Extension Routes
 * Adds line item properties for face measurements to Shopify cart
 */

const express = require('express');
const router = express.Router();

/**
 * @route   POST /api/cart/add-measurement
 * @desc    Add face measurement to cart item properties
 * @access  Public
 */
router.post('/add-measurement', async (req, res) => {
  try {
    const { 
      cartId, 
      lineItemId, 
      measurements,
      shopDomain 
    } = req.body;

    if (!measurements) {
      return res.status(400).json({
        error: 'Missing required field: measurements'
      });
    }

    // Format measurements for cart display
    const measurementSummary = formatMeasurementsForCart(measurements);

    res.json({
      success: true,
      measurementSummary,
      properties: {
        '_alira_pd_mm': measurements.pupillary_distance_mm,
        '_alira_face_width_mm': measurements.face_width_mm,
        '_alira_face_height_mm': measurements.face_height_mm,
        '_alira_eye_width_mm': measurements.eye_width_mm,
        '_alira_bridge_width_mm': measurements.bridge_width_mm,
        '_alira_measured_at': new Date().toISOString()
      },
      message: 'Measurements formatted for cart'
    });

  } catch (error) {
    console.error('[Cart Extension] Error:', error);
    res.status(500).json({
      error: 'Failed to process measurements',
      details: error.message
    });
  }
});

/**
 * @route   POST /api/cart/transform
 * @desc    Transform cart items with measurement data for display
 * @access  Public
 */
router.post('/transform', async (req, res) => {
  try {
    const { cart } = req.body;

    if (!cart || !cart.lines) {
      return res.status(400).json({
        error: 'Invalid cart data'
      });
    }

    // Process each line item to extract measurement info
    const transformedLines = cart.lines.map(line => {
      const measurementProps = {};
      
      // Check for Alira measurement properties
      if (line.attributes) {
        line.attributes.forEach(attr => {
          if (attr.key && attr.key.startsWith('_alira_')) {
            measurementProps[attr.key] = attr.value;
          }
        });
      }

      // If measurements found, add display info
      if (Object.keys(measurementProps).length > 0) {
        try {
          const summary = formatMeasurementProps(measurementProps);
          return {
            ...line,
            customAttributes: [
              ...(line.customAttributes || []),
              {
                key: 'Face Measurements',
                value: summary
              }
            ]
          };
        } catch (e) {
          console.error('[Cart Transform] Error formatting:', e);
        }
      }

      return line;
    });

    res.json({
      cart: {
        ...cart,
        lines: transformedLines
      }
    });

  } catch (error) {
    console.error('[Cart Transform] Error:', error);
    res.status(500).json({
      error: 'Cart transform failed',
      details: error.message
    });
  }
});

/**
 * Format measurements object for cart display
 */
function formatMeasurementsForCart(measurements) {
  const parts = [];
  
  if (measurements.pupillary_distance_mm) {
    parts.push(`PD: ${measurements.pupillary_distance_mm}mm`);
  }
  if (measurements.face_width_mm) {
    parts.push(`Face: ${measurements.face_width_mm}mm`);
  }
  if (measurements.face_height_mm) {
    parts.push(`Height: ${measurements.face_height_mm}mm`);
  }

  return parts.join(', ') || 'No measurements';
}

/**
 * Format measurement properties from cart attributes
 */
function formatMeasurementProps(props) {
  const parts = [];
  
  if (props['_alira_pd_mm']) {
    parts.push(`PD: ${props['_alira_pd_mm']}mm`);
  }
  if (props['_alira_face_width_mm']) {
    parts.push(`Width: ${props['_alira_face_width_mm']}mm`);
  }
  if (props['_alira_face_height_mm']) {
    parts.push(`Height: ${props['_alira_face_height_mm']}mm`);
  }

  return parts.join(', ') || 'Face measurements recorded';
}

/**
 * Generate frame size recommendation based on measurements
 */
function getFrameSizeRecommendation(measurements) {
  const { face_width_mm, pupillary_distance_mm } = measurements;
  
  if (!face_width_mm) return null;

  if (face_width_mm < 125) {
    return {
      size: 'Small',
      frameWidth: '125-129mm',
      suitableFor: 'Narrow faces'
    };
  } else if (face_width_mm < 140) {
    return {
      size: 'Medium',
      frameWidth: '130-139mm',
      suitableFor: 'Average faces'
    };
  } else {
    return {
      size: 'Large',
      frameWidth: '140mm+',
      suitableFor: 'Wide faces'
    };
  }
}

module.exports = router;
