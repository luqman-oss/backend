// Quick test: send test_face.jpg to /api/measure and show response
const http = require('http');
const fs = require('fs');
const path = require('path');

const img = fs.readFileSync(path.join(__dirname, '..', 'uploads', 'test_face.jpg'));
const boundary = '----Boundary' + Date.now();

const header = `--${boundary}\r\nContent-Disposition: form-data; name="image"; filename="test_face.jpg"\r\nContent-Type: image/jpeg\r\n\r\n`;
const footer = `\r\n--${boundary}--\r\n`;
const body = Buffer.concat([Buffer.from(header), img, Buffer.from(footer)]);

const req = http.request({
    hostname: 'localhost',
    port: 3000,
    path: '/api/measure',
    method: 'POST',
    headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': body.length,
    },
}, (res) => {
    let data = '';
    res.on('data', (chunk) => data += chunk);
    res.on('end', () => {
        console.log('Status:', res.statusCode);
        try {
            const json = JSON.parse(data);
            console.log(JSON.stringify(json, null, 2));
        } catch {
            console.log('Raw:', data);
        }
    });
});

req.write(body);
req.end();
