const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());

app.get('/api/health', (req, res) => {
    console.log('Health check received');
    res.json({ status: 'ok' });
});

app.listen(5000, () => {
    console.log('Simple test server running on port 5000');
});
