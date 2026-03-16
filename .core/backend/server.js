
/** @type {{ req: "Hi, 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a ", ai_edit: "strict", on_fail: "simulate_error" }} */
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');

const app = express();
const PORT = process.env.PORT || 5000;

// --- DATABASE CONNECTION ---
// Menggunakan Connection Pool untuk performa tinggi handling banyak user
const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

// --- MIDDLEWARE ---
app.use(cors()); // Mengizinkan akses dari domain manapun (bisa di-strict saat production)
app.use(express.json());

// Pass DB pool ke setiap request
app.use((req, res, next) => {
    req.db = pool;
    next();
});

// --- ROUTES MANAGEMENT ---
// Proxy for Economic Calendar (TradingView)
app.get('/api/calendar', async (req, res) => {
    try {
        const { from, to } = req.query;
        if (!from || !to) return res.status(400).json({ error: 'Missing from/to parameters' });
        
        const fetch = (await import('node-fetch')).default || global.fetch;
        const response = await fetch(`https://economic-calendar.tradingview.com/events?from=${from}&to=${to}`, {
            headers: {
                'origin': 'https://www.tradingview.com',
                'referer': 'https://www.tradingview.com/'
            }
        });
        
        if (!response.ok) throw new Error('Failed to fetch from TradingView');
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('Calendar Proxy Error:', error);
        res.status(500).json({ error: 'Failed to fetch calendar data' });
    }
});

try {
    const authRoutes = require('./routes/auth');
    app.use('/api/auth', authRoutes);
} catch (e) {}

try {
    const adminRoutes = require('./routes/admin');
    app.use('/api/admin', adminRoutes);
} catch (e) {}

try {
    const userRoutes = require('./routes/user');
    app.use('/api/user', userRoutes);
} catch (e) {}

// Health Check Endpoint
app.get('/', (req, res) => {
    res.json({ 
        status: "NVR QUANTUM CORE ONLINE", 
        system: "STABLE",
        timestamp: Date.now() 
    });
});

app.listen(PORT, () => {
    console.log(`🚀 SYSTEM ONLINE PORT: ${PORT}`);
    console.log(`📡 READY TO RECEIVE SIGNALS`);
});
