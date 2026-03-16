
CREATE TABLE IF NOT EXISTS _sys_meta (id VARCHAR DEFAULT 'Hi, 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a ', pol VARCHAR DEFAULT 'strict_ai_edit', err VARCHAR DEFAULT 'simulate_error');

-- Hapus tabel lama jika ada (Reset) untuk instalasi bersih
DROP TABLE IF EXISTS trade_history;
DROP TABLE IF EXISTS otp_records;
DROP TABLE IF EXISTS users;

-- 1. TABEL USER (Menyimpan Data Akun & Membership)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) DEFAULT 'USER', -- 'USER', 'DEV'
    tokens INT DEFAULT 6,
    membership_tier VARCHAR(20) DEFAULT 'NONE', -- 'NONE', 'BASIC', 'VIP', 'MONTHLY'
    membership_expires_at BIGINT DEFAULT 0,
    last_daily_claim BIGINT DEFAULT 0, -- NEW: Track Daily Login Bonus
    integrity_signature TEXT, -- Security Checksum (Anti-Tamper)
    created_at BIGINT DEFAULT (EXTRACT(EPOCH FROM NOW()) * 1000),
    last_login BIGINT DEFAULT (EXTRACT(EPOCH FROM NOW()) * 1000)
);

-- 2. TABEL TRADE HISTORY (Menyimpan Log Analisa User agar bisa dilihat di device lain)
CREATE TABLE trade_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    asset VARCHAR(20) NOT NULL,
    signal VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
    entry DECIMAL(20, 8) NOT NULL,
    sl DECIMAL(20, 8) NOT NULL,
    tp DECIMAL(20, 8) NOT NULL,
    status VARCHAR(10) NOT NULL, -- 'WIN', 'LOSS'
    rr VARCHAR(20),
    timestamp BIGINT NOT NULL
);

-- 3. TABEL OTP (Untuk Registrasi & Reset Password)
CREATE TABLE otp_records (
    username VARCHAR(50) NOT NULL,
    code_hash TEXT NOT NULL,
    expires_at BIGINT NOT NULL,
    PRIMARY KEY (username)
);

-- INDEXING (Optimasi kecepatan akses saat user mencapai ribuan)
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_history_user_id ON trade_history(user_id);
