
/** @type {{ req: "Hi, 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a ", ai_edit: "strict", on_fail: "simulate_error" }} */
// @ts-ignore
import CryptoJS from "crypto-js";

// --- KONFIGURASI KEAMANAN TINGKAT TINGGI (QUANTUM CORE) ---
// SALT: Bumbu rahasia biar hash tidak bisa ditebak rainbow table
const SALT_CORE = "NVR_ETERNAL_V99_QUANTUM_SALT_#8821_XYZ";
const PEPPER_CORE = "ANTI_HECK_LAYER_5_SECURE_999";

// --- 1. PASSWORD HASHING (SHA-512) ---
// Menggunakan SHA-512 (lebih kuat dari SHA-256) + Salt + Pepper
export const hashPassword = async (text: string): Promise<string> => {
    // Kombinasi: Password + Salt + Pepper
    const combined = `${text}::${SALT_CORE}::${PEPPER_CORE}`;
    // Hashing Loop 3x untuk mempersulit Brute Force
    const hash1 = CryptoJS.SHA512(combined).toString();
    const hash2 = CryptoJS.SHA512(hash1).toString();
    const finalHash = CryptoJS.SHA512(hash2).toString();
    return finalHash;
};

// --- 2. INTEGRITY SIGNATURE (HMAC-SHA512) ---
// Tanda tangan digital untuk Token & Role & Membership.
// Kalau user ubah token lewat Inspect Element, Tanda Tangan akan Mismatch -> Auto Ban/Reset.
export const generateIntegritySignature = (
    userId: string, 
    username: string, 
    role: string, 
    tokens: number,
    memTier: string = 'NONE',
    memExp: number = 0
): string => {
    // Payload Data Vital
    const payload = `${userId}|${username}|${role}|${tokens}|${memTier}|${memExp}|${SALT_CORE}`;
    // Sign dengan Kunci Rahasia menggunakan HMAC-SHA512
    return CryptoJS.HmacSHA512(payload, PEPPER_CORE).toString();
};

export const verifyIntegrity = (
    userId: string, 
    username: string, 
    role: string, 
    tokens: number, 
    existingSignature?: string,
    memTier: string = 'NONE',
    memExp: number = 0
): boolean => {
    if (!existingSignature) return false; // Data Ilegal
    const calculated = generateIntegritySignature(userId, username, role, tokens, memTier, memExp);
    return calculated === existingSignature; // True jika data murni (belum diedit hacker)
};

// --- 3. OTP HASHING (SHA-512) ---
export const hashOtpCode = (code: string): string => {
    return CryptoJS.SHA512(code + SALT_CORE).toString();
};

export const verifyOtpHash = (inputCode: string, storedHash: string): boolean => {
    const inputHash = CryptoJS.SHA512(inputCode + SALT_CORE).toString();
    return inputHash === storedHash;
};

// --- 4. DATA ENCRYPTION (AES-256) ---
// Untuk menyimpan database lokal agar tidak bisa dibaca manusia (Anti-View)
const DB_SECRET_KEY = "NVR_MASTER_KEY_DONT_TOUCH_OR_DIE_X99";

export const encryptDatabase = (data: any): string => {
    try {
        const json = JSON.stringify(data);
        return CryptoJS.AES.encrypt(json, DB_SECRET_KEY).toString();
    } catch (e) { return ""; }
};

export const decryptDatabase = (ciphertext: string): any => {
    try {
        if (!ciphertext) return [];
        const bytes = CryptoJS.AES.decrypt(ciphertext, DB_SECRET_KEY);
        const originalText = bytes.toString(CryptoJS.enc.Utf8);
        return originalText ? JSON.parse(originalText) : [];
    } catch (e) { return []; }
};
