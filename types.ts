
export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export enum Asset {
  // CRYPTO (BINANCE REAL-TIME)
  BTCUSD = 'BTCUSD',
  ETHUSD = 'ETHUSD',
  SOLUSD = 'SOLUSD',
  BNBUSD = 'BNBUSD', // ADDED BNB

  // COMMODITY (METAL)
  XAUUSD = 'XAUUSD', // Gold
  XAGUSD = 'XAGUSD', // Silver
  XPTUSD = 'XPTUSD', // Platinum
  
  // FOREX MAJORS
  EURUSD = 'EURUSD',
  GBPUSD = 'GBPUSD',
  USDJPY = 'USDJPY',
  AUDUSD = 'AUDUSD',
  USDCHF = 'USDCHF',
  NZDUSD = 'NZDUSD',

  // FOREX CROSS PAIRS
  GBPJPY = 'GBPJPY',
  EURJPY = 'EURJPY',
}

export type TimeFrame = 'M1' | 'M5' | 'M15' | 'M30' | 'H1' | 'H4' | 'D1';

export type TradeFeedback = 'WIN' | 'LOSS' | null;

export type VoiceGender = 'MALE' | 'FEMALE';

export type UserRole = 'GUEST' | 'USER' | 'DEV';

export type MembershipTier = 'NONE' | 'BASIC' | 'VIP' | 'MONTHLY' | 'LIFETIME';

export interface UserSession {
    isLoggedIn: boolean;
    role: UserRole;
    username: string;
    tokens: number; // ADDED: Token System
    membershipTier?: MembershipTier; // NEW: Track UI Membership
    membershipExpiresAt?: number;    // NEW: Track UI Expiry
    deviceId?: string;               // NEW: Persistent Guest ID
}

// --- DATABASE STRUCTURE ---
export interface ExecutionRecord {
    id: string;
    asset: Asset;
    signal: 'BUY' | 'SELL';
    entry: number;
    sl: number;
    tp: number;
    status: 'WIN' | 'LOSS';
    timestamp: number;
    rr: string;
}

export interface UserRecord {
    id: string;
    username: string;
    passwordHash: string; // Encrypted Password
    role: UserRole;
    tokens: number;
    integritySignature?: string; // NEW: ANTI-HACK SIGNATURE (HMAC)
    createdAt: number;
    lastLogin: number;
    history?: ExecutionRecord[]; // NEW: Trade History
    // NEW: MEMBERSHIP FIELDS
    membershipTier?: MembershipTier;
    membershipExpiresAt?: number;
    deviceId?: string; // NEW: Persistent Guest ID
}

export interface OtpRecord {
    username: string;
    code: string;
    expiresAt: number; // Timestamp
}

export interface AnalysisResult {
  signal: 'BUY' | 'SELL' | 'WAIT';
  confidence: number;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  riskRewardRatio: string;
  reasoning: string[];
  smcConceptsFound: string[];
  timestamp: string;
  timeframe: string;
  prediction?: string; // NEW: Future prediction
}

export interface GeminiResponseSchema {
  signal: string;
  confidence: number;
  entry: number;
  sl: number;
  tp: number;
  rr: string;
  reasoning: string[];
  concepts: string[];
  prediction?: string; // NEW: Future prediction
}
