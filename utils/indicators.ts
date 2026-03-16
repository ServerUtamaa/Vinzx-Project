
import { Candle, TimeFrame } from "../types";

// --- CORE MATHEMATICS (THE LEFT BRAIN) ---

export const calculateSMA = (data: number[], period: number): number => {
  if (data.length < period) return 0;
  const slice = data.slice(-period);
  const sum = slice.reduce((a, b) => a + b, 0);
  return sum / period;
};

// Return Single Value (For Prompt Context)
export const calculateEMA = (data: number[], period: number, prevEMA?: number): number => {
  if (data.length < period) return 0;
  const k = 2 / (period + 1);
  const price = data[data.length - 1];
  
  if (prevEMA === undefined || isNaN(prevEMA)) {
    return calculateSMA(data, period); // First EMA is SMA
  }
  // Standard Formula: (Close - Prev) * k + Prev
  return (price - prevEMA) * k + prevEMA;
};

// Return Array Series (For Chart Visualization)
// REFACTORED: Precise calculation loop that respects live updates
export const calculateEMAArray = (candles: Candle[], period: number): number[] => {
    const emaArray: number[] = [];
    const k = 2 / (period + 1);
    
    // If not enough data, fill with NaN
    if (candles.length < period) {
        return new Array(candles.length).fill(NaN);
    }

    // 1. Calculate Initial SMA (Basis for first EMA)
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += candles[i].close;
    }
    let prevEMA = sum / period;

    // Fill NaN for periods before the EMA starts
    for (let i = 0; i < period - 1; i++) {
        emaArray.push(NaN);
    }
    
    // Push the first EMA (which is the SMA)
    emaArray.push(prevEMA);

    // 2. Calculate the rest using EMA Formula
    for (let i = period; i < candles.length; i++) {
        const close = candles[i].close;
        const currentEMA = (close - prevEMA) * k + prevEMA;
        emaArray.push(currentEMA);
        prevEMA = currentEMA;
    }

    return emaArray;
};

// --- NEW: EMA STRUCTURAL ANALYSIS (BASED ON USER MATERIAL) ---
export const analyzeEMACondition = (candles: Candle[]) => {
    const ema50Array = calculateEMAArray(candles, 50);
    const ema200Array = calculateEMAArray(candles, 200);

    const currentIdx = candles.length - 1;
    const prevIdx = candles.length - 2;

    if (currentIdx < 201) return { bias: 'NEUTRAL', cross: 'NONE', zone: 'NONE' };

    const currPrice = candles[currentIdx].close;
    const curr50 = ema50Array[currentIdx];
    const curr200 = ema200Array[currentIdx];
    
    const prev50 = ema50Array[prevIdx];
    const prev200 = ema200Array[prevIdx];

    // 1. BIAS DETERMINATION (Rule: Price vs EMA 200)
    let bias = 'RANGING';
    if (currPrice > curr200) bias = 'BULLISH (Bias BUY)';
    else if (currPrice < curr200) bias = 'BEARISH (Bias SELL)';

    // 2. CROSS DETECTION
    let cross = 'NONE';
    if (prev50 < prev200 && curr50 > curr200) cross = 'GOLDEN CROSS (Start Uptrend)';
    else if (prev50 > prev200 && curr50 < curr200) cross = 'DEATH CROSS (Start Downtrend)';

    // 3. DYNAMIC ZONE DETECTION (Price vs EMA 50)
    // Check distance percent to EMA 50
    const distTo50 = Math.abs(currPrice - curr50) / curr50 * 100;
    let zone = 'FAR';
    if (distTo50 < 0.05) zone = 'AT EMA 50 (Dynamic S/R)';

    return {
        bias,
        cross,
        zone,
        values: { ema50: curr50, ema200: curr200 }
    };
};

// Return Array Series for Bollinger Bands
export const calculateBollingerBandsArray = (candles: Candle[], period: number = 20, multiplier: number = 2) => {
    const upper: number[] = [];
    const lower: number[] = [];
    const closes = candles.map(c => c.close);

    for (let i = 0; i < candles.length; i++) {
        if (i < period - 1) {
            upper.push(NaN);
            lower.push(NaN);
            continue;
        }

        const slice = closes.slice(i - period + 1, i + 1);
        const sum = slice.reduce((a, b) => a + b, 0);
        const mean = sum / period;

        const squaredDiffs = slice.map(val => Math.pow(val - mean, 2));
        const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / period;
        const stdDev = Math.sqrt(avgSquaredDiff);

        upper.push(mean + (multiplier * stdDev));
        lower.push(mean - (multiplier * stdDev));
    }
    return { upper, lower };
};

export const calculateRSI = (candles: Candle[], period: number = 14): number => {
  if (candles.length < period + 1) return 50;

  let gains = 0;
  let losses = 0;

  // Calculate initial average
  for (let i = 1; i <= period; i++) {
    const change = candles[i].close - candles[i - 1].close;
    if (change > 0) gains += change;
    else losses += Math.abs(change);
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  // Smooth it (Wilder's Smoothing) for the rest
  for (let i = period + 1; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = ((avgGain * (period - 1)) + gain) / period;
    avgLoss = ((avgLoss * (period - 1)) + loss) / period;
  }

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
};

export const calculateBollingerBands = (candles: Candle[], period: number = 20, multiplier: number = 2) => {
  if (candles.length < period) return { upper: 0, middle: 0, lower: 0 };
  
  const closes = candles.map(c => c.close);
  const middle = calculateSMA(closes, period); 
  
  const slice = closes.slice(-period);
  const squaredDiffs = slice.map(val => Math.pow(val - middle, 2));
  const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / period;
  const stdDev = Math.sqrt(avgSquaredDiff);

  return {
    upper: middle + (multiplier * stdDev),
    middle: middle,
    lower: middle - (multiplier * stdDev)
  };
};

export const calculateATR = (candles: Candle[], period: number = 14): number => {
  if (candles.length < period + 1) return 0;

  const trs: number[] = [];
  for(let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const closePrev = candles[i-1].close;
    
    const tr = Math.max(
      high - low,
      Math.abs(high - closePrev),
      Math.abs(low - closePrev)
    );
    trs.push(tr);
  }

  return calculateSMA(trs, period);
};

// --- NEW: COMPREHENSIVE CANDLESTICK PATTERN DETECTION ---

export interface CandlestickPattern {
    name: string;
    type: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    strength: number; // 1-3
}

export const detectCandlestickPatterns = (candles: Candle[]): CandlestickPattern[] => {
    if (candles.length < 5) return [];

    const patterns: CandlestickPattern[] = [];
    const c = candles[candles.length - 1]; // Current
    const c1 = candles[candles.length - 2]; // Previous
    const c2 = candles[candles.length - 3]; // 2nd Previous
    const c3 = candles[candles.length - 4]; // 3rd Previous
    const c4 = candles[candles.length - 5]; // 4th Previous

    const body = Math.abs(c.close - c.open);
    const upperShadow = c.high - Math.max(c.close, c.open);
    const lowerShadow = Math.min(c.close, c.open) - c.low;
    const isBullish = c.close > c.open;
    const isBearish = c.close < c.open;
    const isDoji = body <= 0.1 * (c.high - c.low);

    const body1 = Math.abs(c1.close - c1.open);
    const upperShadow1 = c1.high - Math.max(c1.close, c1.open);
    const lowerShadow1 = Math.min(c1.close, c1.open) - c1.low;
    const isBullish1 = c1.close > c1.open;
    const isBearish1 = c1.close < c1.open;
    const isDoji1 = body1 <= 0.1 * (c1.high - c1.low);

    const body2 = Math.abs(c2.close - c2.open);
    const isBullish2 = c2.close > c2.open;
    const isBearish2 = c2.close < c2.open;
    const isDoji2 = body2 <= 0.1 * (c2.high - c2.low);

    const body3 = Math.abs(c3.close - c3.open);
    const isBullish3 = c3.close > c3.open;
    const isBearish3 = c3.close < c3.open;

    // Helper: Trend detection (simple)
    const isUptrend = c1.close > c2.close && c2.close > c3.close;
    const isDowntrend = c1.close < c2.close && c2.close < c3.close;

    const avgBody = (body + body1 + body2 + body3) / 4;
    const isLongBody = body > avgBody * 1.5;
    const isShortBody = body < avgBody * 0.5;

    // --- 1. SINGLE CANDLE PATTERNS (15) ---
    
    // 1. Hammer: B kecil & D >= 2B & U kecil & C > O
    if (isShortBody && lowerShadow >= 2 * body && upperShadow <= 0.1 * body && isBullish) {
        patterns.push({ name: 'Hammer', type: 'BULLISH', strength: 2 });
    }
    // 2. Hanging Man: Hammer in uptrend
    if (isShortBody && lowerShadow >= 2 * body && upperShadow <= 0.1 * body && isUptrend) {
        patterns.push({ name: 'Hanging Man', type: 'BEARISH', strength: 2 });
    }
    // 3. Inverted Hammer: B kecil & U >= 2B & D kecil
    if (isShortBody && upperShadow >= 2 * body && lowerShadow <= 0.1 * body) {
        patterns.push({ name: 'Inverted Hammer', type: 'BULLISH', strength: 1 });
    }
    // 4. Shooting Star: Inverted Hammer in uptrend
    if (isShortBody && upperShadow >= 2 * body && lowerShadow <= 0.1 * body && isUptrend) {
        patterns.push({ name: 'Shooting Star', type: 'BEARISH', strength: 2 });
    }
    // 5. Bullish Marubozu
    if (isBullish && upperShadow < 0.05 * body && lowerShadow < 0.05 * body && isLongBody) {
        patterns.push({ name: 'Bullish Marubozu', type: 'BULLISH', strength: 3 });
    }
    // 6. Bearish Marubozu
    if (isBearish && upperShadow < 0.05 * body && lowerShadow < 0.05 * body && isLongBody) {
        patterns.push({ name: 'Bearish Marubozu', type: 'BEARISH', strength: 3 });
    }
    // 7-11. Dojis
    if (isDoji) {
        if (upperShadow > 3 * body && lowerShadow > 3 * body) patterns.push({ name: 'Long-Legged Doji', type: 'NEUTRAL', strength: 1 });
        else if (lowerShadow > 3 * body && upperShadow <= 0.1 * body) patterns.push({ name: 'Dragonfly Doji', type: 'BULLISH', strength: 2 });
        else if (upperShadow > 3 * body && lowerShadow <= 0.1 * body) patterns.push({ name: 'Gravestone Doji', type: 'BEARISH', strength: 2 });
        else if (c.open === c.high && c.high === c.low && c.low === c.close) patterns.push({ name: 'Four Price Doji', type: 'NEUTRAL', strength: 1 });
        else patterns.push({ name: 'Standard Doji', type: 'NEUTRAL', strength: 1 });
    }
    // 12. Spinning Top
    if (isShortBody && upperShadow > body && lowerShadow > body && !isDoji) {
        patterns.push({ name: 'Spinning Top', type: 'NEUTRAL', strength: 1 });
    }
    // 13. Long Bullish Candle
    if (isBullish && isLongBody) patterns.push({ name: 'Long Bullish Candle', type: 'BULLISH', strength: 2 });
    // 14. Long Bearish Candle
    if (isBearish && isLongBody) patterns.push({ name: 'Long Bearish Candle', type: 'BEARISH', strength: 2 });
    // 15. Short Body Candle
    if (isShortBody && !isDoji) patterns.push({ name: 'Short Body Candle', type: 'NEUTRAL', strength: 1 });

    // --- 2. DOUBLE CANDLE PATTERNS (15) ---
    
    // 16. Bullish Engulfing
    if (isBearish1 && isBullish && body > body1 && c.open < c1.close && c.close > c1.open) {
        patterns.push({ name: 'Bullish Engulfing', type: 'BULLISH', strength: 3 });
    }
    // 17. Bearish Engulfing
    if (isBullish1 && isBearish && body > body1 && c.open > c1.close && c.close < c1.open) {
        patterns.push({ name: 'Bearish Engulfing', type: 'BEARISH', strength: 3 });
    }
    // 18. Piercing Line
    if (isBearish1 && isBullish && c.open < c1.low && c.close > (c1.open + c1.close) / 2 && c.close < c1.open) {
        patterns.push({ name: 'Piercing Line', type: 'BULLISH', strength: 2 });
    }
    // 19. Dark Cloud Cover
    if (isBullish1 && isBearish && c.open > c1.high && c.close < (c1.open + c1.close) / 2 && c.close > c1.open) {
        patterns.push({ name: 'Dark Cloud Cover', type: 'BEARISH', strength: 2 });
    }
    // 20. Bullish Harami
    if (isBearish1 && isBullish && body1 > body && c.high < c1.open && c.low > c1.close) {
        patterns.push({ name: 'Bullish Harami', type: 'BULLISH', strength: 2 });
    }
    // 21. Bearish Harami
    if (isBullish1 && isBearish && body1 > body && c.high < c1.close && c.low > c1.open) {
        patterns.push({ name: 'Bearish Harami', type: 'BEARISH', strength: 2 });
    }
    // 22. Harami Cross
    if (body1 > body && c.high < Math.max(c1.open, c1.close) && c.low > Math.min(c1.open, c1.close) && isDoji) {
        patterns.push({ name: 'Harami Cross', type: 'NEUTRAL', strength: 2 });
    }
    // 23. Tweezer Top
    if (Math.abs(c.high - c1.high) < 0.0001 * c.high && isUptrend) {
        patterns.push({ name: 'Tweezer Top', type: 'BEARISH', strength: 2 });
    }
    // 24. Tweezer Bottom
    if (Math.abs(c.low - c1.low) < 0.0001 * c.low && isDowntrend) {
        patterns.push({ name: 'Tweezer Bottom', type: 'BULLISH', strength: 2 });
    }
    // 25. Bullish Kicker
    if (isBearish1 && isBullish && c.open > c1.open) {
        patterns.push({ name: 'Bullish Kicker', type: 'BULLISH', strength: 3 });
    }
    // 26. Bearish Kicker
    if (isBullish1 && isBearish && c.open < c1.open) {
        patterns.push({ name: 'Bearish Kicker', type: 'BEARISH', strength: 3 });
    }
    // 27. Matching Low
    if (isBearish1 && isBearish && Math.abs(c.close - c1.close) < 0.0001 * c.close) {
        patterns.push({ name: 'Matching Low', type: 'BULLISH', strength: 2 });
    }
    // 28. Matching High
    if (isBullish1 && isBullish && Math.abs(c.close - c1.close) < 0.0001 * c.close) {
        patterns.push({ name: 'Matching High', type: 'BEARISH', strength: 2 });
    }
    // 29. On-Neck
    if (isBearish1 && isLongBody && isBullish && c.close === c1.low) {
        patterns.push({ name: 'On-Neck', type: 'BEARISH', strength: 2 });
    }
    // 30. In-Neck
    if (isBearish1 && isLongBody && isBullish && c.close > c1.low && c.close < c1.close) {
        patterns.push({ name: 'In-Neck', type: 'BEARISH', strength: 2 });
    }

    // --- 3. TRIPLE CANDLE PATTERNS (20) ---
    
    // 31. Morning Star
    if (isBearish2 && body2 > avgBody && isShortBody && isBullish && c.close > (c2.open + c2.close) / 2) {
        patterns.push({ name: 'Morning Star', type: 'BULLISH', strength: 3 });
    }
    // 32. Evening Star
    if (isBullish2 && body2 > avgBody && isShortBody && isBearish && c.close < (c2.open + c2.close) / 2) {
        patterns.push({ name: 'Evening Star', type: 'BEARISH', strength: 3 });
    }
    // 33. Morning Doji Star
    if (isBearish2 && body2 > avgBody && isDoji1 && isBullish && c.close > (c2.open + c2.close) / 2) {
        patterns.push({ name: 'Morning Doji Star', type: 'BULLISH', strength: 3 });
    }
    // 34. Evening Doji Star
    if (isBullish2 && body2 > avgBody && isDoji1 && isBearish && c.close < (c2.open + c2.close) / 2) {
        patterns.push({ name: 'Evening Doji Star', type: 'BEARISH', strength: 3 });
    }
    // 35. Three White Soldiers
    if (isBullish && isBullish1 && isBullish2 && c.close > c1.close && c1.close > c2.close && c.open > c1.open && c1.open > c2.open) {
        patterns.push({ name: 'Three White Soldiers', type: 'BULLISH', strength: 3 });
    }
    // 36. Three Black Crows
    if (isBearish && isBearish1 && isBearish2 && c.close < c1.close && c1.close < c2.close && c.open < c1.open && c1.open < c2.open) {
        patterns.push({ name: 'Three Black Crows', type: 'BEARISH', strength: 3 });
    }
    // 37. Three Inside Up
    if (isBearish2 && isBullish1 && c1.close < c2.open && c1.open > c2.close && isBullish && c.close > c2.open) {
        patterns.push({ name: 'Three Inside Up', type: 'BULLISH', strength: 3 });
    }
    // 38. Three Inside Down
    if (isBullish2 && isBearish1 && c1.close > c2.open && c1.open < c2.close && isBearish && c.close < c2.open) {
        patterns.push({ name: 'Three Inside Down', type: 'BEARISH', strength: 3 });
    }
    // 39. Three Outside Up
    if (isBearish2 && isBullish1 && c1.open < c2.close && c1.close > c2.open && isBullish && c.close > c1.close) {
        patterns.push({ name: 'Three Outside Up', type: 'BULLISH', strength: 3 });
    }
    // 40. Three Outside Down
    if (isBullish2 && isBearish1 && c1.open > c2.close && c1.close < c2.open && isBearish && c.close < c1.close) {
        patterns.push({ name: 'Three Outside Down', type: 'BEARISH', strength: 3 });
    }
    // 41. Rising Three Methods
    if (isBullish && isBearish1 && isBearish2 && isBearish3 && c4.close > c4.open && c.close > c4.close && c1.close > c4.open && c2.close > c4.open && c3.close > c4.open) {
        patterns.push({ name: 'Rising Three Methods', type: 'BULLISH', strength: 3 });
    }
    // 42. Falling Three Methods
    if (isBearish && isBullish1 && isBullish2 && isBullish3 && c4.close < c4.open && c.close < c4.close && c1.close < c4.open && c2.close < c4.open && c3.close < c4.open) {
        patterns.push({ name: 'Falling Three Methods', type: 'BEARISH', strength: 3 });
    }
    // 43. Three Line Strike
    if (isBullish && isBearish1 && isBearish2 && isBearish3 && c.open < c1.close && c.close > c3.open) {
        patterns.push({ name: 'Three Line Strike (Bullish)', type: 'BULLISH', strength: 3 });
    } else if (isBearish && isBullish1 && isBullish2 && isBullish3 && c.open > c1.close && c.close < c3.open) {
        patterns.push({ name: 'Three Line Strike (Bearish)', type: 'BEARISH', strength: 3 });
    }
    // 44. Deliberation Pattern
    if (isBullish && isBullish1 && isBullish2 && c2.close > c2.open && c1.close > c1.open && body < body1 && body1 < body2) {
        patterns.push({ name: 'Deliberation Pattern', type: 'BEARISH', strength: 2 });
    }
    // 45. Abandoned Baby (Bullish)
    if (isBearish2 && isDoji1 && isBullish && c1.high < c2.low && c1.high < c.low) {
        patterns.push({ name: 'Abandoned Baby (Bullish)', type: 'BULLISH', strength: 3 });
    }
    // 46. Abandoned Baby (Bearish)
    if (isBullish2 && isDoji1 && isBearish && c1.low > c2.high && c1.low > c.high) {
        patterns.push({ name: 'Abandoned Baby (Bearish)', type: 'BEARISH', strength: 3 });
    }
    // 47. Tri-Star
    if (isDoji && isDoji1 && isDoji2) {
        if (c1.low > c2.high && c1.low > c.high) patterns.push({ name: 'Tri-Star Top', type: 'BEARISH', strength: 3 });
        if (c1.high < c2.low && c1.high < c.low) patterns.push({ name: 'Tri-Star Bottom', type: 'BULLISH', strength: 3 });
    }
    // 48. Upside Tasuki Gap
    if (isBullish2 && isBullish1 && isBearish && c1.open > c2.close && c.open < c1.open && c.close < c1.open && c.close > c2.close) {
        patterns.push({ name: 'Upside Tasuki Gap', type: 'BULLISH', strength: 2 });
    }
    // 49. Downside Tasuki Gap
    if (isBearish2 && isBearish1 && isBullish && c1.open < c2.close && c.open > c1.open && c.close > c1.open && c.close < c2.close) {
        patterns.push({ name: 'Downside Tasuki Gap', type: 'BEARISH', strength: 2 });
    }
    // 50. Stick Sandwich
    if (isBearish2 && isBullish1 && isBearish && Math.abs(c.close - c2.close) < 0.0001 * c.close) {
        patterns.push({ name: 'Stick Sandwich', type: 'BULLISH', strength: 2 });
    }

    return patterns;
};

export interface SMCResult {
    structure: 'BULLISH' | 'BEARISH' | 'SIDEWAYS';
    phase: 'ACCUMULATION' | 'MARKUP' | 'DISTRIBUTION' | 'MARKDOWN';
    bos: 'BULLISH' | 'BEARISH' | 'NONE';
    choch: 'BULLISH' | 'BEARISH' | 'NONE';
    liquiditySweep: 'BUY_SIDE' | 'SELL_SIDE' | 'NONE';
    orderBlock: 'BULLISH' | 'BEARISH' | 'NONE';
    fvg: 'BULLISH' | 'BEARISH' | 'NONE';
    mitigationBlock: 'BULLISH' | 'BEARISH' | 'NONE';
    inducement: boolean;
    score: number;
    bullishStrongCount: number;
    bearishStrongCount: number;
    dojiCount: number;
}

export const detectSMC = (candles: Candle[], timeframe: TimeFrame = 'H1'): SMCResult => {
    const result: SMCResult = {
        structure: 'SIDEWAYS',
        phase: 'ACCUMULATION',
        bos: 'NONE',
        choch: 'NONE',
        liquiditySweep: 'NONE',
        orderBlock: 'NONE',
        fvg: 'NONE',
        mitigationBlock: 'NONE',
        inducement: false,
        score: 0,
        bullishStrongCount: 0,
        bearishStrongCount: 0,
        dojiCount: 0
    };

    if (candles.length < 150) return result;

    const isM1M5 = timeframe === 'M1' || timeframe === 'M5';
    const isM15M30 = timeframe === 'M15' || timeframe === 'M30';
    const isH1H4 = timeframe === 'H1' || timeframe === 'H4' || timeframe === 'D1';

    const wStructure = isM1M5 ? 2 : isM15M30 ? 3 : 4;
    const wMomentum = 2;
    const wBOS = 3;
    const wSweep = 3;
    const wOB = isM1M5 ? 2 : isM15M30 ? 3 : 4;
    const wFVG = isM1M5 ? 2 : isM15M30 ? 2 : 3;
    const wInducement = isM1M5 ? 2 : isM15M30 ? 2 : 1;

    const recentCandles = candles.slice(-150);
    const current = recentCandles[recentCandles.length - 1];
    const prev1 = recentCandles[recentCandles.length - 2];
    const prev2 = recentCandles[recentCandles.length - 3];
    const prev3 = recentCandles[recentCandles.length - 4];

    // 1. MARKET STRUCTURE (150 Candles)
    let swings: { type: 'HIGH' | 'LOW', price: number, index: number }[] = [];
    for (let i = 2; i < recentCandles.length - 2; i++) {
        const c = recentCandles[i];
        if (c.high > recentCandles[i-1].high && c.high > recentCandles[i-2].high && c.high > recentCandles[i+1].high && c.high > recentCandles[i+2].high) {
            swings.push({ type: 'HIGH', price: c.high, index: i });
        }
        if (c.low < recentCandles[i-1].low && c.low < recentCandles[i-2].low && c.low < recentCandles[i+1].low && c.low < recentCandles[i+2].low) {
            swings.push({ type: 'LOW', price: c.low, index: i });
        }
    }

    let hhCount = 0, hlCount = 0, llCount = 0, lhCount = 0;
    let lastHigh = 0, lastLow = Infinity;

    swings.forEach(s => {
        if (s.type === 'HIGH') {
            if (lastHigh !== 0 && s.price > lastHigh) hhCount++;
            else if (lastHigh !== 0 && s.price < lastHigh) lhCount++;
            lastHigh = s.price;
        } else {
            if (lastLow !== Infinity && s.price > lastLow) hlCount++;
            else if (lastLow !== Infinity && s.price < lastLow) llCount++;
            lastLow = s.price;
        }
    });

    const totalBullishSwings = hhCount + hlCount;
    const totalBearishSwings = llCount + lhCount;
    const totalSwings = totalBullishSwings + totalBearishSwings;

    if (totalSwings > 0) {
        if (totalBullishSwings / totalSwings >= 0.7) {
            result.structure = 'BULLISH';
            result.score += wStructure;
        } else if (totalBearishSwings / totalSwings >= 0.7) {
            result.structure = 'BEARISH';
            result.score -= wStructure;
        }
    }

    // 2. MOMENTUM DISTRIBUTION
    recentCandles.forEach(c => {
        const body = Math.abs(c.close - c.open);
        const range = c.high - c.low;
        const isBullish = c.close > c.open;
        const isBearish = c.close < c.open;
        
        if (range === 0) return;

        // Bullish Strong: Body >= 70% of range
        if (body / range >= 0.7) {
            if (isBullish) result.bullishStrongCount++;
            else result.bearishStrongCount++;
        } else if (body / range <= 0.1) {
            result.dojiCount++;
        }
    });

    if (result.bullishStrongCount > result.bearishStrongCount) {
        result.score += wMomentum;
    } else if (result.bearishStrongCount > result.bullishStrongCount) {
        result.score -= wMomentum;
    }

    // 3. SIKLUS MARKET (Wyckoff)
    if (result.bullishStrongCount > result.bearishStrongCount * 1.5) {
        result.phase = 'MARKUP';
        result.score += 2;
    } else if (result.bearishStrongCount > result.bullishStrongCount * 1.5) {
        result.phase = 'MARKDOWN';
        result.score -= 2;
    } else if (result.dojiCount > 30 && result.structure === 'BULLISH') {
        result.phase = 'DISTRIBUTION';
        result.score -= 2;
    } else if (result.dojiCount > 30 && result.structure === 'BEARISH') {
        result.phase = 'ACCUMULATION';
        result.score += 2;
    }

    // 4. BOS & CHoCH
    const currentBodySize = Math.abs(current.close - current.open) / (current.high - current.low);
    const isStrongBreak = currentBodySize >= 0.6;

    if (lastHigh !== 0 && current.close > lastHigh && isStrongBreak) {
        if (result.structure === 'BULLISH') {
            result.bos = 'BULLISH';
            result.score += wBOS;
        } else {
            result.choch = 'BULLISH';
            result.score += 2;
        }
    } else if (lastLow !== Infinity && current.close < lastLow && isStrongBreak) {
        if (result.structure === 'BEARISH') {
            result.bos = 'BEARISH';
            result.score -= wBOS;
        } else {
            result.choch = 'BEARISH';
            result.score -= 2;
        }
    }

    // 5. LIQUIDITY SWEEP
    const upperWick = current.high - Math.max(current.open, current.close);
    const lowerWick = Math.min(current.open, current.close) - current.low;
    const currentRange = current.high - current.low;

    if (lastHigh !== 0 && current.high > lastHigh && current.close < lastHigh && upperWick / currentRange >= 0.5) {
        result.liquiditySweep = 'BUY_SIDE';
        result.score -= wSweep; 
    } else if (lastLow !== Infinity && current.low < lastLow && current.close > lastLow && lowerWick / currentRange >= 0.5) {
        result.liquiditySweep = 'SELL_SIDE';
        result.score += wSweep;
    }

    // 6. FVG (Fair Value Gap)
    if (current.low > prev2.high && prev1.close > prev1.open && (prev1.close - prev1.open) / (prev1.high - prev1.low) >= 0.6) {
        result.fvg = 'BULLISH';
        // Score based on bias
        if (result.score > 0) result.score += wFVG;
        else result.score -= 1;
    } else if (current.high < prev2.low && prev1.close < prev1.open && (prev1.open - prev1.close) / (prev1.high - prev1.low) >= 0.6) {
        result.fvg = 'BEARISH';
        if (result.score < 0) result.score -= wFVG;
        else result.score += 1;
    }

    // 7. ORDER BLOCK (OB)
    if (result.fvg === 'BULLISH' && prev2.close < prev2.open) {
        result.orderBlock = 'BULLISH';
        if (current.close > prev2.high) result.score += wOB;
    } else if (result.fvg === 'BEARISH' && prev2.close > prev2.open) {
        result.orderBlock = 'BEARISH';
        if (current.close < prev2.low) result.score -= wOB;
    }

    // 8. INDUCEMENT
    if (result.liquiditySweep !== 'NONE' && !isStrongBreak) {
        result.inducement = true;
        if (result.liquiditySweep === 'SELL_SIDE') result.score += wInducement;
        else result.score -= wInducement;
    }

    // 8. MITIGATION BLOCK
    if (result.bos === 'BEARISH' && prev3.close > prev3.open && current.high >= prev3.low) {
        result.mitigationBlock = 'BEARISH';
    } else if (result.bos === 'BULLISH' && prev3.close < prev3.open && current.low <= prev3.high) {
        result.mitigationBlock = 'BULLISH';
    }

    return result;
};

export const getTrendStatus = (candles: Candle[]): string => {
  const emaAnalysis = analyzeEMACondition(candles);
  return emaAnalysis.bias; // Now returns specific EMA 200 Bias
};
