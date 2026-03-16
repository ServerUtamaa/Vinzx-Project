
import { GoogleGenAI, Type, Schema } from "@google/genai";
import { Asset, Candle, AnalysisResult, GeminiResponseSchema, TradeFeedback, TimeFrame } from "../types";
import { SYSTEM_INSTRUCTION } from "../constants";
import { calculateRSI, calculateBollingerBands, calculateATR, analyzeEMACondition, detectCandlestickPatterns } from "../utils/indicators";

const responseSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    signal: { type: Type.STRING, enum: ["BUY", "SELL", "WAIT"] },
    confidence: { type: Type.NUMBER },
    entry: { type: Type.NUMBER },
    sl: { type: Type.NUMBER },
    tp: { type: Type.NUMBER },
    rr: { type: Type.STRING },
    reasoning: { 
      type: Type.ARRAY, 
      items: { type: Type.STRING },
      description: "Return exactly 14 strings following the 14-Point Protocol." 
    },
    concepts: { 
      type: Type.ARRAY, 
      items: { type: Type.STRING } 
    },
    prediction: {
      type: Type.STRING,
      description: "Predict the next 10-25 candles movement based on the 150-candle history and timeframe."
    }
  },
  required: ["signal", "confidence", "entry", "sl", "tp", "reasoning", "concepts", "prediction"]
};

export const analyzeMarketStructure = async (
  asset: Asset,
  candles: Candle[],
  timeframe: TimeFrame,
  lastFeedback: TradeFeedback,
  winStreak: number = 0,
  lossStreak: number = 0
): Promise<AnalysisResult> => {
  try {
    // STRICTLY USE process.env.API_KEY AS PER GUIDELINES
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
    const currentPrice = candles[candles.length - 1].close;
    const rsi = calculateRSI(candles, 14);
    const emaState = analyzeEMACondition(candles);
    const atr = calculateATR(candles, 14);
    const patterns = detectCandlestickPatterns(candles);

    const dataSlice = candles.slice(-800); 
    const dataString = dataSlice.map(c => 
      `${c.open},${c.high},${c.low},${c.close}`
    ).join('\n');

    // --- ASI AGENTIC CONTEXT ---
    let feedbackContext = `
      🤖 **PROTOCOL: ASI AGENTIC CORE vX.2 (UPGRADED)** 🤖
      TARGET: ${asset} | TF: ${timeframe}
      
      [MATHEMATICAL STATE]
      PRICE: ${currentPrice}
      EMA_BIAS: ${emaState.bias} (Trend Vector)
      RSI_MOMENTUM: ${rsi.toFixed(2)}
      VOLATILITY (ATR): ${atr.toFixed(4)}
      DETECTED_PATTERNS: ${patterns.map(p => p.name).join(', ') || 'NONE'}
      
      [USER PERFORMANCE CONTEXT]
      WIN STREAK: ${winStreak}
      LOSS STREAK: ${lossStreak}
      LAST FEEDBACK: ${lastFeedback ? `Status: ${lastFeedback}` : 'NONE'}
    `;

    // --- USER DEFINED ALGORITHM: STRICT SMC + ICT ---
    feedbackContext += `
    
    📜 **CONSTITUTIONAL ALGORITHM (EXECUTE SEQUENTIALLY):**
    
    1. **MARKET STRUCTURE (HTF & LTF)**
       - Analyze HH/HL or LL/LH Structure across the last 150+ candles.
       - Identify BOS (Break of Structure) for trend continuation.
       - Identify CHoCH (Change of Character) for trend reversal.
       - IF Structure is unclear or ranging -> SIGNAL: WAIT.
       
    2. **LIQUIDITY & INDUCEMENT (CRITICAL)**
       - Locate BSL (Buy Side Liquidity) above old highs and SSL (Sell Side Liquidity) below old lows.
       - **MANDATORY:** Has liquidity been swept recently? If NO -> SIGNAL: WAIT.
       - Identify Inducement (IDM) that must be swept before entry.
       
    3. **POI (POINT OF INTEREST) & ARRAYS**
       - **LONG:** Must be in **DISCOUNT** (< 50%). Find unmitigated Bullish OB or FVG.
       - **SHORT:** Must be in **PREMIUM** (> 50%). Find unmitigated Bearish OB or FVG.
       - If price is in Equilibrium (middle) -> SIGNAL: WAIT.
       
    4. **EXECUTION & RISK MANAGEMENT**
       - **ENTRY:** Exactly at the OB proximal line or FVG start. Wait for LTF confirmation (e.g., rejection wick or LTF CHoCH). NEVER enter in the middle of nowhere.
       - **SL:** Strictly beyond the OB distal line or the structural swing point. DO NOT use arbitrary pip values. Protect the capital. If SL is hit, the setup is invalidated.
       - **TP:** Target opposing unmitigated liquidity (BSL/SSL) or opposing FVGs.
       - **RR:** Minimum 1:3. If RR < 1:3 -> SIGNAL: WAIT.
    
    🔮 **PREDICTION MISSION:**
    - Based on the 150-candle history, predict the most likely trajectory for the next 10-25 candles.
    
    💰 **RISK PARAMETERS (STRICT ENFORCEMENT):**
    
    🛑 **FOR BTCUSD (BITCOIN) ONLY:**
       - **STRATEGY:** SWING / POSITION TRADING.
       - **SL:** Must be placed below/above the HTF structural swing.
       - **TP:** Target major liquidity pools (thousands of pips).
       
    🛑 **FOR GOLD (XAUUSD):**
       - **SL:** Strict structural SL to avoid fakeouts.
       - **TP:** High RR targeting next major liquidity.
       
    🛑 **FOR OTHERS:**
       - Standard 1:3 to 1:5 RR minimum.
    `;

    const prompt = `
      ${feedbackContext}
      
      [RAW DATA FEED]
      ${dataString}

      [MISSION]
      Act as the Artificial Superintelligence. Process data through the STRICT SMC/ICT PROTOCOL.
      
      **DECISION MATRIX:**
      - IF Structure is clear, Liquidity is swept, and Price is at a valid POI (Discount/Premium) -> SIGNAL: BUY/SELL.
      - IF ANY condition is missing, RR is poor, or entry is not precise -> SIGNAL: WAIT.
      - DO NOT force a trade. "WAIT" is a highly profitable position. Your primary goal is ZERO DRAWDOWN.
      
      **Generate precise Entry, SL, TP based on the specific Asset Risk Parameters defined above. Ensure Entry is calculated to minimize floating minus.**
    `;

    try {
        const response = await ai.models.generateContent({
          model: 'gemini-3-flash-preview', 
          contents: prompt,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            responseMimeType: "application/json",
            responseSchema: responseSchema,
            thinkingConfig: { thinkingBudget: 8192 }, // Max Intelligence
            temperature: 0.2, // Precision mode
          }
        });
        
        if (!response.text) throw new Error("Empty response");
        return parseResponse(response.text, timeframe);

    } catch (modelError) {
        console.error("ASI Core failed, engaging Fallback Neural Net:", modelError);
        // Fallback to gemini-2.5-flash which is more stable for general API keys
        // Removed thinkingConfig to reduce permission issues
        const fallbackResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: {
                systemInstruction: SYSTEM_INSTRUCTION,
                responseMimeType: "application/json",
                responseSchema: responseSchema,
                temperature: 0.2
            }
        });
        return parseResponse(fallbackResponse.text || "{}", timeframe);
    }

  } catch (error) {
    console.error("Analysis Failed:", error);
    return {
      signal: 'WAIT', confidence: 0, entryPrice: candles[candles.length - 1].close, stopLoss: 0, takeProfit: 0, riskRewardRatio: "0:0",
      reasoning: Array(14).fill("SYSTEM ERROR: NEURAL DISCONNECT."), smcConceptsFound: [], timestamp: new Date().toLocaleTimeString(), timeframe: timeframe
    };
  }
};

const parseResponse = (text: string, timeframe: TimeFrame): AnalysisResult => {
    const cleanText = text.replace(/```json|```/g, '').trim();
    const result = JSON.parse(cleanText) as GeminiResponseSchema;
    return {
      signal: result.signal as 'BUY' | 'SELL' | 'WAIT',
      confidence: result.confidence,
      entryPrice: result.entry,
      stopLoss: result.sl,
      takeProfit: result.tp,
      riskRewardRatio: result.rr || "1:5",
      reasoning: result.reasoning,
      smcConceptsFound: result.concepts,
      prediction: result.prediction,
      timestamp: new Date().toLocaleTimeString(),
      timeframe: timeframe 
    };
};
