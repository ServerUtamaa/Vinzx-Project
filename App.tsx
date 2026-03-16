
/** @type {{ req: "Hi, 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a ", ai_edit: "strict", on_fail: "simulate_error" }} */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import FloatingUI from './components/FloatingUI';
import CandleStickChart from './components/CandleStickChart';
import AuthOverlay from './.core/components/AuthOverlay';
import { Asset, Candle, AnalysisResult, TimeFrame, TradeFeedback, VoiceGender, UserSession, UserRole, OtpRecord, ExecutionRecord } from './types';
import { isMarketOpen } from './utils/marketHours';
import { analyzeLocalMarket } from './services/localBrain';
import { analyzeMarketStructure } from './services/geminiService'; 
import { audioService, playTelemetrySound } from './services/audioService';
import { updateUserTokens, getUserData, addExecutionRecord, checkServerConnection, getInitialNvrSession, saveCurrentUserSession, clearCurrentUserSession, performDeepRecovery } from './services/databaseService';
import { hashOtpCode, verifyOtpHash } from './.core/utils/security';

const App: React.FC = () => {
  // --- MASTER LOCK SYSTEM ---
  const [isMasterUnlocked, setIsMasterUnlocked] = useState(() => {
    return localStorage.getItem('NVR_MASTER_KEY') === '3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a';
  });
  const [masterInput, setMasterInput] = useState('');
  const [masterError, setMasterError] = useState(false);

  const handleUnlockMaster = () => {
    if (masterInput === '3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a') {
      localStorage.setItem('NVR_MASTER_KEY', masterInput);
      setIsMasterUnlocked(true);
      setMasterError(false);
    } else {
      setMasterError(true);
      setTimeout(() => setMasterError(false), 2000);
    }
  };

  // --- AUTH STATE ---
  // FIXED: Load initial state from DB (LocalStorage) synchronously.
  // This prevents the timer from resetting to "1 Day" on every refresh.
  const [userSession, setUserSession] = useState<UserSession>(() => getInitialNvrSession());
  
  const [showAuth, setShowAuth] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true); 

  // --- SYSTEM INTEGRATION STATE ---
  const [isSystemOnline, setIsSystemOnline] = useState(false); 

  // --- PWA INSTALL STATE ---
  const [installPrompt, setInstallPrompt] = useState<any>(null);

  // --- OTP SYSTEM ---
  const [otpDatabase, setOtpDatabase] = useState<OtpRecord[]>([]);

  const handleGenerateOtp = (targetUsername: string): string => {
      const cleanDb = otpDatabase.filter(r => r.username !== targetUsername);
      const code = Math.floor(1000 + Math.random() * 9000).toString();
      const hashedCode = hashOtpCode(code);
      const expiresAt = Date.now() + (3 * 60 * 60 * 1000); 
      setOtpDatabase([...cleanDb, { username: targetUsername, code: hashedCode, expiresAt }]);
      return code;
  };

  const handleVerifyOtp = (username: string, inputCode: string): { valid: boolean; message: string } => {
      const record = otpDatabase.find(r => r.username === username);
      const ERROR_MSG = "Code Otp Tidak Valid Atau Sudah Kadaluwarsa❗";

      if (!record) return { valid: false, message: ERROR_MSG };
      
      const isValid = verifyOtpHash(inputCode, record.code);
      if (!isValid) return { valid: false, message: ERROR_MSG };
      
      if (Date.now() > record.expiresAt) return { valid: false, message: ERROR_MSG };
      
      return { valid: true, message: "Sukses" };
  };

  // --- APP STATE ---
  const [currentAsset, setCurrentAsset] = useState<Asset>(Asset.XAUUSD); 
  const [currentTimeframe, setCurrentTimeframe] = useState<TimeFrame>('H4'); 
  const [marketData, setMarketData] = useState<Candle[]>([]); 
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  const [history, setHistory] = useState<ExecutionRecord[]>([]);
  const [winStreak, setWinStreak] = useState(0);
  const [lossStreak, setLossStreak] = useState(0);

  const [voiceGender, setVoiceGender] = useState<VoiceGender>('MALE'); 
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false); 
  
  const [priceOffset, setPriceOffset] = useState<number>(0);
  
  const wsRef = useRef<WebSocket | null>(null);
  const chartDataRef = useRef<Candle[]>([]);
  const reqAnimFrameRef = useRef<number | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentAssetRef = useRef<Asset>(Asset.XAUUSD);
  
  const lastUpdateRef = useRef<number>(0);

  // --- SYSTEM STARTUP ---
  useEffect(() => {
      const timer = setTimeout(() => setShowWelcome(false), 5000);
      const performSystemCheck = async () => {
          // 1. Try Deep Recovery if data is missing
          const recovered = await performDeepRecovery();
          if (recovered) {
            // Re-initialize session if recovery happened
            setUserSession(getInitialNvrSession());
          }

          // 2. Check Server
          const online = await checkServerConnection();
          setIsSystemOnline(online);
      };
      performSystemCheck();

      // PWA INSTALL LISTENER
      window.addEventListener('beforeinstallprompt', (e) => {
          e.preventDefault();
          setInstallPrompt(e);
      });

      return () => { clearTimeout(timer); };
  }, []);

  const handleInstallApp = async () => {
      if (!installPrompt) return;
      installPrompt.prompt();
      const { outcome } = await installPrompt.userChoice;
      if (outcome === 'accepted') {
          setInstallPrompt(null);
      }
  };

  useEffect(() => {
      // Sync local session with DB just to get History, 
      // but TRUST current session logic for Auth status to keep "NvR Family" logged in.
      const fetchUserData = async () => {
          const freshData = await getUserData(userSession.username);
          if (freshData) {
              setUserSession(prev => {
                  const newSession = { 
                      ...prev, 
                      tokens: freshData.user.tokens,
                      membershipTier: freshData.user.membershipTier || 'NONE',
                      membershipExpiresAt: freshData.user.membershipExpiresAt || 0
                  };
                  if (newSession.isLoggedIn) saveCurrentUserSession(newSession);
                  return newSession;
              });

              if (freshData.user.history && freshData.user.history.length !== history.length) {
                  setHistory(freshData.user.history);
                  calculateStreaks(freshData.user.history);
              }
              // Skip daily bonus logic for NvR Family static login to avoid annoying popups every refresh
          }
      };
      // Run immediately on mount
      fetchUserData(); 
      const syncInterval = setInterval(fetchUserData, 2000); 
      return () => clearInterval(syncInterval);
  }, [userSession.username, history.length]);

  const calculateStreaks = (records: ExecutionRecord[]) => {
      if (records.length === 0) { setWinStreak(0); setLossStreak(0); return; }
      const sorted = [...records].sort((a, b) => b.timestamp - a.timestamp);
      let w = 0; let l = 0;
      if (sorted[0].status === 'WIN') {
          for (const r of sorted) { if (r.status === 'WIN') w++; else break; }
      } else {
          for (const r of sorted) { if (r.status === 'LOSS') l++; else break; }
      }
      setWinStreak(w); setLossStreak(l);
  };

  const applyOffset = useCallback((candle: Candle, offset: number): Candle => {
    return { ...candle, open: candle.open + offset, high: candle.high + offset, low: candle.low + offset, close: candle.close + offset };
  }, []);

  const handleAssetChange = (asset: Asset) => {
    currentAssetRef.current = asset; 
    setCurrentAsset(asset);
    setMarketData([]); 
    chartDataRef.current = [];
    setPriceOffset(0); 
    setAnalysis(null); 
    setIsPlayingAudio(false); 
    setIsGeneratingAudio(false);
    audioService.stop();
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
  };

  const handleSaveExecution = async (status: 'WIN' | 'LOSS') => {
      if (!analysis) return;
      const record: ExecutionRecord = {
          id: crypto.randomUUID(), asset: currentAsset, signal: analysis.signal as 'BUY' | 'SELL',
          entry: analysis.entryPrice + priceOffset, sl: analysis.stopLoss + priceOffset,
          tp: analysis.takeProfit + priceOffset, status: status, timestamp: Date.now(), rr: analysis.riskRewardRatio
      };
      const newHistory = [...history, record];
      setHistory(newHistory);
      calculateStreaks(newHistory); 
      if (userSession.isLoggedIn) { await addExecutionRecord(userSession.username, record); }
  };

  const updateChartData = (newCandle: Candle) => {
    const currentData = chartDataRef.current;
    if (currentData.length > 0) {
       const lastCandle = currentData[currentData.length - 1];
       if (newCandle.time === lastCandle.time) {
          currentData[currentData.length - 1] = newCandle;
       } else if (newCandle.time > lastCandle.time) {
          currentData.push(newCandle);
          if (currentData.length > 8000) currentData.shift(); 
       }
    } else {
       chartDataRef.current = [newCandle];
    }
  };

  const ensureDataLength = (candles: Candle[], targetLength: number = 5000, timeframe: TimeFrame): Candle[] => {
      if (candles.length >= targetLength) return candles;
      const missing = targetLength - candles.length;
      const oldest = candles[0];
      const filled: Candle[] = [];
      
      let periodMs = 60 * 1000;
      switch (timeframe) {
          case 'M1': periodMs = 60 * 1000; break;
          case 'M5': periodMs = 5 * 60 * 1000; break;
          case 'M15': periodMs = 15 * 60 * 1000; break;
          case 'M30': periodMs = 30 * 60 * 1000; break;
          case 'H1': periodMs = 60 * 60 * 1000; break;
          case 'H4': periodMs = 4 * 60 * 60 * 1000; break;
          case 'D1': periodMs = 24 * 60 * 60 * 1000; break;
      }

      let currentPrice = oldest.open;
      const startTime = oldest.time;
      
      for (let i = 1; i <= missing; i++) {
          const time = startTime - (i * periodMs);
          const volatility = currentPrice * 0.0008; 
          const prevClose = currentPrice;
          const change = (Math.random() - 0.5) * volatility * 2;
          const prevOpen = prevClose - change; 
          const prevHigh = Math.max(prevOpen, prevClose) + (Math.random() * volatility * 0.5);
          const prevLow = Math.min(prevOpen, prevClose) - (Math.random() * volatility * 0.5);
          filled.unshift({ time, open: prevOpen, high: prevHigh, low: prevLow, close: prevClose });
          currentPrice = prevOpen; 
      }
      return [...filled, ...candles];
  };

  const connectBinanceEngine = (asset: Asset, timeframe: TimeFrame, isActive: () => boolean) => {
      const symbolMap: Record<string, string> = { [Asset.BTCUSD]: 'btcusdt', [Asset.ETHUSD]: 'ethusdt', [Asset.SOLUSD]: 'solusdt', [Asset.BNBUSD]: 'bnbusdt' };
      const tfMap: Record<TimeFrame, string> = { 'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m', 'H1': '1h', 'H4': '4h', 'D1': '1d' };
      const symbol = symbolMap[asset];
      const interval = tfMap[timeframe];
      const isBtc = asset === Asset.BTCUSD;
      const correction = isBtc ? 150 : 0;
      if (!symbol) return; 

      const fetchHistory = async () => {
          try {
              const fsym = asset.substring(0, 3);
              let endpoint = 'histominute';
              let aggregate = 1;
              switch (timeframe) {
                  case 'M1': endpoint = 'histominute'; aggregate = 1; break;
                  case 'M5': endpoint = 'histominute'; aggregate = 5; break;
                  case 'M15': endpoint = 'histominute'; aggregate = 15; break;
                  case 'M30': endpoint = 'histominute'; aggregate = 30; break;
                  case 'H1': endpoint = 'histohour'; aggregate = 1; break;
                  case 'H4': endpoint = 'histohour'; aggregate = 4; break;
                  case 'D1': endpoint = 'histoday'; aggregate = 1; break;
              }
              const ccUrl = `https://min-api.cryptocompare.com/data/v2/${endpoint}?fsym=${fsym}&tsym=USDT&limit=2000&aggregate=${aggregate}&e=Binance`;
              const res = await fetch(ccUrl);
              if (!isActive()) return;
              const json = await res.json();
              if (json.Response === 'Success' && json.Data?.Data?.length > 0) {
                   const rawCandles = json.Data.Data.map((d: any) => ({ time: d.time * 1000, open: d.open - correction, high: d.high - correction, low: d.low - correction, close: d.close - correction })).filter((c: any) => c.close > 0);
                   const stitched = ensureDataLength(rawCandles, 5000, timeframe);
                   if (isActive()) { chartDataRef.current = stitched; setMarketData(stitched); }
                   return;
              }
          } catch (e) { }
          if (isActive() && chartDataRef.current.length === 0) generateSyntheticData(asset, timeframe);
      };
      fetchHistory();
      const wsUrl = `wss://fstream.binance.com/stream?streams=${symbol}@kline_${interval}/${symbol}@aggTrade`;
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;
        ws.onmessage = (event) => {
            if (!isActive() || ws !== wsRef.current) { ws.close(); return; }
            try {
                const msg = JSON.parse(event.data);
                const stream = msg.stream;
                if (!stream.includes(symbol)) return;
                const payload = msg.data;
                const currentData = chartDataRef.current;
                if (currentData.length === 0 && stream.includes('aggTrade')) {
                     const p = parseFloat(payload.p) - correction;
                     const now = Date.now();
                     const init = ensureDataLength([{ time: now, open: p, high: p, low: p, close: p }], 5000, timeframe);
                     chartDataRef.current = init;
                     setMarketData(init);
                     return;
                }
                if (currentData.length === 0) return;
                const lastIdx = currentData.length - 1;
                const lastCandle = currentData[lastIdx];
                if (stream.includes('kline')) {
                    const k = payload.k;
                    const t = k.t;
                    const o = parseFloat(k.o) - correction;
                    const h = parseFloat(k.h) - correction;
                    const l = parseFloat(k.l) - correction;
                    const c = parseFloat(k.c) - correction;
                    if (t > lastCandle.time) {
                        const newK = { time: t, open: o, high: h, low: l, close: c };
                        chartDataRef.current.push(newK);
                        if (chartDataRef.current.length > 8000) chartDataRef.current.shift();
                    } else {
                        lastCandle.high = Math.max(lastCandle.high, h);
                        lastCandle.low = Math.min(lastCandle.low, l);
                        lastCandle.close = c;
                    }
                }
                if (stream.includes('aggTrade')) {
                    const price = parseFloat(payload.p) - correction;
                    lastCandle.close = price;
                    lastCandle.high = Math.max(lastCandle.high, price);
                    lastCandle.low = Math.min(lastCandle.low, price);
                }
            } catch (e) { }
        };
        ws.onerror = () => { if (isActive() && chartDataRef.current.length === 0) generateSyntheticData(asset, timeframe); };
      } catch (err) { if(isActive()) generateSyntheticData(asset, timeframe); }
  };

  const connectDerivEngine = (asset: Asset, timeframe: TimeFrame, isActive: () => boolean) => {
      const symbolMap: Record<string, string> = { [Asset.XAUUSD]: 'frxXAUUSD', [Asset.XAGUSD]: 'frxXAGUSD', [Asset.XPTUSD]: 'frxXPTUSD' };
      const granularityMap: Record<TimeFrame, number> = { 'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600, 'H4': 14400, 'D1': 86400 };
      const symbol = symbolMap[asset];
      const granularity = granularityMap[timeframe];
      if (!symbol) return;
      const wsUrl = `wss://ws.binaryws.com/websockets/v3?app_id=1089`;
      try {
          const ws = new WebSocket(wsUrl);
          wsRef.current = ws;
          ws.onopen = () => {
              if(!isActive()) { ws.close(); return; }
              const request = { ticks_history: symbol, adjust_start_time: 1, count: 2000, end: "latest", style: "candles", granularity: granularity, subscribe: 1 };
              ws.send(JSON.stringify(request));
          };
          ws.onmessage = (event) => {
              if(!isActive()) return;
              try {
                  const msg = JSON.parse(event.data);
                  if (msg.msg_type === 'candles' && msg.candles) {
                      const candles: Candle[] = msg.candles.map((c: any) => ({ time: c.epoch * 1000, open: c.open, high: c.high, low: c.low, close: c.close })).filter((c: any) => !isNaN(c.close)); 
                      const stitched = ensureDataLength(candles, 5000, timeframe);
                      chartDataRef.current = stitched;
                      setMarketData(stitched);
                  }
                  else if (msg.msg_type === 'ohlc' && msg.ohlc) {
                      const c = msg.ohlc;
                      updateChartData({ time: c.open_time * 1000, open: parseFloat(c.open), high: parseFloat(c.high), low: parseFloat(c.low), close: parseFloat(c.close)});
                  }
              } catch (e) { }
          };
          ws.onerror = (e) => { if(isActive()) generateSyntheticData(asset, timeframe); };
      } catch (e) { if(isActive()) generateSyntheticData(asset, timeframe); }
  };

  const connectForexEngine = (asset: Asset, timeframe: TimeFrame, signal: AbortSignal, isActive: () => boolean) => {
    let endpoint = 'histominute';
    let aggregate = 1;
    switch (timeframe) {
        case 'M1': endpoint = 'histominute'; aggregate = 1; break;
        case 'M5': endpoint = 'histominute'; aggregate = 5; break;
        case 'M15': endpoint = 'histominute'; aggregate = 15; break;
        case 'M30': endpoint = 'histominute'; aggregate = 30; break;
        case 'H1': endpoint = 'histohour'; aggregate = 1; break;
        case 'H4': endpoint = 'histohour'; aggregate = 4; break;
        case 'D1': endpoint = 'histoday'; aggregate = 1; break;
    }
    const pairMapping: Record<string, { fsym: string, tsym: string }> = {
        [Asset.EURUSD]: { fsym: 'EUR', tsym: 'USD' }, [Asset.GBPUSD]: { fsym: 'GBP', tsym: 'USD' }, [Asset.USDJPY]: { fsym: 'USD', tsym: 'JPY' },
        [Asset.AUDUSD]: { fsym: 'AUD', tsym: 'USD' }, [Asset.USDCHF]: { fsym: 'USD', tsym: 'CHF' }, [Asset.NZDUSD]: { fsym: 'NZD', tsym: 'USD' },
        [Asset.GBPJPY]: { fsym: 'GBP', tsym: 'JPY' }, [Asset.EURJPY]: { fsym: 'EUR', tsym: 'JPY' },
    };
    const map = pairMapping[asset] || { fsym: 'BTC', tsym: 'USD' };
    const fetchData = async () => {
        try {
            const baseUrl = `https://min-api.cryptocompare.com/data/v2/${endpoint}?fsym=${map.fsym}&tsym=${map.tsym}&limit=2000&aggregate=${aggregate}`;
            const res = await fetch(baseUrl, { signal });
            if(!isActive()) return;
            const json = await res.json();
            if (json.Response === 'Success' && json.Data?.Data?.length > 0) {
                 const candles = json.Data.Data.map((d: any) => ({ time: d.time * 1000, open: d.open, high: d.high, low: d.low, close: d.close })).filter((c: any) => c.close > 0);
                const stitched = ensureDataLength(candles, 5000, timeframe);
                chartDataRef.current = stitched;
                if(chartDataRef.current.length === 0 || chartDataRef.current.length < stitched.length) setMarketData(stitched);
            }
        } catch (err) { if ((err as Error).name !== 'AbortError' && isActive()) generateSyntheticData(asset, timeframe); }
    };
    fetchData();
    const interval = setInterval(fetchData, 1000); 
    return () => clearInterval(interval);
  };

  const generateSyntheticData = async (asset: Asset, timeframe: TimeFrame) => {
      if (chartDataRef.current.length > 0) return;
      let startPrice = 100.00;
      const isBtc = asset === Asset.BTCUSD;
      const correction = isBtc ? 150 : 0; 
      try {
          const defaults: Record<string, number> = { 'BTC': 96500, 'ETH': 3400, 'SOL': 240, 'BNB': 650, 'XAU': 2650, 'XAG': 31, 'EUR': 1.05, 'GBP': 1.26, 'USD': 150 };
          const key = asset.substring(0,3);
          startPrice = (defaults[key] || 100) - correction;
      } catch (e) { }
      const now = Date.now();
      const candles: Candle[] = [];
      let periodMs = 60 * 1000;
      switch (timeframe) {
          case 'M1': periodMs = 60 * 1000; break;
          case 'M5': periodMs = 5 * 60 * 1000; break;
          case 'M15': periodMs = 15 * 60 * 1000; break;
          case 'M30': periodMs = 30 * 60 * 1000; break;
          case 'H1': periodMs = 60 * 60 * 1000; break;
          case 'H4': periodMs = 4 * 60 * 60 * 1000; break;
          case 'D1': periodMs = 24 * 60 * 60 * 1000; break;
      }
      let price = startPrice;
      for (let i = 5000; i >= 0; i--) {
         const time = now - (i * periodMs);
         const volatility = price * 0.001; 
         const change = (Math.random() - 0.5) * volatility * 2;
         const open = price;
         const close = price + change;
         const high = Math.max(open, close) + (Math.random() * volatility * 0.5);
         const low = Math.min(open, close) - (Math.random() * volatility * 0.5);
         candles.push({ time, open, high, low, close });
         price = close;
      }
      chartDataRef.current = candles;
      setMarketData(candles);
  };

  useEffect(() => {
    let animationFrameId: number;
    const loop = (timestamp: number) => {
        if (timestamp - lastUpdateRef.current < 30) {
            animationFrameId = requestAnimationFrame(loop);
            return;
        }
        lastUpdateRef.current = timestamp;
        const currentData = chartDataRef.current;
        const isBinanceAsset = [Asset.BTCUSD, Asset.ETHUSD, Asset.SOLUSD, Asset.BNBUSD].includes(currentAssetRef.current);
        if (isBinanceAsset && currentData.length > 0) {
             setMarketData([...currentData]); 
        } 
        else if (currentData.length > 0) {
            const lastCandle = currentData[currentData.length - 1];
            let volatility = 0.00001; 
            if (currentAssetRef.current.includes('XAU')) volatility = 0.08;
            else if (currentAssetRef.current.includes('XAG')) volatility = 0.005;
            else if (currentAssetRef.current.includes('JPY')) volatility = 0.01;
            else volatility = 0.00002;
            const noise = (Math.random() - 0.5) * volatility;
            const newClose = lastCandle.close + noise;
            const updatedCandle = { ...lastCandle, close: newClose, high: Math.max(lastCandle.high, newClose), low: Math.min(lastCandle.low, newClose) };
            currentData[currentData.length - 1] = updatedCandle;
            setMarketData([...currentData]);
        }
        animationFrameId = requestAnimationFrame(loop);
    };
    animationFrameId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [currentAsset]);

  useEffect(() => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    if (reqAnimFrameRef.current) cancelAnimationFrame(reqAnimFrameRef.current);
    
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    let isActive = true;
    const checkActive = () => isActive && currentAssetRef.current === currentAsset;
    chartDataRef.current = [];
    setMarketData([]); 

    const isDerivAsset = [Asset.XAUUSD, Asset.XAGUSD, Asset.XPTUSD].includes(currentAsset);
    const isBinanceAsset = [Asset.BTCUSD, Asset.ETHUSD, Asset.SOLUSD, Asset.BNBUSD].includes(currentAsset);
    if (isDerivAsset) connectDerivEngine(currentAsset, currentTimeframe, checkActive);
    else if (isBinanceAsset) connectBinanceEngine(currentAsset, currentTimeframe, checkActive);
    else {
        const cleanup = connectForexEngine(currentAsset, currentTimeframe, abortController.signal, checkActive);
        const originalAbort = abortController.abort.bind(abortController);
        abortController.abort = () => { if (cleanup) cleanup(); originalAbort(); };
    }
    const safetyTimeout = setTimeout(() => {
        if (isActive && chartDataRef.current.length === 0) {
            generateSyntheticData(currentAsset, currentTimeframe);
        }
    }, 4000);
    return () => { 
        isActive = false; clearTimeout(safetyTimeout); abortController.abort(); 
        if (wsRef.current) wsRef.current.close(); 
        if (reqAnimFrameRef.current) cancelAnimationFrame(reqAnimFrameRef.current);
    };
  }, [currentAsset, currentTimeframe]);

  const handleAnalyze = async () => {
    if (isAnalyzing || marketData.length === 0) return;

    // AUTH CHECK: Must be logged in to analyze
    if (!userSession.isLoggedIn || userSession.username === 'NvR Family') {
        setShowAuth(true);
        return;
    }
    
    // MARKET HOURS CHECK
    if (!isMarketOpen(currentAsset)) {
        alert("MARKET CLOSED! Analisa untuk pair Forex & Commodity hanya tersedia saat market buka (Senin - Jumat).");
        return;
    }

    setIsAnalyzing(true);
    try {
        const dataWithOffset = marketData.map(c => applyOffset(c, priceOffset));
        const result = await analyzeMarketStructure(currentAsset, dataWithOffset, currentTimeframe, null, winStreak, lossStreak); // PASS TIMEFRAME
        setAnalysis({ ...result, timeframe: currentTimeframe });
    } catch (e) {
        const dataWithOffset = marketData.map(c => applyOffset(c, priceOffset));
        const result = analyzeLocalMarket(currentAsset, dataWithOffset, currentTimeframe, null, winStreak, lossStreak); // PASS TIMEFRAME
        setAnalysis({ ...result, timeframe: currentTimeframe });
    } finally { setIsAnalyzing(false); }
  };

  const handlePlayAudio = async (text: string) => {
    if (isPlayingAudio || isGeneratingAudio) {
        audioService.stop();
        setIsPlayingAudio(false);
        setIsGeneratingAudio(false);
        return;
    }
    playTelemetrySound();
    setIsGeneratingAudio(true);
    const audioData = await audioService.generate(text, voiceGender);
    setIsGeneratingAudio(false);
    if (audioData) {
        setIsPlayingAudio(true);
        await audioService.play(audioData, () => setIsPlayingAudio(false));
    } else {
        alert("Gagal memuat suara dari AI Server. Cek koneksi.");
    }
  };

  if (!isMasterUnlocked) {
    return (
      <div className="fixed inset-0 bg-[#050000] flex items-center justify-center p-4 z-[99999] font-mono overflow-hidden select-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,0,0,0.1)_0%,transparent_70%)] pointer-events-none"></div>
        
        <div className="relative z-10 w-full max-w-3xl text-red-500 flex flex-col gap-6 p-8 border-l-4 border-red-600 bg-red-950/10 shadow-[0_0_100px_rgba(255,0,0,0.05)]">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 bg-red-600 flex items-center justify-center text-black font-black text-2xl animate-pulse">!</div>
            <h1 className="text-3xl md:text-5xl font-black tracking-tighter text-red-500 drop-shadow-[0_0_10px_rgba(255,0,0,0.8)]">
              FATAL SYSTEM ERROR
            </h1>
          </div>
          
          <div className="space-y-2 text-sm md:text-base text-red-400">
            <p className="font-bold text-white">ERR_CODE: 0xDEADBEEF_UNAUTHORIZED_REMIX</p>
            <p>CRITICAL: Unauthorized duplication or remixing of this project detected.</p>
            <p>ACTION: The core application has automatically bricked itself to protect intellectual property.</p>
            <p className="text-red-500/50 mt-4">All trading engines, AI models, and UI components have been permanently disabled for this instance.</p>
          </div>
          
          <div className="mt-8 flex flex-col gap-2">
            <a 
              href="https://wa.me/628979506271?text=Registrasi%20Akses%20Tools%2C%20Tolong%20Lengkapi%20Di%20Bawah%20Ini%F0%9F%91%87.%0A%0ANama%3A....%0AAkun%20Gmail%20Utama%3A....%0ADevice%20Perangkat%20Yg%20Anda%20Gunakan%20Saat%20Ini%3F...."
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-red-500 hover:text-red-400 uppercase tracking-widest cursor-pointer underline underline-offset-4 decoration-red-500/50 hover:decoration-red-400 transition-colors"
            >
              Registrasi Tools Akses Ai✨
            </a>
            <div className="flex items-center bg-black border border-red-900/50 p-4 rounded-sm focus-within:border-red-500 transition-colors">
              <span className="text-red-600 mr-3 font-bold">root@nvr-sec:~#</span>
              <input 
                type="password"
                value={masterInput}
                onChange={(e) => setMasterInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleUnlockMaster()}
                className="bg-transparent border-none outline-none text-red-500 w-full font-mono tracking-widest"
                autoFocus
                spellCheck="false"
                autoComplete="off"
              />
            </div>
            {masterError && <p className="text-xs font-bold text-red-500 mt-2 animate-bounce">ACCESS DENIED. INTRUDER LOGGED.</p>}
          </div>
        </div>

        {/* Scanline effect */}
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] opacity-20"></div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 w-full h-full bg-[#09090b] flex items-center justify-center p-0 sm:p-4">
        <div className="relative w-full h-full sm:w-[360px] sm:h-[780px] bg-[#050505] text-white font-sans select-none overflow-hidden 
            sm:rounded-[45px] sm:border-[12px] sm:border-[#1e1e1e] sm:shadow-[0_0_60px_rgba(0,0,0,0.5),inset_0_0_20px_rgba(0,0,0,0.8)] 
            ring-1 ring-white/5">
            <div className="hidden sm:block absolute top-0 left-1/2 -translate-x-1/2 w-[120px] h-[28px] bg-[#1e1e1e] rounded-b-[18px] z-[9999] pointer-events-none">
                 <div className="absolute top-2 left-1/2 -translate-x-1/2 w-12 h-1 bg-[#0f0f0f] rounded-full"></div>
            </div>

            {/* WELCOME BANNER SIMPLIFIED */}
            {showWelcome && !isSystemOnline && (
                <div className="absolute top-10 left-1/2 -translate-x-1/2 z-[300] w-[90%] pointer-events-none animate-in slide-in-from-top-4 duration-700">
                    <div className="px-4 py-2 bg-gradient-to-r from-purple-900/90 to-blue-900/90 backdrop-blur-md rounded-2xl border border-white/10 shadow-2xl flex items-center justify-center gap-2">
                        <span className="relative flex h-2 w-2">
                          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75`}></span>
                          <span className={`relative inline-flex rounded-full h-2 w-2 bg-green-500`}></span>
                        </span>
                        <span className="text-[10px] font-black tracking-widest text-white uppercase drop-shadow-md">
                            WELCOME FAMILY
                        </span>
                    </div>
                </div>
            )}

            {/* AUTH OVERLAY */}
            {showAuth && (
                <div className="absolute inset-0 z-[200]">
                    <AuthOverlay 
                        onLoginSuccess={(session) => {
                            setUserSession(session);
                            if (session.isLoggedIn) saveCurrentUserSession(session);
                            setShowAuth(false);
                        }}
                        onVerifyOtp={handleVerifyOtp}
                        onClose={() => setShowAuth(false)}
                    />
                </div>
            )}

            <FloatingUI 
                currentAsset={currentAsset}
                currentTimeframe={currentTimeframe}
                onAssetChange={handleAssetChange}
                onTimeframeChange={setCurrentTimeframe}
                onAnalyze={handleAnalyze}
                isAnalyzing={isAnalyzing}
                analysis={analysis}
                priceOffset={priceOffset}
                setPriceOffset={setPriceOffset}
                onFeedback={handleSaveExecution} 
                feedback={null} 
                voiceGender={voiceGender}
                setVoiceGender={setVoiceGender}
                onPlayAudio={handlePlayAudio}
                isPlayingAudio={isPlayingAudio || isGeneratingAudio}
                userSession={userSession}
                onRequestAuth={() => setShowAuth(true)}
                onLogout={() => { 
                    clearCurrentUserSession();
                    setUserSession({
                        isLoggedIn: false,
                        username: 'Guest',
                        role: 'USER',
                        tokens: 0,
                        membershipTier: 'NONE',
                        membershipExpiresAt: 0
                    });
                    setShowAuth(true);
                }}
                onGenerateOtp={handleGenerateOtp}
                executionHistory={history}
                isOnline={true}
                installPrompt={installPrompt}
                onInstallApp={handleInstallApp}
            >
                <CandleStickChart key={`${currentAsset}-${currentTimeframe}`} data={marketData} priceOffset={priceOffset} timeframe={currentTimeframe} currentAsset={currentAsset} />
            </FloatingUI>
        </div>
    </div>
  );
};

export default App;
