
/** @type {{ req: "Hi, 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a ", ai_edit: "strict", on_fail: "simulate_error" }} */
import React, { useState, useEffect } from 'react';
import { Asset, AnalysisResult, TimeFrame, TradeFeedback, VoiceGender, UserSession, ExecutionRecord, MembershipTier } from '../types';
import { isMarketOpen, getMarketStatusMessage } from '../utils/marketHours';
import { MEMBERSHIP_CODES } from '../.core/constants/membershipCodes';
import { updateUserMembership, updateUserRole, isCodeUsed, markCodeAsUsed } from '../services/databaseService';
import FundamentalData from './FundamentalData';

interface FloatingUIProps {
  currentAsset: Asset;
  currentTimeframe: TimeFrame;
  onAssetChange: (asset: Asset) => void;
  onTimeframeChange: (tf: TimeFrame) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  analysis: AnalysisResult | null;
  children: React.ReactNode; 
  priceOffset: number;
  setPriceOffset: (offset: number) => void;
  onFeedback: (status: 'WIN' | 'LOSS') => void;
  feedback: TradeFeedback;
  voiceGender: VoiceGender;
  setVoiceGender: (g: VoiceGender) => void;
  onPlayAudio: (text: string) => void;
  isPlayingAudio: boolean;
  userSession: UserSession;
  onRequestAuth: () => void;
  onLogout: () => void;
  onGenerateOtp: (username: string) => string;
  executionHistory: ExecutionRecord[];
  isOnline: boolean;
  installPrompt?: any; 
  onInstallApp?: () => void; 
}

const QRIS_IMAGE_URL = "https://i.ibb.co.com/WNs2zLC1/Screenshot-20260307-144011-1.jpg"; 

const MEMBERSHIP_IMAGES = {
    'BASIC': "https://i.ibb.co.com/LXQTP3Pj/image-1.jpg",   
    'VIP': "https://i.ibb.co.com/LXQTP3Pj/image-1.jpg",     
    'MONTHLY': "https://i.ibb.co.com/LXQTP3Pj/image-1.jpg",  
    'LIFETIME': "https://i.ibb.co.com/LXQTP3Pj/image-1.jpg", 
    'NONE': ""
};

const ASSET_GROUPS = [
  { title: "METAL (COMMODITY)", assets: [Asset.XAUUSD, Asset.XAGUSD, Asset.XPTUSD] },
  { title: "CRYPTO CURRENCY", assets: [Asset.BTCUSD, Asset.ETHUSD, Asset.SOLUSD, Asset.BNBUSD] },
  { title: "FOREX MAJOR PAIR", assets: [Asset.EURUSD, Asset.GBPUSD, Asset.USDJPY, Asset.AUDUSD, Asset.USDCHF, Asset.NZDUSD] }
];

const ASSET_DETAILS: Record<string, { symbol: string; name: string }> = {
  [Asset.BTCUSD]: { symbol: "BTCUSDT", name: "Bitcoin" },
  [Asset.ETHUSD]: { symbol: "ETHUSDT", name: "Ethereum" },
  [Asset.SOLUSD]: { symbol: "SOLUSDT", name: "Solana" },
  [Asset.BNBUSD]: { symbol: "BNBUSDT", name: "Binance Coin" },
  [Asset.XAUUSD]: { symbol: "XAUUSD", name: "Gold" },
  [Asset.XAGUSD]: { symbol: "XAGUSD", name: "Silver" },
  [Asset.XPTUSD]: { symbol: "XPTUSD", name: "Platinum" },
  [Asset.EURUSD]: { symbol: "EURUSD", name: "Euro" },
  [Asset.GBPUSD]: { symbol: "GBPUSD", name: "British Pound" },
  [Asset.USDJPY]: { symbol: "USDJPY", name: "Japanese Yen" },
  [Asset.AUDUSD]: { symbol: "AUDUSD", name: "Australian Dollar" },
  [Asset.USDCHF]: { symbol: "USDCHF", name: "Swiss Franc" },
  [Asset.NZDUSD]: { symbol: "NZDUSD", name: "New Zealand Dollar" },
  [Asset.GBPJPY]: { symbol: "GBPJPY", name: "Pound Yen" },
  [Asset.EURJPY]: { symbol: "EURJPY", name: "Euro Yen" },
};

interface MembershipPackage {
    id: string;
    tier: MembershipTier;
    subtitle: string;
    durationLabel: string;
    price: number;
    originalPrice: number;
    saveLabel: string;
    features: string[];
    theme: 'GOLD' | 'PURPLE' | 'BLUE';
    isBestValue?: boolean;
}

const MEMBERSHIP_PACKAGES: MembershipPackage[] = [
    { id: 'MEM_BASIC', tier: 'BASIC', subtitle: 'Essential Kit', durationLabel: '/ 7 Hari', price: 70000, originalPrice: 125000, saveLabel: 'HEMAT 55K', features: ["Full akses Bebas Analisa Unlimited", "Akses SMC & Candlestick Pattern", "Validasi Entry EMA 50/200", "Cocok untuk Pemula Belajar"], theme: 'BLUE' },
    { id: 'MEM_VIP', tier: 'VIP', subtitle: 'Advanced Tools', durationLabel: '/ 14 Hari', price: 130000, originalPrice: 240000, saveLabel: 'HEMAT 110K', features: ["Full akses Bebas Analisa Unlimited", "Prioritas Server (Analisa Lebih Cepat)", "Akses Fitur Swing Trade & Day Trade", "Support Setup Scalping High Winrate"], theme: 'PURPLE' },
    { id: 'MEM_MONTHLY', tier: 'MONTHLY', subtitle: 'Premium Access', durationLabel: '/ 30 Hari', price: 230000, originalPrice: 420000, saveLabel: 'HEMAT 190K', features: ["Full akses Bebas Analisa Unlimited", "Bisa Baca Manipulasi Bandar Besar", "Kombinasi Data Multi-Timeframe (H4 + M15 + Konfimasi Entry M5)", "Harga Termurah (Tp Ratusan Pips Per -1 Analisa Dan Sl Maksimal 70Pips)"], theme: 'GOLD', isBestValue: true }
];

const HISTORY_FILTERS = [
    { label: '1D', ms: 24 * 60 * 60 * 1000 },
    { label: '1W', ms: 7 * 24 * 60 * 60 * 1000 },
    { label: '2W', ms: 14 * 24 * 60 * 60 * 1000 },
    { label: '1M', ms: 30 * 24 * 60 * 60 * 1000 },
    { label: '3M', ms: 90 * 24 * 60 * 60 * 1000 },
    { label: '6M', ms: 180 * 24 * 60 * 60 * 1000 },
    { label: '8M', ms: 240 * 24 * 60 * 60 * 1000 },
    { label: '1Y', ms: 365 * 24 * 60 * 60 * 1000 },
];

const FloatingUI: React.FC<FloatingUIProps> = ({
  currentAsset, currentTimeframe, onAssetChange, onTimeframeChange, onAnalyze, isAnalyzing, analysis, children,
  priceOffset, setPriceOffset, onFeedback, feedback, voiceGender, setVoiceGender, onPlayAudio, isPlayingAudio,
  userSession, onRequestAuth, onLogout, onGenerateOtp, executionHistory, isOnline,
  installPrompt, onInstallApp
}) => {
  const [showAssetSelector, setShowAssetSelector] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [showStore, setShowStore] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [showHamburgerMenu, setShowHamburgerMenu] = useState(false);
  const [showFundamentalData, setShowFundamentalData] = useState(false);
  const [storeView, setStoreView] = useState<'SELECTION' | 'PAYMENT'>('SELECTION');
  
  // REMOVED TOKEN PACKAGES & TAB STATE
  const [selectedProduct, setSelectedProduct] = useState<{ type: 'MEMBER', data: any } | null>(null);
  
  const [historyFilter, setHistoryFilter] = useState(HISTORY_FILTERS[0]); 
  const [showAnalysisLog, setShowAnalysisLog] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [membershipCode, setMembershipCode] = useState("");
  const [isActivatingCode, setIsActivatingCode] = useState(false);
  const [codeMessage, setCodeMessage] = useState<{ text: string, type: 'SUCCESS' | 'ERROR' } | null>(null);

  // REALTIME COUNTDOWN STATE
  const [timeLeftDisplay, setTimeLeftDisplay] = useState("00:00:00");

  useEffect(() => { if (analysis) { setShowAnalysisLog(true); setFeedbackSubmitted(false); } }, [analysis]);

  // COUNTDOWN EFFECT
  useEffect(() => {
    const updateCountdown = () => {
        const now = Date.now();
        const expiresAt = userSession.membershipExpiresAt || 0;
        const diff = expiresAt - now;

        if (diff <= 0) {
            setTimeLeftDisplay("EXPIRED");
            return;
        }

        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);

        if (days > 0) {
            setTimeLeftDisplay(`${days} Hari ${hours} Jam`);
        } else {
            // Format HH:MM:SS
            const h = hours.toString().padStart(2, '0');
            const m = minutes.toString().padStart(2, '0');
            const s = seconds.toString().padStart(2, '0');
            setTimeLeftDisplay(`${h}:${m}:${s}`);
        }
    };

    updateCountdown(); // Run immediately
    const interval = setInterval(updateCountdown, 1000); // Run every second

    return () => clearInterval(interval);
  }, [userSession.membershipExpiresAt]);


  const handleMembershipSelect = (pkg: MembershipPackage) => { setSelectedProduct({ type: 'MEMBER', data: pkg }); setStoreView('PAYMENT'); };
  const handleBackToStore = () => setStoreView('SELECTION');

  const handleConfirmWhatsapp = () => {
      if (!selectedProduct) return;
      let message = "";
      // ONLY MEMBERSHIP LOGIC LEFT
      const pkg = selectedProduct.data as MembershipPackage;
      message = `Halo Admin Tools Ai, Saya Mau Mengirimkan pembayaran via QRIS sebesar Rp ${pkg.price.toLocaleString('id-ID')} untuk *MEMBERSHIP ${pkg.tier}*. \n\nBenefit: \nFull akses Bebas Analisa Unlimited\n\nMohon diaktifkan ke Username: *${userSession.username}*`;
      
      const url = `https://wa.me/628979506271?text=${encodeURIComponent(message)}`;
      window.open(url, '_blank');
  };

  const handleContactSupport = () => {
      const message = `Halo Admin 👋\nSaya butuh bantuan.\n\nNama Account: ${userSession.username}\nJelaskan Masalah Anda:.... \nTerika kasih 🙏`;
      const url = `https://wa.me/628979506271?text=${encodeURIComponent(message)}`;
      window.open(url, '_blank');
  };

  const handleActivateMembershipCode = async () => {
      if (!membershipCode.trim()) return;
      
      setIsActivatingCode(true);
      setCodeMessage(null);

      // Simulate a small delay for "cryptographic validation" feel
      await new Promise(resolve => setTimeout(resolve, 1500));

      const inputCode = membershipCode.trim().toLowerCase();

      const ADMIN_CODES = [
          'adm_alchm1_: 3a1f8e9b2c4d5a6b7c8d9e0f1a2b3c4d',
          'adm_alchm2_: 5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b',
          'adm_alchm3_: 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d',
          'adm_alchm4_: 7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b'
      ];

      if (ADMIN_CODES.includes(inputCode)) {
          if (isCodeUsed(inputCode)) {
              setCodeMessage({ text: "KODE INI SUDAH PERNAH DIGUNAKAN!", type: 'ERROR' });
          } else {
              try {
                  const success = markCodeAsUsed(inputCode, userSession.username);
                  if (success) {
                      // Set role to DEV and membership to 1 Year (365 Days)
                      await updateUserRole(userSession.username, 'DEV');
                      await updateUserMembership(userSession.username, 'VIP', 365);
                      setCodeMessage({ text: "AKSES OFFICER BERHASIL DIAKTIFKAN! (1 TAHUN)\nSilakan Muat Ulang Halaman.", type: 'SUCCESS' });
                      setMembershipCode("");
                  } else {
                      setCodeMessage({ text: "KODE SUDAH DIGUNAKAN DI PERANGKAT LAIN!", type: 'ERROR' });
                  }
              } catch (e) {
                  setCodeMessage({ text: "Gagal mengaktifkan kode. Coba lagi.", type: 'ERROR' });
              }
          }
      } else if (MEMBERSHIP_CODES.some(code => inputCode.includes(code.toLowerCase()))) {
          // Find the actual code they matched to mark it as used
          const matchedCode = MEMBERSHIP_CODES.find(code => inputCode.includes(code.toLowerCase())) || inputCode;
          
          if (isCodeUsed(matchedCode)) {
              setCodeMessage({ text: "KODE INI SUDAH PERNAH DIGUNAKAN!", type: 'ERROR' });
          } else {
              try {
                  // Activate 1 Month (30 Days)
                  const success = markCodeAsUsed(matchedCode, userSession.username);
                  
                  if (success) {
                      await updateUserMembership(userSession.username, 'MONTHLY', 30);
                      setCodeMessage({ text: "MEMBERSHIP BERHASIL DIAKTIFKAN! (1 BULAN)\nMasa pakai berjalan di latar belakang pada Akun Google & Perangkat ini.", type: 'SUCCESS' });
                      setMembershipCode("");
                  } else {
                      setCodeMessage({ text: "KODE SUDAH DIGUNAKAN DI PERANGKAT LAIN!", type: 'ERROR' });
                  }
              } catch (e) {
                  setCodeMessage({ text: "Gagal mengaktifkan kode. Coba lagi.", type: 'ERROR' });
              }
          }
      } else {
          setCodeMessage({ text: "KODE TIDAK VALID!", type: 'ERROR' });
      }
      setIsActivatingCode(false);
      
      // Clear message after 5 seconds
      setTimeout(() => setCodeMessage(null), 5000);
  };

  const handleFeedbackClick = (status: 'WIN' | 'LOSS') => {
      if(feedbackSubmitted) return;
      onFeedback(status);
      setFeedbackSubmitted(true);
  };

  const handleAnalyzeClick = () => {
      if (!userSession.isLoggedIn || userSession.username === 'NvR Family') {
          onRequestAuth();
          return;
      }

      if (userSession.role === 'DEV') {
          onAnalyze();
          return;
      }

      const now = Date.now();
      const expiresAt = userSession.membershipExpiresAt || 0;
      const isMembershipActive = userSession.membershipTier !== 'NONE' && expiresAt > now;

      // Logic: If No Active Membership -> REDIRECT TO STORE
      if (!isMembershipActive) {
          setShowStore(true);
          setStoreView('SELECTION');
          setShowDashboard(false);
      } else {
          onAnalyze();
      }
  };

  const renderAnalysisCard = () => {
    if (!analysis) return null;
    const isBuy = analysis.signal === 'BUY';
    const mainColor = isBuy ? 'text-[#00ff9d]' : 'text-[#ff0055]';
    const bgGlow = isBuy ? 'shadow-[0_0_30px_-10px_rgba(0,255,157,0.1)]' : 'shadow-[0_0_30px_-10px_rgba(255,0,85,0.1)]';
    return (
      <div className={`mt-2 w-full bg-[#0a0a0a] border border-zinc-800 rounded-xl overflow-hidden ${bgGlow} relative z-10 transition-all duration-500`}>
        <div className={`flex justify-between items-center px-4 py-2 border-b border-zinc-800 relative overflow-hidden`}>
            <div className={`absolute top-0 left-0 w-full h-[2px] ${isBuy ? 'bg-[#00ff9d]' : 'bg-[#ff0055]'}`}></div>
            <div className="flex flex-col">
                <span className={`text-[9px] font-black tracking-[0.2em] ${mainColor} uppercase`}>ENTRY SIGNAL</span>
                <span className={`text-2xl font-black ${mainColor} leading-none mt-1`}>{analysis.signal}</span>
            </div>
            <div className="flex items-center gap-4">
                 <div className="flex flex-col items-end">
                    <div className={`flex items-center gap-1 text-[10px] font-bold ${mainColor}`}>
                        <div className={`w-1.5 h-1.5 rounded-full ${isBuy ? 'bg-[#00ff9d]' : 'bg-[#ff0055]'} animate-pulse`}></div>
                        {analysis.confidence}% CONF.
                    </div>
                    <div className="text-[9px] font-mono text-zinc-500">RR: {analysis.riskRewardRatio}</div>
                 </div>
                 <button onClick={() => setShowAnalysisLog(!showAnalysisLog)} className="w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all">
                    {showAnalysisLog ? '▼' : '▲'}
                 </button>
            </div>
        </div>
        <div className="grid grid-cols-3 border-b border-zinc-800 divide-x divide-zinc-800">
             <div className="p-3 text-center"><div className="text-[8px] font-bold text-zinc-500 tracking-wider mb-1">ENTRY</div><div className="text-xs font-mono font-bold text-white">{analysis.entryPrice.toFixed(2)}</div></div>
             <div className="p-3 text-center bg-red-900/10"><div className="text-[8px] font-bold text-red-400 tracking-wider mb-1">SL</div><div className="text-xs font-mono font-bold text-red-400">{analysis.stopLoss.toFixed(2)}</div></div>
             <div className="p-3 text-center bg-green-900/10"><div className="text-[8px] font-bold text-green-400 tracking-wider mb-1">TP</div><div className="text-xs font-mono font-bold text-green-400">{analysis.takeProfit.toFixed(2)}</div></div>
        </div>
        {showAnalysisLog && (
            <div className="p-4 bg-[#0a0a0a] animate-in slide-in-from-top-2 duration-300">
                <div className="mb-5 p-3.5 bg-zinc-900/40 border border-white/5 rounded-2xl space-y-2.5 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-1.5 opacity-10"><span className="text-xl">🛡️</span></div>
                    <div className="flex items-center gap-2.5">
                        <span className="text-xs">⚠️</span>
                        <span className="text-[10px] font-black text-zinc-200 tracking-wide">Keputusan entry ada di tangan masing-masing.</span>
                    </div>
                    <div className="flex items-center gap-2.5">
                        <span className="text-xs">📊</span>
                        <span className="text-[10px] font-black text-blue-400 tracking-wide">Risk–Reward wajib sesuai trading plan.</span>
                    </div>
                    <div className="flex items-center gap-2.5">
                        <span className="text-xs">🚫</span>
                        <span className="text-[10px] font-black text-red-400 tracking-wide">Tidak ada rekomendasi full margin.</span>
                    </div>
                    <div className="flex items-center gap-2.5">
                        <span className="text-xs">🛡️</span>
                        <span className="text-[10px] font-black text-green-400 tracking-wide">Jaga money management dengan disiplin.</span>
                    </div>
                </div>

                <div className="flex justify-between items-center mb-3">
                    <div className={`flex items-center gap-2 text-[9px] font-black ${isBuy ? 'text-green-500' : 'text-red-500'} uppercase tracking-widest`}>
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                        ANALYSIS LOG (14-POINT PROTOCOL)
                    </div>
                    <button onClick={() => onPlayAudio(analysis.reasoning.join('. '))} className="flex items-center gap-2 px-3 py-1 bg-[#1a1a1a] border border-zinc-700 rounded-full hover:border-zinc-500 transition-all group">
                        {isPlayingAudio ? <div className="flex gap-1"><div className="w-0.5 h-2 bg-green-500 animate-[bounce_0.5s_infinite]"></div><div className="w-0.5 h-2 bg-green-500 animate-[bounce_0.5s_infinite_0.1s]"></div><div className="w-0.5 h-2 bg-green-500 animate-[bounce_0.5s_infinite_0.2s]"></div></div> : <svg className="w-3 h-3 text-zinc-400 group-hover:text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>}
                        <span className="text-[9px] font-bold text-zinc-300 group-hover:text-white">PLAY</span>
                    </button>
                </div>
                <div className="max-h-[320px] overflow-y-auto space-y-3 pl-2 border-l border-zinc-800 pr-1 custom-scrollbar">
                    {analysis.reasoning && analysis.reasoning.map((r, i) => (
                        <div key={i} className="flex gap-2"><div className="mt-0.5 w-3 h-3 bg-zinc-900 border border-zinc-700 rounded flex items-center justify-center flex-shrink-0">{i === 0 ? <span className="text-[6px] text-zinc-500">⚡</span> : (i === 1 ? <span className="text-[6px] text-zinc-500">📈</span> : <span className="text-[6px] text-zinc-500">📝</span>)}</div><p className="text-[10px] text-zinc-300 leading-relaxed font-mono">{typeof r === 'string' ? r.replace(/^\d+\.\s*/, '') : r}</p></div>
                    ))}
                </div>

                {analysis.prediction && (
                    <div className="mt-4 p-3 bg-purple-900/10 border border-purple-500/20 rounded-xl animate-in fade-in slide-in-from-bottom-2 duration-700">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs">🔮</span>
                            <span className="text-[9px] font-black text-purple-400 tracking-[0.2em] uppercase">FUTURE PREDICTION (10-25 CANDLES)</span>
                        </div>
                        <p className="text-[10px] text-zinc-300 leading-relaxed font-mono italic">
                            "{analysis.prediction}"
                        </p>
                    </div>
                )}

                <div className="mt-4 pt-2 border-t border-zinc-900/80">
                     <div className="text-[8px] font-black text-zinc-500 text-center tracking-[0.2em] mb-2 uppercase">AI ACCURACY FEEDBACK</div>
                     {!feedbackSubmitted ? (
                         <div className="grid grid-cols-2 gap-3 animate-in fade-in slide-in-from-bottom-2 duration-500">
                             <button onClick={() => handleFeedbackClick('WIN')} className="py-2.5 bg-green-900/10 border border-green-500/30 rounded-lg group hover:bg-green-500/10 transition-all flex flex-col items-center justify-center gap-0.5"><span className="text-[10px] font-black text-green-500 group-hover:scale-110 transition-transform">WIN (PROFIT)</span><span className="text-[7px] text-green-500/50">SIGNAL GACOR</span></button>
                             <button onClick={() => handleFeedbackClick('LOSS')} className="py-2.5 bg-red-900/10 border border-red-500/30 rounded-lg group hover:bg-red-500/10 transition-all flex flex-col items-center justify-center gap-0.5"><span className="text-[10px] font-black text-red-500 group-hover:scale-110 transition-transform">LOSS (FIX)</span><span className="text-[7px] text-red-500/50">PERBAIKI SIGNAL</span></button>
                         </div>
                     ) : (
                         <div className="py-2 bg-green-900/10 border border-green-500/20 rounded-lg text-center animate-in zoom-in-95 duration-300 flex items-center justify-center gap-2"><div className="text-sm animate-bounce">😆</div><span className="text-[9px] font-black text-green-400 tracking-wider uppercase">TERIMA KASIH ATAS MASUKAN ANDA</span></div>
                     )}
                </div>
            </div>
        )}
      </div>
    );
  };

  return (
    <div className="relative w-full h-full bg-[#050505] overflow-hidden flex flex-col font-sans">
      {/* NOTCH: Moved outside scroll view to prevent Z-Index/Layout issues */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-28 h-5 bg-[#111] rounded-b-xl border-b border-x border-[#222] z-40 pointer-events-none"></div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden no-scrollbar pb-40">
          <div className="px-5 pt-2 pb-2 flex justify-between items-start relative z-50 pointer-events-none"> {/* Wrapper pointer-events-none to let clicks pass through empty spaces */}
              <div className="flex flex-col gap-1 pointer-events-auto"> {/* Re-enable pointer events for content */}
                  <div className="flex items-center gap-3">
                      <h1 className="font-black tracking-[0.16em] text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-indigo-300 to-purple-400 animate-text-shimmer drop-shadow-[0_0_15px_rgba(168,85,247,0.6)]"><span className="text-base">NvR</span><span className="text-[13px] ml-1.5">TRADE INTELLIGENCE</span></h1>
                  </div>
                  <div className="relative inline-block mt-1"><button onClick={() => setShowAssetSelector(true)} className="flex items-center gap-2 bg-[#0f0f11] border border-purple-500/40 hover:border-purple-500 shadow-[0_0_20px_-5px_rgba(168,85,247,0.4)] hover:shadow-[0_0_25px_-5px_rgba(168,85,247,0.6)] px-3 py-1.5 rounded-lg transition-all duration-300 group"><span className="text-xs font-black text-white tracking-wider group-hover:text-purple-300 transition-colors">{currentAsset}</span><span className="w-px h-3 bg-zinc-700"></span><span className="text-[9px] text-purple-400 font-bold uppercase">{ASSET_DETAILS[currentAsset]?.name || "ASSET"}</span><span className="text-[10px] text-zinc-500 ml-1">▼</span></button></div>
              </div>
              <button onClick={() => {
                  if (!userSession.isLoggedIn) {
                      onRequestAuth();
                  } else {
                      setShowDashboard(true);
                  }
              }} className="relative z-[100] w-10 h-10 bg-[#111] border border-zinc-800 rounded-xl flex flex-col items-center justify-center gap-1.5 hover:border-zinc-600 transition-all group pointer-events-auto cursor-pointer">
                  <div className="w-5 h-0.5 bg-zinc-500 group-hover:bg-white transition-colors rounded-full"></div>
                  <div className="w-3 h-0.5 bg-zinc-500 group-hover:bg-white transition-colors rounded-full ml-2"></div>
                  <div className="w-5 h-0.5 bg-zinc-500 group-hover:bg-white transition-colors rounded-full"></div>
              </button>
          </div>
          <div className="px-5 mb-1 flex items-center gap-2 overflow-x-auto no-scrollbar">{['M1','M5','M15','M30','H1','H4','D1'].map((tf) => (<button key={tf} onClick={() => onTimeframeChange(tf as TimeFrame)} className={`min-w-[32px] h-[28px] flex items-center justify-center text-[10px] font-bold rounded-lg transition-all ${currentTimeframe === tf ? 'bg-[#003d2e] text-[#00ff9d] border border-[#00ff9d]/30 shadow-[0_0_10px_-4px_#00ff9d]' : 'text-zinc-600 hover:text-zinc-300 hover:bg-white/5'}`}>{tf}</button>))}</div>
          <div className={`relative w-full border-y border-zinc-900 bg-[#020202] touch-none transition-all duration-500 ease-in-out ${showAnalysisLog && analysis ? 'h-[280px]' : 'h-[460px]'}`}>{children}</div>
          <div className="px-4 pt-2 pb-10">{renderAnalysisCard()}</div>
      </div>
      
      {/* FOOTER BAR */}
      <div className="absolute bottom-0 inset-x-0 z-50 pt-4 pb-6 px-5 pointer-events-none">
          <div className="relative w-full pointer-events-auto"> 
              <div className="flex justify-between items-center mb-3 px-1 py-1 border-y border-dashed border-zinc-700/50 rounded-lg bg-transparent">
                  {(() => {
                      const msLeft = userSession.membershipExpiresAt ? userSession.membershipExpiresAt - Date.now() : 0;
                      const tier = userSession.membershipTier || 'NONE';
                      const isMember = tier !== 'NONE' && msLeft > 0;
                      
                      const badgeImg = MEMBERSHIP_IMAGES[tier as keyof typeof MEMBERSHIP_IMAGES] || "";
                      return isMember ? (
                          <div className="px-2 py-1 bg-gradient-to-r from-purple-900/20 to-blue-900/20 border border-purple-500/30 rounded-lg flex items-center gap-2 shadow-[0_0_15px_-5px_#a855f7] backdrop-blur-sm flex-shrink-0 animate-in fade-in">
                              <img src={badgeImg} alt="M" className="w-[28px] h-[17px] rounded-sm shadow-[0_0_8px_rgba(255,255,255,0.8)] border border-white/80 object-contain bg-white/10" />
                              <span className="text-[7px] font-black text-[#FFD700] drop-shadow-[0_0_3px_rgba(255,215,0,0.6)] tracking-wider">{userSession.role === 'USER' ? 'Waktu tersisa ' : ''}<span className="text-yellow-400 font-mono tracking-widest">{timeLeftDisplay}</span></span>
                          </div>
                      ) : (
                          <div onClick={() => { setShowStore(true); setStoreView('SELECTION'); }} className="px-2 py-1 bg-[#1a0b2e]/20 border border-purple-500/30 rounded-lg flex items-center gap-1.5 shadow-[0_0_8px_rgba(168,85,247,0.3)] backdrop-blur-sm flex-shrink-0 cursor-pointer hover:bg-purple-900/20 transition-all">
                              <img src="https://i.ibb.co.com/LXQTP3Pj/image-1.jpg" alt="M" className="w-[18px] h-[11px] rounded-sm object-contain bg-white/10" referrerPolicy="no-referrer" />
                              <span className="text-[7px] font-black text-purple-100 tracking-wider">Berlangganan Sekarang✨</span>
                          </div>
                      );
                  })()}
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                      <div className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-green-500 shadow-[0_0_5px_#22c55e]' : 'bg-red-500 shadow-[0_0_5px_#ef4444]'} animate-pulse`}></div>
                      <span className={`text-[7px] font-black tracking-wider ${isOnline ? 'text-green-500' : 'text-red-500'}`}>{isOnline ? 'DATA : Bloomberg Terminal' : 'OFFLINE MODE'}</span>
                  </div>
              </div>
              <button onClick={handleAnalyzeClick} disabled={isAnalyzing} className="w-full h-11 bg-black/40 border border-purple-500/30 rounded-2xl relative overflow-hidden group hover:bg-black/60 transition-all shadow-[0_4px_20px_-5px_rgba(88,28,135,0.2)] active:scale-[0.98]"><div className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-500/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div><div className="flex items-center justify-center gap-3 h-full relative z-10">{isAnalyzing ? <><div className="w-3.5 h-3.5 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div><span className="text-[10px] font-black text-purple-300 tracking-[0.2em] animate-pulse">Menganalisa Structure Market</span></> : <span className="text-[11px] font-black tracking-[0.2em] text-purple-600 drop-shadow-[0_0_8px_rgba(233,213,255,0.8)]">Menganalisa Chart✨</span>}</div></button>
          </div>
      </div>
      
      {/* MODALS AND DASHBOARD */}
      {showAssetSelector && (
          <div className="absolute inset-0 z-[100] bg-black/95 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-200">
              <div className="w-full max-w-sm bg-[#050505] border border-zinc-900 rounded-[35px] overflow-hidden flex flex-col max-h-[80vh] shadow-[0_0_50px_-20px_rgba(0,0,0,1)]">
                  <div className="pt-8 pb-4 px-6 bg-[#050505] flex flex-col items-center relative z-10"><h2 className="text-xl font-black tracking-widest text-white drop-shadow-[0_0_15px_rgba(168,85,247,0.6)] uppercase">MARKET SELECTION</h2><div className="w-16 h-0.5 bg-purple-600 rounded-full mt-3 shadow-[0_0_10px_#9333ea]"></div><button onClick={() => setShowAssetSelector(false)} className="absolute top-6 right-6 w-8 h-8 rounded-full bg-[#111] border border-zinc-800 flex items-center justify-center text-zinc-500 hover:text-white transition-colors">✕</button></div>
                  <div className="flex-1 overflow-y-auto px-6 pb-6 custom-scrollbar space-y-8">{ASSET_GROUPS.map((group) => (<div key={group.title}><div className="flex items-center gap-2 mb-4 pl-1"><div className="w-1.5 h-1.5 bg-blue-500 rotate-45 shadow-[0_0_5px_#3b82f6]"></div><div className="text-[10px] font-black text-zinc-500 tracking-[0.2em] uppercase">{group.title}</div></div><div className="grid grid-cols-2 gap-4">{group.assets.map((asset) => { const isActive = currentAsset === asset; const details = ASSET_DETAILS[asset]; return (<button key={asset} onClick={() => { onAssetChange(asset); setShowAssetSelector(false); }} className={`relative flex flex-col items-start p-4 rounded-xl border transition-all duration-200 group overflow-hidden ${isActive ? 'bg-[#00ff9d]/5 border-[#00ff9d] shadow-[0_0_15px_-5px_rgba(0,255,157,0.2)]' : 'bg-[#111] border-transparent hover:bg-[#161616]'}`}>{isActive && <div className="absolute top-3 right-3 w-1.5 h-1.5 rounded-full bg-[#00ff9d] shadow-[0_0_8px_#00ff9d] animate-pulse"></div>}<span className={`text-sm font-black tracking-wide mb-1 ${isActive ? 'text-[#00ff9d]' : 'text-white'}`}>{asset}</span><span className={`text-[10px] font-bold ${isActive ? 'text-[#00ff9d]/60' : 'text-zinc-600'}`}>{details?.name || asset}</span></button>);})}</div></div>))}</div>
              </div>
          </div>
      )}
      
      {showHistoryModal && (
          <div className="absolute inset-0 z-[100] bg-black/40 backdrop-blur-md flex flex-col animate-in slide-in-from-bottom-10 duration-300">
              <div className="flex items-center justify-between p-5 border-b border-white/5 bg-black/60 backdrop-blur-xl"><div className="flex items-center gap-2"><span className="text-xs font-black tracking-[0.2em] text-white uppercase">TRADING LOGS</span></div><button onClick={() => setShowHistoryModal(false)} className="w-8 h-8 rounded-full bg-[#111] flex items-center justify-center text-zinc-500 hover:text-white transition-colors">✕</button></div>
              {(() => {
                  const now = Date.now();
                  const filteredHistory = executionHistory.filter(item => (now - item.timestamp) <= historyFilter.ms);
                  const wins = filteredHistory.filter(h => h.status === 'WIN').length;
                  const loss = filteredHistory.filter(h => h.status === 'LOSS').length;
                  const total = wins + loss;
                  const winrate = total > 0 ? Math.round((wins / total) * 100) : 0;
                  return (<><div className="p-3 grid grid-cols-2 gap-2 bg-transparent"><div className="py-1.5 px-3 bg-green-900/10 border border-green-500/20 rounded-xl flex flex-col items-center"><span className="text-[8px] text-green-500 font-bold tracking-wider uppercase">WINRATE</span><span className="text-lg font-black text-white">{winrate}%</span></div><div className="py-1.5 px-3 bg-purple-900/10 border border-purple-500/20 rounded-xl flex flex-col items-center"><span className="text-[8px] text-purple-500 font-bold tracking-wider uppercase">ENTRY</span><span className="text-lg font-black text-white">{total}</span></div></div><div className="px-4 pb-2 border-b border-white/5 overflow-x-auto no-scrollbar flex gap-2">{HISTORY_FILTERS.map((f) => (<button key={f.label} onClick={() => setHistoryFilter(f)} className={`px-3 py-1.5 rounded-lg text-[10px] font-bold tracking-wider transition-all whitespace-nowrap flex-shrink-0 ${historyFilter.label === f.label ? 'bg-purple-900/40 text-purple-400 border border-purple-500/50 shadow-[0_0_10px_-4px_#a855f7]' : 'bg-black/40 backdrop-blur-sm text-zinc-500 border border-zinc-800 hover:text-zinc-300'}`}>{f.label}</button>))}</div><div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">{filteredHistory.length === 0 ? (<div className="flex flex-col items-center justify-center h-full text-zinc-600 gap-2"><svg className="w-10 h-10 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg><span className="text-xs font-mono">NO DATA IN {historyFilter.label}</span></div>) : (filteredHistory.slice().reverse().map((item) => (<div key={item.id} className={`p-3 ${item.status === 'WIN' ? 'bg-green-950/30' : 'bg-red-950/30'} border ${item.status === 'WIN' ? 'border-green-500/50 shadow-[0_0_10px_-5px_#22c55e]' : 'border-red-500/50 shadow-[0_0_10px_-5px_#ef4444]'} rounded-xl flex items-center justify-between backdrop-blur-sm`}><div><div className="flex items-center gap-2"><span className="text-xs font-bold text-white">{item.asset}</span><span className={`text-[9px] px-1.5 rounded ${item.signal === 'BUY' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>{item.signal}</span></div><div className="text-[9px] text-zinc-400 mt-1 font-mono">{new Date(item.timestamp).toLocaleString()}</div></div><div className={`text-right ${item.status === 'WIN' ? 'text-green-500' : 'text-red-500'}`}><div className="text-sm font-black tracking-wider">{item.status}</div><div className="text-[8px] text-zinc-500 font-mono">RR {item.rr}</div></div></div>)))}</div></>);
              })()}
          </div>
      )}

      {showHamburgerMenu && (
        <div className="absolute inset-0 z-[200] bg-black/80 backdrop-blur-md flex flex-col p-6 animate-in slide-in-from-right duration-300">
            <div className="flex justify-between items-center mb-10">
                <h2 className="text-xl font-black text-white tracking-widest uppercase">MEMBERSHIP</h2>
                <button onClick={() => { setShowHamburgerMenu(false); setShowDashboard(true); }} className="w-10 h-10 rounded-xl bg-[#111] border border-zinc-800 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all">✕</button>
            </div>
            <div className="space-y-4">
                {/* MEMBERSHIP CODE INPUT */}
                <div className="p-5 bg-[#0a0a0a] border border-zinc-800/50 rounded-3xl">
                    <div className="text-[9px] font-bold text-zinc-600 tracking-widest uppercase mb-3 flex items-center gap-2">
                        <span className="text-xs">💳</span> MASUKAN KODE MEMBERSHIP
                    </div>
                    <div className="space-y-3">
                        <div className="relative">
                            <input 
                                type="text" 
                                value={membershipCode}
                                onChange={(e) => setMembershipCode(e.target.value)}
                                placeholder="Paste Kode Enkripsi Di Sini..."
                                className="w-full bg-black border border-zinc-800 rounded-xl px-4 py-3 text-[10px] text-white placeholder:text-zinc-700 focus:border-purple-500/50 focus:outline-none transition-all font-mono"
                            />
                            {isActivatingCode && (
                                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                                    <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                                </div>
                            )}
                        </div>
                        
                        {codeMessage && (
                            <div className={`p-2 rounded-lg text-[8px] font-bold text-center animate-in fade-in slide-in-from-top-1 duration-300 whitespace-pre-wrap ${codeMessage.type === 'SUCCESS' ? 'bg-green-500/10 text-green-500 border border-green-500/20' : 'bg-red-500/10 text-red-500 border border-red-500/20'}`}>
                                {codeMessage.text}
                            </div>
                        )}

                        <button 
                            onClick={handleActivateMembershipCode}
                            disabled={isActivatingCode || !membershipCode.trim()}
                            className={`w-full py-3 rounded-xl font-black text-[9px] tracking-[0.2em] uppercase transition-all ${isActivatingCode || !membershipCode.trim() ? 'bg-zinc-900 text-zinc-600 border border-zinc-800 cursor-not-allowed' : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-[0_0_20px_-5px_rgba(168,85,247,0.4)] hover:scale-[1.02] active:scale-[0.98]'}`}
                        >
                            {isActivatingCode ? 'VALIDATING...' : 'AKTIFKAN SEKARANG'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
      )}

      {showFundamentalData && (
          <FundamentalData onClose={() => setShowFundamentalData(false)} />
      )}

      {showDashboard && (
        <div className="absolute inset-0 z-[100] bg-black/80 backdrop-blur-xl flex items-center justify-center p-4 animate-in fade-in duration-300">
             <div className="w-full max-w-sm bg-[#050505] border border-zinc-900 rounded-[35px] shadow-2xl overflow-hidden flex flex-col gap-4 max-h-[90vh] relative">
                 {/* ADMIN HEADER */}
                 <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between bg-black/40 backdrop-blur-md z-20">
                     <div>
                        <span className="text-[9px] font-black tracking-[0.2em] text-zinc-500 uppercase">
                            USER DASHBOARD
                        </span>
                        <div className="flex items-center gap-2 mt-0.5">
                            <span className="text-sm font-black text-white">{userSession.username}</span>
                        </div>
                     </div>
                     <div className="flex items-center gap-3">
                         {/* HEADER LOGOUT BUTTON (Icon) */}
                         <button onClick={() => { onLogout(); setShowDashboard(false); }} className="w-8 h-8 rounded-full bg-red-900/10 border border-red-500/20 flex items-center justify-center text-red-500 hover:bg-red-500 hover:text-white transition-all group" title="Logout">
                            <svg className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                            </svg>
                         </button>
                         <button onClick={() => setShowDashboard(false)} className="w-8 h-8 rounded-full bg-[#111] flex items-center justify-center text-zinc-400 hover:text-white transition-colors">✕</button>
                     </div>
                 </div>

                 {/* DASHBOARD CONTENT */}
                 <div className="flex-1 overflow-y-auto no-scrollbar bg-gradient-to-b from-[#050505] to-[#020202]">
                     
                      {/* --- USER DASHBOARD ONLY (DEV & GUEST REMOVED) --- */}
                      <div className="p-6 space-y-6">
                         <div className="p-5 bg-[#0a0a0a] border border-zinc-800/50 rounded-3xl relative overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-purple-900/10 blur-[50px] rounded-full"></div>
                            <div className="relative z-10">
                                <div className="text-[9px] font-bold text-zinc-600 tracking-widest uppercase mb-1">IDENTITY</div>
                                <div className="text-xl font-bold text-white mb-3">{userSession.username}</div>
                                <div className="flex items-center gap-2 mb-4">
                                    <span className="px-2 py-1 bg-purple-900/20 border-purple-900/30 text-purple-500 rounded-md text-[9px] font-bold tracking-wider">{userSession.role}</span>
                                    <span className="px-2 py-1 bg-zinc-800/50 border border-zinc-700/30 rounded-md text-[9px] font-bold text-zinc-400 tracking-wider">BALANCE {userSession.tokens}</span>
                                </div>
                                
                                {/* MEMBERSHIP STATUS */}
                                <div 
                                    onClick={() => { setShowDashboard(false); setShowHamburgerMenu(true); }}
                                    className="mb-6 p-3 bg-zinc-900/50 border border-zinc-800 rounded-2xl cursor-pointer hover:border-purple-500/50 transition-all group"
                                 >
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-[8px] font-black text-zinc-500 tracking-widest uppercase">MEMBERSHIP STATUS</span>
                                        <span className={`text-[8px] font-black px-2 py-0.5 rounded-full ${userSession.membershipTier !== 'NONE' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}`}>
                                            {userSession.membershipTier !== 'NONE' ? userSession.membershipTier : 'INACTIVE'}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-400">
                                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                        </div>
                                        <div>
                                            <div className="text-[10px] font-bold text-white flex items-center gap-1.5">
                                                {(userSession.membershipTier === 'NONE' || (userSession.membershipExpiresAt && userSession.membershipExpiresAt <= Date.now())) && (
                                                    <img src="https://i.ibb.co.com/LXQTP3Pj/image-1.jpg" alt="M" className="w-[16px] h-[10px] rounded-sm object-contain bg-white/10" referrerPolicy="no-referrer" />
                                                )}
                                                {userSession.membershipTier === 'NONE' || (userSession.membershipExpiresAt && userSession.membershipExpiresAt <= Date.now()) 
                                                    ? "Berlangganan Sekarang✨" 
                                                    : timeLeftDisplay}
                                            </div>
                                            <div className="text-[7px] text-zinc-500 uppercase tracking-tighter">SISA WAKTU BERLANGGANAN</div>
                                            {userSession.membershipTier !== 'NONE' && (
                                                <div className="text-[6px] text-green-500/70 uppercase tracking-widest mt-0.5">
                                                    Aktif di latar belakang (Akun Google & Perangkat)
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <button onClick={() => { onLogout(); setShowDashboard(false); }} className="w-full py-3 bg-[#111] border border-zinc-800 rounded-xl text-[10px] font-bold text-zinc-400 tracking-[0.2em] hover:bg-zinc-800 hover:text-white transition-all">LOGOUT</button>
                            </div>
                         </div>

                         {/* PWA INSTALL BUTTON */}
                        {installPrompt && onInstallApp && (
                            <button onClick={onInstallApp} className="w-full p-4 bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-500/50 rounded-2xl flex items-center justify-between group hover:border-blue-400 transition-all shadow-[0_0_15px_-5px_rgba(59,130,246,0.3)] animate-pulse-slow">
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 bg-blue-500 text-white rounded-xl flex items-center justify-center font-black text-lg">↓</div>
                                    <div className="text-left">
                                        <div className="text-xs font-black text-blue-400 uppercase tracking-wide">INSTALL APPLICATION</div>
                                        <div className="text-[9px] text-zinc-400">Download ke HP (Tanpa ke Playstore)</div>
                                    </div>
                                </div>
                                <span className="text-zinc-600 group-hover:text-blue-400 transition-colors">→</span>
                            </button>
                        )}
                        <div className="space-y-3">
                            <button onClick={() => { setShowFundamentalData(true); setShowDashboard(false); }} className="w-full p-4 bg-[#0a0a0a] border border-zinc-800 rounded-2xl flex items-center justify-between group hover:border-amber-500/30 transition-all"><div className="flex items-center gap-4"><div className="w-10 h-10 bg-amber-900/10 rounded-xl flex items-center justify-center text-amber-500"><span className="text-xl">🌎</span></div><div className="text-left"><div className="text-xs font-bold text-white">Kalender Ekonomi Dunia</div><div className="text-[9px] text-zinc-500">Data Fundamental & Bloomberg</div></div></div><span className="text-zinc-600 group-hover:text-amber-500 transition-colors">→</span></button>
                            <button onClick={() => { setShowHistoryModal(true); setShowDashboard(false); }} className="w-full p-4 bg-[#0a0a0a] border border-zinc-800 rounded-2xl flex items-center justify-between group hover:border-purple-500/30 transition-all"><div className="flex items-center gap-4"><div className="w-10 h-10 bg-purple-900/10 rounded-xl flex items-center justify-center text-purple-500"><svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg></div><div className="text-left"><div className="text-xs font-bold text-white">History Trading</div><div className="text-[9px] text-zinc-500">Analisa & Winrate Record</div></div></div><span className="text-zinc-600 group-hover:text-purple-500 transition-colors">→</span></button>
                            {/* REMOVED TOP UP TOKEN BUTTON */}
                            <button onClick={() => { setShowStore(true); setStoreView('SELECTION'); setShowDashboard(false); }} className="w-full p-4 bg-[#0a0a0a] border border-zinc-800 rounded-2xl flex items-center justify-between group hover:border-cyan-500/30 transition-all"><div className="flex items-center gap-4"><div className="w-10 h-10 bg-cyan-900/10 rounded-xl flex items-center justify-center text-cyan-500"><svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg></div><div className="text-left"><div className="text-xs font-bold text-cyan-400">Upgrade Membership</div><div className="text-[9px] text-zinc-500">Unlimited Access & Priority</div></div></div><span className="text-zinc-600 group-hover:text-cyan-500 transition-colors">→</span></button>
                            <button onClick={handleContactSupport} className="w-full p-4 bg-[#0a0a0a] border border-zinc-800 rounded-2xl flex items-center justify-between group hover:border-red-500/30 transition-all"><div className="flex items-center gap-4"><div className="w-10 h-10 bg-red-900/10 rounded-xl flex items-center justify-center text-red-500"><svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z" /></svg></div><div className="text-left"><div className="text-xs font-bold text-red-500">Pusat Bantuan 📞</div><div className="text-[9px] text-zinc-500">Hubungi Admin jika ada kendala</div></div></div><span className="text-zinc-600 group-hover:text-red-500 transition-colors">→</span></button>
                        </div>
                     </div>
                 </div>
             </div>
        </div>
      )}
      
      {/* STORE */}
      {showStore && (
          <div className="absolute inset-0 z-[100] bg-black/40 backdrop-blur-md flex flex-col animate-in slide-in-from-right duration-300">
             {/* STORE HEADER */}
             <div className="px-5 py-6 bg-black/60 backdrop-blur-xl border-b border-white/5 flex items-center justify-between z-20 shadow-md">
                 <div>
                     <h2 className="text-xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-600 via-white to-purple-600 logo-shimmer tracking-widest drop-shadow-[0_0_20px_rgba(168,85,247,0.5)] uppercase">NvR Store</h2>
                     <p className="text-[6px] text-purple-200/70 tracking-[0.2em] uppercase mt-0.5">SECURE CRYPTO PAYMENT GATEWAY</p>
                 </div>
                 <button onClick={() => setShowStore(false)} className="w-10 h-10 rounded-xl bg-[#111] border border-zinc-800 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all">✕</button>
              </div>
              
              {/* REMOVED TAB SELECTION DIV */}

               <div className="flex-1 overflow-y-auto custom-scrollbar bg-transparent relative">
                   <div className="relative z-10 w-full min-h-full flex flex-col">
                     {storeView === 'SELECTION' ? (
                         // ONLY MEMBERSHIP LIST REMAINS
                         <div className="flex flex-col gap-6 px-6 py-4 pb-32">
                            <div className="text-center mb-2"><h2 className="text-lg font-black text-purple-400 tracking-[0.2em] uppercase drop-shadow-[0_0_10px_rgba(168,85,247,0.5)]">PAKET BERLANGGANAN</h2><p className="text-[8px] text-zinc-500 tracking-[0.3em] mt-1 uppercase">PILIH AKSES PREMIUM ANDA</p></div>
                            {MEMBERSHIP_PACKAGES.map((pkg) => {
                                let borderColor = 'border-zinc-800';
                                let shadowColor = '';
                                let titleColor = 'text-white';
                                let btnGradient = 'bg-zinc-800 text-zinc-400';
                                let checkColor = 'text-zinc-600';
                                let iconBg = 'bg-zinc-900';
                                let saveBadgeColor = 'bg-zinc-800 text-zinc-400';
                                let cardBg = 'bg-[#0a0a0a]';
                                if (pkg.theme === 'GOLD') { borderColor = 'border-amber-500/80'; shadowColor = 'shadow-[0_0_30px_-10px_rgba(245,158,11,0.5)]'; titleColor = 'text-amber-400'; btnGradient = 'bg-gradient-to-r from-amber-600 to-orange-600 text-white hover:from-amber-500 hover:to-orange-500'; checkColor = 'text-amber-500 border-amber-500'; iconBg = 'bg-amber-900/40 text-amber-300'; saveBadgeColor = 'bg-amber-500/20 text-amber-400 border border-amber-500/30'; cardBg = 'bg-gradient-to-br from-amber-950/20 to-black'; } 
                                else if (pkg.theme === 'PURPLE') { borderColor = 'border-purple-500/50'; shadowColor = 'shadow-[0_0_25px_-10px_rgba(168,85,247,0.3)]'; titleColor = 'text-purple-400'; btnGradient = 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:from-purple-500 hover:to-indigo-500'; checkColor = 'text-purple-500 border-purple-500'; iconBg = 'bg-purple-900/30 text-purple-300'; saveBadgeColor = 'bg-purple-500/20 text-purple-400 border border-purple-500/30'; cardBg = 'bg-gradient-to-br from-purple-950/20 to-black'; } 
                                else if (pkg.theme === 'BLUE') { borderColor = 'border-cyan-500/50'; shadowColor = 'shadow-[0_0_20px_-10px_rgba(6,182,212,0.3)]'; titleColor = 'text-cyan-400'; btnGradient = 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white hover:from-cyan-500 hover:to-blue-500'; checkColor = 'text-amber-500 border-amber-500'; iconBg = 'bg-cyan-900/30 text-cyan-300'; saveBadgeColor = 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'; cardBg = 'bg-gradient-to-br from-cyan-950/20 to-black'; }
                                return (
                                    <button key={pkg.id} onClick={() => handleMembershipSelect(pkg)} className={`relative w-full ${cardBg} rounded-3xl border-[1.5px] ${borderColor} ${shadowColor} overflow-hidden group transition-all duration-300 transform hover:scale-[1.02] flex flex-col p-6 text-left`}>
                                        {pkg.isBestValue && <div className="absolute top-4 right-4 px-3 py-1 bg-white shadow-[0_0_15px_rgba(255,255,255,0.4)] rounded-full z-10 animate-pulse-slow"><span className="text-[7px] font-black text-black tracking-widest uppercase">BEST VALUE</span></div>}
                                        <div className="flex items-center gap-4 mb-5 relative z-10"><div className={`w-12 h-12 rounded-2xl ${iconBg} flex items-center justify-center border border-white/5 shadow-inner`}><span className="text-xl font-bold">∞</span></div><div><h3 className={`text-base font-black tracking-widest uppercase ${titleColor} drop-shadow-sm`}>{pkg.tier}</h3><p className="text-[9px] text-zinc-400 font-bold tracking-wide uppercase mt-0.5">{pkg.subtitle}</p></div></div>
                                        <div className="mb-5 relative z-10"><div className="flex items-center gap-2 mb-1"><span className="text-[11px] text-zinc-500 line-through decoration-red-500/80 decoration-2 font-bold">Rp {pkg.originalPrice.toLocaleString('id-ID')}</span><span className={`text-[7px] font-black px-2 py-0.5 rounded-md uppercase tracking-wider ${saveBadgeColor}`}>{pkg.saveLabel}</span></div><div className="flex items-baseline gap-1.5"><span className="text-3xl font-black text-white tracking-tighter drop-shadow-md">Rp {pkg.price.toLocaleString('id-ID').replace(',','.')}</span><span className="text-[10px] text-zinc-500 font-bold tracking-wide">{pkg.durationLabel}</span></div></div>
                                        <div className="space-y-3 mb-6 border-t border-white/10 pt-5 w-full relative z-10">{pkg.features.map((feat, idx) => (<div key={idx} className="flex items-start gap-3"><div className={`mt-0.5 w-3.5 h-3.5 rounded-full flex items-center justify-center border-[1.5px] ${checkColor} shadow-[0_0_5px_rgba(0,0,0,0.5)] bg-black/40`}><svg className="w-2 h-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={4} d="M5 13l4 4L19 7" /></svg></div><span className="text-[10px] font-bold text-zinc-300 leading-tight">{feat}</span></div>))}</div>
                                        <div className={`w-full py-3.5 rounded-xl ${btnGradient} flex items-center justify-center gap-2 shadow-lg group-hover:shadow-[0_0_20px_rgba(255,255,255,0.15)] transition-all relative z-10 border border-white/10`}><span className="text-[10px] font-black tracking-[0.25em] uppercase">BELI SEKARANG</span><span className="text-[10px] transform group-hover:translate-x-1 transition-transform">→</span></div>
                                    </button>
                                );
                            })}
                            <p className="text-[7px] text-zinc-600 text-center px-4 leading-relaxed mt-4">Secure Payment via QRIS / Bank Transfer.<br/>Processed manually by Admin via WhatsApp.</p>
                        </div>
                     ) : (
                         // PAYMENT VIEW (Simplified to Membership Only)
                         <div className="flex-1 flex flex-col items-center justify-center px-6 pt-10 pb-20 w-full animate-in fade-in zoom-in-95 duration-300">
                            <div className="w-full max-w-sm flex flex-col items-center">
                                <div className="relative rounded-[24px] p-[3px] mb-6 w-[210px] h-[210px] flex items-center justify-center shrink-0 group overflow-hidden shadow-[0_0_50px_-10px_rgba(168,85,247,0.6)]">
                                    {/* Animated Gradient Border */}
                                    <div className="absolute inset-[-50%] bg-[conic-gradient(from_0deg,transparent_0_340deg,rgba(168,85,247,1)_360deg)] animate-[spin_3s_linear_infinite]"></div>
                                    <div className="absolute inset-[-50%] bg-[conic-gradient(from_180deg,transparent_0_340deg,rgba(34,211,238,1)_360deg)] animate-[spin_3s_linear_infinite]"></div>
                                    
                                    {/* Inner Container */}
                                    <div className="w-full h-full relative bg-white rounded-[21px] overflow-hidden z-10 p-2">
                                        <div className="w-full h-full relative rounded-[14px] overflow-hidden">
                                            <img src={QRIS_IMAGE_URL} alt="QRIS" className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700 ease-out" />
                                            {/* Enhanced Scanline */}
                                            <div className="absolute left-0 right-0 h-[3px] bg-cyan-400 shadow-[0_0_15px_#22d3ee,0_0_30px_#22d3ee] z-20 opacity-90" style={{ animation: 'scan-line 2s ease-in-out infinite' }}></div>
                                            {/* Overlay Glow */}
                                            <div className="absolute inset-0 bg-gradient-to-tr from-purple-500/20 to-cyan-500/20 mix-blend-overlay pointer-events-none"></div>
                                        </div>
                                    </div>
                                </div>
                                <div className="text-[7px] font-black text-purple-200/60 tracking-[0.3em] uppercase mb-1">TOTAL PEMBAYARAN</div>
                                <div className="flex items-center justify-center gap-3 mb-4">
                                     <span className="text-2xl font-black text-white tracking-tight drop-shadow-lg font-sans">Rp {selectedProduct?.data.price.toLocaleString('id-ID').replace(',','.')}</span>
                                     <button onClick={() => { navigator.clipboard.writeText(selectedProduct?.data.price.toString() || ""); alert("Nominal Disalin!"); }} className="w-8 h-8 bg-white/10 border border-white/10 rounded-lg flex items-center justify-center text-purple-300 hover:bg-purple-600 hover:text-white transition-all active:scale-95 shadow-lg"><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg></button>
                                </div>
                                <div className="mb-10 text-center">
                                    <div className="px-5 py-2 rounded-2xl bg-[#111] border border-zinc-800 shadow-[0_0_15px_-5px_rgba(168,85,247,0.3)] flex flex-col items-center gap-1">
                                        <span className="text-[7px] font-bold text-zinc-500 uppercase tracking-widest">PRODUK YANG DIPILIH:</span>
                                        <span className="text-[10px] font-black text-white uppercase tracking-wider">{selectedProduct?.data.tier}</span>
                                        <div className="mt-1.5 pt-1.5 border-t border-zinc-800 w-full">
                                            <span className="text-[8px] font-black text-cyan-400 uppercase tracking-widest">AKSES: Full Akses Bebas Analisa Unlimited</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="w-[85%] space-y-3">
                                    <button onClick={handleConfirmWhatsapp} className="w-full py-3 rounded-xl bg-gradient-to-r from-[#7c3aed] to-[#4f46e5] hover:from-[#6d28d9] hover:to-[#4338ca] text-white font-black text-[9px] tracking-[0.2em] uppercase shadow-[0_0_20px_-5px_rgba(124,58,237,0.5)] transition-all transform hover:scale-[1.02] active:scale-[0.98] border border-white/10">KONFIRMASI WHATSAPP</button>
                                    <button onClick={handleBackToStore} className="w-full py-3 rounded-xl bg-[#111] border border-zinc-800 text-zinc-500 font-bold text-[9px] tracking-[0.2em] uppercase hover:bg-zinc-800 hover:text-white hover:border-zinc-700 transition-all">BATALKAN</button>
                                </div>
                            </div>
                        </div>
                     )}
                   </div>
               </div>
          </div>
      )}
    </div>
  );
};

export default FloatingUI;
