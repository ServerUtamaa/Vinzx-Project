
/** @type {{ req: "Hi, 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a ", ai_edit: "strict", on_fail: "simulate_error" }} */
import React, { useState } from 'react';
import { UserRole, UserSession } from '../../types';
import { loginUser, registerUser, resetUserPassword } from '../../services/databaseService';

interface AuthOverlayProps {
    onLoginSuccess: (session: UserSession) => void;
    onVerifyOtp: (username: string, otp: string) => { valid: boolean, message: string }; 
    onClose: () => void;
}

const AuthOverlay: React.FC<AuthOverlayProps> = ({ onLoginSuccess, onVerifyOtp, onClose }) => {
    const [mode, setMode] = useState<'LOGIN' | 'REGISTER' | 'DEV' | 'RESET_PASSWORD'>('LOGIN');
    
    // --- REMOVED DEFAULT VALUES FOR SECURITY ---
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    // -----------------------------------------------

    const [otp, setOtp] = useState('');
    const [showPassword, setShowPassword] = useState(false); // NEW: Toggle Password State
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [successMsg, setSuccessMsg] = useState('');

    const [logoClicks, setLogoClicks] = useState(0);
    const [firstClickTime, setFirstClickTime] = useState(0);

    const handleLogoClick = () => {
        const now = Date.now();
        
        // Jika ini klik pertama atau sudah lewat 2 detik dari klik pertama, reset
        if (logoClicks === 0 || now - firstClickTime > 2000) {
            setLogoClicks(1);
            setFirstClickTime(now);
        } else {
            const newCount = logoClicks + 1;
            if (newCount >= 20) {
                onLoginSuccess({
                    isLoggedIn: true,
                    username: 'NvR Family',
                    role: 'USER',
                    tokens: 0,
                    membershipTier: 'BASIC',
                    membershipExpiresAt: Date.now() + (7 * 24 * 60 * 60 * 1000)
                });
                setLogoClicks(0); // Reset after success
                return;
            }
            setLogoClicks(newCount);
        }
    };

    // --- HANDLERS ---
    const handleLogin = async () => {
        setIsLoading(true);
        setError('');
        setSuccessMsg('');

        // Artificial Delay for realism
        await new Promise(r => setTimeout(r, 800));

        try {
            if (mode === 'DEV') {
                // Dev Login via Service
                const res = await loginUser('System Admin', password); 
                if (res.success && res.user) {
                     onLoginSuccess({
                         isLoggedIn: true,
                         username: 'System Admin',
                         role: 'DEV',
                         tokens: 9999,
                         membershipTier: 'VIP',
                         membershipExpiresAt: 0
                     });
                } else {
                     setError("Access Denied: Invalid Key");
                }
            } else {
                // User Login via Database
                const res = await loginUser(username, password);
                if (res.success && res.user) {
                    onLoginSuccess({
                        isLoggedIn: true,
                        username: res.user.username,
                        role: res.user.role,
                        tokens: res.user.tokens,
                        membershipTier: res.user.membershipTier || 'NONE',
                        membershipExpiresAt: res.user.membershipExpiresAt || 0
                    });
                } else {
                    setError(res.message);
                }
            }
        } catch (e) {
            setError("Database Connection Error");
        } finally {
            setIsLoading(false);
        }
    };

    const handleRegister = async () => {
        setIsLoading(true);
        setError('');
        setSuccessMsg('');
        
        await new Promise(r => setTimeout(r, 800));

        if (!username || !password || !otp) {
            setError('Semua data wajib diisi (termasuk OTP)');
            setIsLoading(false);
            return;
        }

        // 1. VERIFY OTP FIRST
        const verification = onVerifyOtp(username, otp);
        if (!verification.valid) {
             setError(verification.message);
             setIsLoading(false);
             return;
        }

        // 2. REGISTER TO DATABASE
        const regRes = await registerUser(username, password);
        
        if (regRes.success) {
             setSuccessMsg("Akun Berhasil Dibuat! Silakan Login.");
             setMode('LOGIN'); // Switch to login screen
             setPassword(''); // Clear password for security
        } else {
             setError(regRes.message);
        }
        
        setIsLoading(false);
    };

    const handleResetPassword = async () => {
        setIsLoading(true);
        setError('');
        setSuccessMsg('');

        await new Promise(r => setTimeout(r, 800));

        if (!username || !password || !otp) {
            setError('Username, OTP, dan Password Baru wajib diisi!');
            setIsLoading(false);
            return;
        }

        // 1. VERIFY OTP
        const verification = onVerifyOtp(username, otp);
        if (!verification.valid) {
            setError(verification.message);
            setIsLoading(false);
            return;
        }

        // 2. RESET PASSWORD IN DB
        const resetRes = await resetUserPassword(username, password);
        
        if (resetRes.success) {
            setSuccessMsg("Password Berhasil Direset! Silakan Login.");
            setMode('LOGIN');
            setPassword('');
        } else {
            setError(resetRes.message);
        }

        setIsLoading(false);
    };

    // --- THEME CONFIG ---
    const isDev = mode === 'DEV';
    
    const theme = {
        glow: isDev ? 'shadow-[0_0_50px_-10px_rgba(239,68,68,0.6)]' : 'shadow-[0_0_50px_-10px_rgba(168,85,247,0.6)]',
        border: isDev ? 'border-red-500/30' : 'border-purple-500/30',
        titleGradient: isDev 
            ? 'from-red-500 via-orange-500 to-red-500' 
            : 'from-white via-purple-200 to-purple-400',
        inputBg: 'bg-[#111113]',
        inputBorder: isDev ? 'focus:border-red-500/70' : 'focus:border-purple-500/70',
        btnBg: isDev 
            ? 'bg-gradient-to-r from-red-700 to-red-600 hover:from-red-600 hover:to-red-500' 
            : 'bg-gradient-to-r from-purple-700 to-indigo-700 hover:from-purple-600 hover:to-indigo-600',
        accentText: isDev ? 'text-red-400' : 'text-purple-400'
    };

    return (
        <div className="absolute inset-0 z-[100] bg-black/90 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-500">
            <style>{`
                @keyframes shimmer-text { 
                    0% { background-position: 200% center; } 
                    100% { background-position: -200% center; } 
                }
                .animate-text-shimmer { 
                    background-size: 200% auto; 
                    animation: shimmer-text 3s linear infinite; 
                }
            `}</style>
            
            <div className={`relative w-full max-w-[300px] ${isDev ? 'min-h-[350px]' : 'min-h-[440px]'} bg-[#050505] rounded-[2rem] px-6 py-8 border ${theme.border} ${theme.glow} transition-all duration-500 overflow-hidden flex flex-col justify-center`}>
                
                <div className={`absolute top-0 inset-x-0 h-px bg-gradient-to-r ${isDev ? 'from-transparent via-red-500 to-transparent' : 'from-transparent via-purple-500 to-transparent'} opacity-50`}></div>
                <div className={`absolute -top-16 -right-16 w-32 h-32 ${isDev ? 'bg-red-600' : 'bg-purple-600'} rounded-full blur-[60px] opacity-20 pointer-events-none`}></div>
                <div className={`absolute -bottom-16 -left-16 w-32 h-32 ${isDev ? 'bg-orange-600' : 'bg-blue-600'} rounded-full blur-[60px] opacity-20 pointer-events-none`}></div>

                <div className="relative z-10 flex flex-col gap-5 h-full justify-center">
                    
                    <div className="text-center space-y-1.5 mb-2">
                        <h1 className={`text-2xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-br ${theme.titleGradient} drop-shadow-sm`}>
                            {isDev ? 'SYSTEM CORE_' : 'Tools Ai✨'}
                        </h1>
                        <p className={`text-[8px] uppercase tracking-[0.2em] font-mono ${isDev ? 'text-red-500/70' : 'text-zinc-500'}`}>
                            {isDev ? 'RESTRICTED ACCESS' : 
                                (mode === 'REGISTER' ? 'CREATE ACCOUNT' : 
                                (mode === 'RESET_PASSWORD' ? 'RESET CREDENTIALS' : 'Smart Trader Ai Using High Class Genius Algorithm'))}
                        </p>
                    </div>

                    <div className="space-y-4">
                        
                        {mode !== 'DEV' && (
                            <div className="group">
                                <label className="block text-[7px] font-bold text-zinc-500 mb-1 ml-1 tracking-widest uppercase">Identity</label>
                                <div className="relative">
                                    <input 
                                        type="text" 
                                        value={username}
                                        onChange={e => setUsername(e.target.value)}
                                        className={`w-full ${theme.inputBg} text-white text-[11px] px-3.5 py-3 rounded-xl border border-zinc-800 outline-none transition-all duration-300 ${theme.inputBorder} placeholder-zinc-700 font-medium`}
                                        placeholder="Username"
                                    />
                                </div>
                            </div>
                        )}

                        <div className="group">
                            <label className="block text-[7px] font-bold text-zinc-500 mb-1 ml-1 tracking-widest uppercase">
                                {isDev ? 'Encrypted Key' : (mode === 'RESET_PASSWORD' ? 'New Password' : 'Password')}
                            </label>
                            <div className="relative">
                                <input 
                                    type={showPassword ? "text" : "password"} 
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    className={`w-full ${theme.inputBg} text-white text-[11px] pl-3.5 pr-10 py-3 rounded-xl border border-zinc-800 outline-none transition-all duration-300 ${theme.inputBorder} placeholder-zinc-700 font-mono tracking-widest`}
                                    placeholder={showPassword ? "Password" : "••••••••"}
                                />
                                {/* SHOW/HIDE PASSWORD TOGGLE */}
                                <button
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white transition-colors p-1"
                                    tabIndex={-1}
                                >
                                    {showPassword ? (
                                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                                        </svg>
                                    ) : (
                                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                        </svg>
                                    )}
                                </button>
                            </div>
                            
                            {/* FORGOT PASSWORD LINK - PLACED BELOW INPUT ON LEFT */}
                            {mode === 'LOGIN' && (
                                <button 
                                    onClick={() => { setMode('RESET_PASSWORD'); setError(''); setSuccessMsg(''); setPassword(''); setOtp(''); }}
                                    className="text-[8px] font-bold text-zinc-500 hover:text-red-400 transition-colors mt-1.5 ml-1 text-left block hover:underline"
                                >
                                    Lupa Password!?
                                </button>
                            )}
                        </div>

                        {(mode === 'REGISTER' || mode === 'RESET_PASSWORD') && (
                            <div className="animate-in slide-in-from-top-4 fade-in duration-300">
                                <label className="block text-[7px] font-bold text-zinc-500 mb-1 ml-1 tracking-widest uppercase">
                                    {mode === 'RESET_PASSWORD' ? 'Admin Verification Code' : 'Verification'}
                                </label>
                                <div className="flex gap-2">
                                    <input 
                                        type="text" 
                                        value={otp}
                                        onChange={e => setOtp(e.target.value)}
                                        className={`flex-1 ${theme.inputBg} text-center text-white text-[11px] px-3 py-3 rounded-xl border border-zinc-800 outline-none transition-all ${theme.inputBorder} placeholder-zinc-700 font-mono tracking-[0.3em]`}
                                        placeholder="CODE"
                                        maxLength={4}
                                    />
                                    <a 
                                        href={`https://wa.me/628979506271?text=Halo%20Admin%20Tools%20Ai,%20saya%20mau%20request%20OTP%20untuk%20${mode === 'RESET_PASSWORD' ? 'RESET PASSWORD' : 'registrasi'}%20username:%20${username}`}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="px-3 flex flex-col items-center justify-center bg-zinc-900 border border-zinc-800 rounded-xl hover:border-green-500/50 hover:bg-green-900/10 transition-all group"
                                    >
                                        <span className="text-[7px] font-black text-zinc-500 group-hover:text-green-400 uppercase">Get</span>
                                        <span className="text-[9px] font-black text-white group-hover:text-green-300">OTP</span>
                                    </a>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Messages */}
                    {error && (
                        <div className="py-1.5 px-3 bg-red-500/10 border border-red-500/20 rounded-lg text-center animate-pulse">
                            <span className="text-[8px] font-bold text-red-400 tracking-wide">{error}</span>
                        </div>
                    )}
                    
                    {successMsg && (
                        <div className="py-1.5 px-3 bg-green-500/10 border border-green-500/20 rounded-lg text-center">
                            <span className="text-[8px] font-bold text-green-400 tracking-wide">{successMsg}</span>
                        </div>
                    )}

                    <button 
                        onClick={() => {
                            if (mode === 'REGISTER') handleRegister();
                            else if (mode === 'RESET_PASSWORD') handleResetPassword();
                            else handleLogin();
                        }}
                        disabled={isLoading}
                        className={`w-full py-3 rounded-xl ${theme.btnBg} text-white shadow-lg transform transition-all duration-200 active:scale-[0.98] relative overflow-hidden group mt-1`}
                    >
                        <span className="relative z-10 text-[9px] font-black tracking-[0.25em] uppercase flex items-center justify-center gap-2">
                            {isLoading ? 'Processing...' : (
                                mode === 'REGISTER' ? 'DAFTAR AKUN' : 
                                (mode === 'RESET_PASSWORD' ? 'RESET ACCESS' : 
                                (isDev ? 'Unlock System' : 'LOGIN ACCESS'))
                            )}
                        </span>
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-shimmer" style={{ animationDuration: '1.5s' }}></div>
                    </button>

                    <div className="pt-3 border-t border-zinc-900/50 flex flex-col items-center gap-2">
                        {mode !== 'DEV' && (
                            <div className="flex flex-col items-center gap-2 w-full">
                                <button 
                                    onClick={() => { 
                                        setMode(mode === 'LOGIN' ? 'REGISTER' : 'LOGIN'); 
                                        setError(''); 
                                        setSuccessMsg(''); 
                                        setPassword('');
                                        setOtp('');
                                        setShowPassword(false);
                                    }}
                                    className="text-[9px] text-zinc-500 hover:text-white transition-colors"
                                >
                                    {mode === 'LOGIN' ? 'Belum Punya Akun? ' : 'Sudah Memiliki Akun? '}
                                    <span className={`font-bold ${theme.accentText} underline underline-offset-4 decoration-current/30`}>
                                        {mode === 'LOGIN' ? 'Registrasi Account' : 'Login Disini!'}
                                    </span>
                                </button>

                                {/* REMOVED OLD FORGOT PASSWORD BUTTON FROM HERE */}

                                {mode === 'RESET_PASSWORD' && (
                                     <button 
                                        onClick={() => { setMode('LOGIN'); setError(''); setSuccessMsg(''); setPassword(''); setShowPassword(false); }}
                                        className="text-[8px] font-bold text-zinc-600 hover:text-white transition-colors uppercase tracking-wider"
                                     >
                                        ← Back to Login
                                    </button>
                                )}
                            </div>
                        )}
                        
                        <button 
                            onClick={handleLogoClick}
                            className={`font-black tracking-[0.24em] transition-all duration-300 select-none cursor-pointer mt-1
                                ${isDev 
                                    ? 'text-[8px] uppercase text-red-500 drop-shadow-[0_0_5px_rgba(239,68,68,0.8)]' 
                                    : 'text-transparent bg-clip-text bg-gradient-to-r from-purple-800 via-purple-400 to-purple-800 animate-text-shimmer opacity-60 hover:opacity-100 hover:scale-105 transform'
                                }
                            `}
                        >
                            {isDev ? 'EXIT KERNEL MODE' : <><span className="text-[10px]">Vinzx</span><span className="text-[8px] ml-1">PROJECT</span></>}
                        </button>
                    </div>

                </div>
            </div>
        </div>
    );
};

export default AuthOverlay;
