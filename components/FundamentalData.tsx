import React, { useState, useEffect } from 'react';

interface CalendarEvent {
    title: string;
    country: string;
    date: string;
    impact: string;
    forecast: string;
    previous: string;
    actual?: string;
}

interface FundamentalDataProps {
    onClose: () => void;
}

const FundamentalData: React.FC<FundamentalDataProps> = ({ onClose }) => {
    const [events, setEvents] = useState<CalendarEvent[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [filter, setFilter] = useState<'ALL' | 'HIGH' | 'MEDIUM'>('ALL');
    const [currentTime, setCurrentTime] = useState(Date.now());

    // Real-time clock for "Selesai" indicator
    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(Date.now()), 1000);
        return () => clearInterval(timer);
    }, []);

    useEffect(() => {
        const fetchCalendarData = async () => {
            try {
                const now = new Date();
                const from = now.toISOString();
                const to = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000).toISOString();
                
                const targetUrl = `/api/calendar?from=${from}&to=${to}&lang=id`;
                
                const res = await fetch(targetUrl);
                if (!res.ok) throw new Error('Failed to fetch data');
                
                const json = await res.json();
                if (!json.result || !Array.isArray(json.result)) throw new Error('Invalid data format');
                
                const impactMap: Record<number, string> = {
                    1: 'High',
                    0: 'Medium',
                    [-1]: 'Low'
                };
                
                const formatValue = (val: any, unit: string) => val !== undefined && val !== null ? `${val}${unit || ''}` : '';
                
                const mappedData: CalendarEvent[] = json.result.map((item: any) => ({
                    title: item.title,
                    country: item.currency || item.country,
                    date: item.date,
                    impact: impactMap[item.importance] || 'Low',
                    forecast: formatValue(item.forecast, item.unit),
                    previous: formatValue(item.previous, item.unit),
                    actual: formatValue(item.actual, item.unit)
                }));
                
                // Sort by date ascending
                mappedData.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
                
                setEvents(mappedData);
                setError(null);
            } catch (err) {
                if (events.length === 0) {
                    setError('Failed to load fundamental data. Please try again later.');
                }
            } finally {
                setLoading(false);
            }
        };

        fetchCalendarData();
        const interval = setInterval(fetchCalendarData, 60000); // Auto refresh every 1 minute
        return () => clearInterval(interval);
    }, []);

    const getImpactColor = (impact: string) => {
        switch (impact.toLowerCase()) {
            case 'high': return 'bg-rose-500 shadow-[0_0_12px_rgba(244,63,94,0.8)]';
            case 'medium': return 'bg-amber-500 shadow-[0_0_12px_rgba(245,158,11,0.8)]';
            case 'low': return 'bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.8)]';
            default: return 'bg-slate-500';
        }
    };

    const getImpactBg = (impact: string) => {
        switch (impact.toLowerCase()) {
            case 'high': return 'bg-gradient-to-br from-rose-500/15 to-transparent border-rose-500/30';
            case 'medium': return 'bg-gradient-to-br from-amber-500/15 to-transparent border-amber-500/30';
            case 'low': return 'bg-gradient-to-br from-emerald-500/15 to-transparent border-emerald-500/30';
            default: return 'bg-gradient-to-br from-slate-500/15 to-transparent border-slate-500/30';
        }
    };

    const filteredEvents = events.filter(e => {
        if (filter === 'HIGH') return e.impact.toLowerCase() === 'high';
        if (filter === 'MEDIUM') return e.impact.toLowerCase() === 'medium' || e.impact.toLowerCase() === 'high';
        return true;
    });

    return (
        <div className="absolute inset-0 z-[250] bg-black/95 backdrop-blur-2xl flex flex-col animate-in fade-in duration-300">
            {/* Header */}
            <div className="flex items-center justify-between p-5 border-b border-white/10 bg-gradient-to-r from-purple-900/20 to-black/50">
                <div className="flex flex-col">
                    <h2 className="text-sm font-black tracking-widest text-white drop-shadow-[0_0_15px_rgba(168,85,247,0.6)] uppercase">
                        Kalender Ekonomi Dunia 🌎
                    </h2>
                    <span className="text-[10px] text-zinc-400 font-mono mt-0.5 tracking-wider flex items-center gap-1">
                        Data : Bloomberg Terminal <span className="animate-pulse-fast">🟣</span>
                    </span>
                </div>
                <button 
                    onClick={onClose} 
                    className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-white/10 hover:scale-105 transition-all"
                >
                    ✕
                </button>
            </div>

            {/* Filters */}
            <div className="px-5 py-3.5 flex gap-2.5 overflow-x-auto no-scrollbar border-b border-white/5 bg-black/40">
                {['ALL', 'HIGH', 'MEDIUM'].map((f) => (
                    <button
                        key={f}
                        onClick={() => setFilter(f as any)}
                        className={`px-4 py-2 rounded-lg text-[10px] font-bold tracking-wider transition-all whitespace-nowrap ${
                            filter === f 
                            ? 'bg-purple-600 text-white shadow-[0_0_20px_rgba(147,51,234,0.4)] border border-purple-400/50' 
                            : 'bg-white/5 text-zinc-400 border border-white/10 hover:text-white hover:bg-white/10'
                        }`}
                    >
                        {f === 'ALL' ? 'SEMUA IMPACT' : f === 'HIGH' ? 'IMPACT TINGGI' : 'IMPACT SEDANG'}
                    </button>
                ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-3.5 custom-scrollbar">
                {loading ? (
                    <div className="flex flex-col items-center justify-center h-full space-y-5">
                        <div className="w-10 h-10 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                        <span className="text-xs font-mono text-zinc-500 animate-pulse tracking-widest">MENGAMBIL DATA FUNDAMENTAL...</span>
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center h-full text-red-500 space-y-3">
                        <span className="text-3xl">⚠️</span>
                        <span className="text-xs font-mono text-center tracking-wider">{error}</span>
                    </div>
                ) : filteredEvents.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-zinc-600 space-y-3">
                        <span className="text-3xl opacity-50">📅</span>
                        <span className="text-xs font-mono tracking-widest">TIDAK ADA DATA MINGGU INI</span>
                    </div>
                ) : (
                    filteredEvents.map((event, idx) => {
                        const eventDate = new Date(event.date);
                        const isToday = eventDate.toDateString() === new Date().toDateString();
                        const isPast = eventDate.getTime() < currentTime;
                        
                        return (
                            <div 
                                key={idx} 
                                className={`relative p-4 rounded-2xl border ${getImpactBg(event.impact)} backdrop-blur-md transition-all hover:scale-[1.02] hover:shadow-lg ${isPast ? 'opacity-50 grayscale-[40%]' : ''}`}
                            >
                                {isPast && (
                                    <div className="absolute top-3 right-3 px-2 py-0.5 rounded bg-black/60 border border-white/10 text-[8px] font-bold text-zinc-400 tracking-widest uppercase">
                                        Selesai
                                    </div>
                                )}
                                <div className="flex justify-between items-start mb-3">
                                    <div className="flex items-center gap-3">
                                        <div className="flex flex-col items-center justify-center w-12 h-12 rounded-xl bg-black/40 border border-white/10 shadow-inner">
                                            <span className="text-[10px] font-bold text-zinc-300 uppercase">{eventDate.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }).replace(':', '.')}</span>
                                        </div>
                                        <div className="flex flex-col">
                                            <div className="flex items-center gap-2 mb-0.5">
                                                <span className="text-xs font-black text-white tracking-widest">{event.country}</span>
                                                <div className={`w-2 h-2 rounded-full ${getImpactColor(event.impact)}`} title={`${event.impact} Impact`} />
                                            </div>
                                            <span className="text-sm font-bold text-zinc-200 leading-snug">{event.title}</span>
                                            <span className="text-[9px] text-zinc-400 font-mono mt-1 bg-black/30 px-1.5 py-0.5 rounded w-fit border border-white/5">
                                                {isToday ? 'HARI INI' : eventDate.toLocaleDateString('id-ID', { weekday: 'short', month: 'short', day: 'numeric' })}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-white/10 bg-black/20 rounded-xl p-2.5">
                                    <div className="flex flex-col items-center text-center">
                                        <span className="text-[8px] text-zinc-500 font-bold tracking-widest uppercase mb-1">Aktual (Hasil)</span>
                                        <span className={`text-xs font-mono font-bold ${event.actual ? 'text-white drop-shadow-md' : 'text-zinc-600'}`}>{event.actual || '-'}</span>
                                    </div>
                                    <div className="flex flex-col items-center text-center border-x border-white/5">
                                        <span className="text-[8px] text-zinc-500 font-bold tracking-widest uppercase mb-1">Prediksi (Perkiraan)</span>
                                        <span className="text-xs font-mono text-zinc-300">{event.forecast || '-'}</span>
                                    </div>
                                    <div className="flex flex-col items-center text-center">
                                        <span className="text-[8px] text-zinc-500 font-bold tracking-widest uppercase mb-1">Sblmnya (Lalu)</span>
                                        <span className="text-xs font-mono text-zinc-400">{event.previous || '-'}</span>
                                    </div>
                                </div>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};

export default FundamentalData;
