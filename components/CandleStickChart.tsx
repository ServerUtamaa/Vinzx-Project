
import React, { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { Candle, TimeFrame, Asset } from '../types';
import { calculateEMAArray, calculateBollingerBandsArray } from '../utils/indicators';

interface ChartProps {
  data: Candle[];
  priceOffset: number;
  timeframe: TimeFrame; 
  currentAsset: Asset;
}

interface VisibleItem {
    x: number;
    index: number;
    candle: Candle | null;
}

const CandleStickChart: React.FC<ChartProps> = ({ data, priceOffset, timeframe, currentAsset }) => {
  const Y_AXIS_WIDTH = 50; 
  const INITIAL_CANDLE_WIDTH = 10;
  const MIN_CANDLE_WIDTH = 1; 
  const MAX_CANDLE_WIDTH = 150; 
  
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  // Start with a float, but we'll snap it for rendering
  const [candleWidth, setCandleWidth] = useState(INITIAL_CANDLE_WIDTH);
  
  const [offsetX, setOffsetX] = useState(0); 
  const [verticalOffset, setVerticalOffset] = useState(0);
  const [nextCandleLabel, setNextCandleLabel] = useState(""); 

  const [isDragging, setIsDragging] = useState(false);
  const [manualRange, setManualRange] = useState<{min: number, max: number} | null>(null);
  const [showIndicators, setShowIndicators] = useState(true);

  const containerRef = useRef<HTMLDivElement>(null);
  const interactionRef = useRef<{
    startX: number; startY: number; 
    startOffsetX: number; 
    startVerticalOffset: number; 
    startRange: { min: number, max: number };
    startPinchDist: number; startPinchWidth: number; mode: 'NONE' | 'CHART' | 'AXIS' | 'PINCH';
  }>({ 
      startX: 0, startY: 0, 
      startOffsetX: 0, 
      startVerticalOffset: 0, 
      startRange: { min: 0, max: 0 }, 
      startPinchDist: 0, startPinchWidth: 0, mode: 'NONE' 
  });

  useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) setDimensions({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const calculateCountdown = () => {
        const now = Date.now();
        let periodMs = 15 * 60 * 1000;
        switch (timeframe) {
            case 'M1': periodMs = 60 * 1000; break;
            case 'M5': periodMs = 5 * 60 * 1000; break;
            case 'M15': periodMs = 15 * 60 * 1000; break;
            case 'M30': periodMs = 30 * 60 * 1000; break;
            case 'H1': periodMs = 60 * 60 * 1000; break;
            case 'H4': periodMs = 4 * 60 * 60 * 1000; break;
            case 'D1': periodMs = 24 * 60 * 60 * 1000; break;
        }
        const nextTimestamp = (Math.floor(now / periodMs) + 1) * periodMs;
        const diff = nextTimestamp - now;
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);
        const pad = (n: number) => n.toString().padStart(2, '0');
        let label = hours > 0 ? `${pad(hours)}:${pad(minutes)}:${pad(seconds)}` : `${pad(minutes)}:${pad(seconds)}`;
        setNextCandleLabel(label);
    };
    calculateCountdown(); 
    const interval = setInterval(calculateCountdown, 1000); 
    return () => clearInterval(interval);
  }, [timeframe]); 

  const isMarketClosed = useMemo(() => {
    const isCrypto = [Asset.BTCUSD, Asset.ETHUSD, Asset.SOLUSD, Asset.BNBUSD].includes(currentAsset);
    const now = new Date();
    const day = now.getDay(); 
    return !isCrypto && (day === 0 || day === 6);
  }, [currentAsset]);

  const { width, height } = dimensions;
  const chartWidth = Math.max(0, width - Y_AXIS_WIDTH);
  const chartHeight = Math.max(0, height - 12); 
  
  // --- TRADING VIEW PIXEL PERFECT LOGIC ---
  const renderWidth = candleWidth < 4 ? Math.round(candleWidth) : candleWidth;
  const gap = renderWidth <= 2 ? 0 : (renderWidth < 10 ? 1 : Math.ceil(renderWidth * 0.2));
  const candleUnit = renderWidth + gap;
  
  const maxVisibleCandles = candleUnit > 0 ? Math.ceil(chartWidth / candleUnit) + 5 : 10;
  
  const rawStartIndex = data.length - maxVisibleCandles - offsetX;
  
  const fullIndicators = useMemo(() => {
    if(data.length === 0) return null;
    return {
        ema50: calculateEMAArray(data, 50),
        ema200: calculateEMAArray(data, 200),
        bb: calculateBollingerBandsArray(data, 20, 2)
    };
  }, [data]);

  const { visibleItems, visibleMin, visibleMax } = useMemo(() => {
    if (data.length === 0 || width === 0) return { visibleItems: [], visibleMin: 0, visibleMax: 100 };
    const items: VisibleItem[] = [];
    let min = Infinity; let max = -Infinity;

    for (let i = 0; i < maxVisibleCandles; i++) {
        const realIndex = Math.floor(rawStartIndex + i);
        // Use integer math for X to prevent sub-pixel blurring
        const x = i * candleUnit;
        
        if (realIndex >= 0 && realIndex < data.length) {
            const raw = data[realIndex];
            if (raw && !isNaN(raw.close)) {
                const c = { ...raw, open: raw.open + priceOffset, high: raw.high + priceOffset, low: raw.low + priceOffset, close: raw.close + priceOffset };
                items.push({ x, index: realIndex, candle: c });
                
                // CRITICAL FIX: Only calculate Min/Max based on CANDLES.
                // Ignoring indicators for auto-scaling prevents "squashed" candles.
                if (c.low < min) min = c.low; 
                if (c.high > max) max = c.high;
            }
        }
    }
    
    // Fallback if no data visible
    if (min === Infinity) { min = 0; max = 100; }
    
    // Padding logic: Ensure candles take up about 70-80% of screen height
    const range = max - min;
    const padding = range === 0 ? max * 0.1 : range * 0.15; // 15% padding top/bottom
    
    return { visibleItems: items, visibleMin: min - padding, visibleMax: max + padding };
  }, [data, rawStartIndex, maxVisibleCandles, priceOffset, candleUnit, width]);

  const rawMin = manualRange ? manualRange.min : visibleMin;
  const rawMax = manualRange ? manualRange.max : visibleMax;
  
  const minPrice = rawMin + verticalOffset;
  const maxPrice = rawMax + verticalOffset;
  const priceRange = maxPrice - minPrice;

  const getY = useCallback((price: number) => {
    if (priceRange === 0 || chartHeight === 0) return chartHeight / 2;
    return chartHeight - ((price - minPrice) / priceRange) * chartHeight;
  }, [chartHeight, minPrice, priceRange]);

  const createPath = (indicatorData: number[]) => {
      let path = ""; let first = true;
      for (let i = 0; i < maxVisibleCandles; i++) {
          const realIndex = Math.floor(rawStartIndex + i);
          if (realIndex >= 0 && realIndex < data.length) {
              const val = indicatorData[realIndex];
              if (val !== undefined && !isNaN(val)) {
                  const centerOffset = renderWidth <= 2 ? (renderWidth === 1 ? 0.5 : 1) : renderWidth / 2;
                  const x = i * candleUnit + centerOffset;
                  const y = getY(val + priceOffset);
                  
                  // Optimization: Don't draw lines way off screen to improve performance
                  if (y < -1000 || y > chartHeight + 1000) {
                      if (!first) path += ` L ${x} ${y}`; // Keep continuity but don't render crazy paths
                  } else {
                      if (first) { path += `M ${x} ${y}`; first = false; } else { path += ` L ${x} ${y}`; }
                  }
              } else { first = true; }
          }
      }
      return path;
  };

  const handleStart = (cx: number, cy: number, e: any) => {
    const rect = containerRef.current?.getBoundingClientRect(); if (!rect) return;
    if (e.touches && e.touches.length === 2) { 
       interactionRef.current = { ...interactionRef.current, startPinchDist: Math.hypot(e.touches[0].clientX-e.touches[1].clientX, e.touches[0].clientY-e.touches[1].clientY), startPinchWidth: candleWidth, mode: 'PINCH' }; return; 
    }
    const x = cx - rect.left; const isAxis = x > chartWidth;
    
    interactionRef.current = { 
        startX: cx, 
        startY: cy, 
        startOffsetX: offsetX, 
        startVerticalOffset: verticalOffset, 
        startRange: manualRange ? { ...manualRange } : { min: visibleMin, max: visibleMax }, 
        startPinchDist: 0, startPinchWidth: 0, 
        mode: isAxis ? 'AXIS' : 'CHART' 
    };
    
    setIsDragging(true); if (isAxis && !manualRange) setManualRange({ min: visibleMin, max: visibleMax });
  };

  const handleMove = (cx: number, cy: number, e: any) => {
    if (interactionRef.current.mode === 'PINCH' && e.touches && e.touches.length === 2) {
        const dist = Math.hypot(e.touches[0].clientX-e.touches[1].clientX, e.touches[0].clientY-e.touches[1].clientY);
        const scale = dist / interactionRef.current.startPinchDist;
        setCandleWidth(Math.max(MIN_CANDLE_WIDTH, Math.min(MAX_CANDLE_WIDTH, interactionRef.current.startPinchWidth * scale))); return;
    }
    if (!isDragging || interactionRef.current.mode === 'NONE') return;
    
    const dy = cy - interactionRef.current.startY; 
    const dx = cx - interactionRef.current.startX;
    
    if (interactionRef.current.mode === 'AXIS') {
        const scale = 1 + dy * 0.003; 
        const center = (interactionRef.current.startRange.max + interactionRef.current.startRange.min)/2; 
        const span = interactionRef.current.startRange.max - interactionRef.current.startRange.min;
        setManualRange({ min: center - span*scale/2, max: center + span*scale/2 });
    } else if (interactionRef.current.mode === 'CHART') {
        const deltaOffset = (dx / candleUnit) * 1.5;
        const potentialOffset = interactionRef.current.startOffsetX + deltaOffset;
        
        const minOffset = -50; 
        const maxOffset = data.length + 50; 
        
        setOffsetX(Math.max(minOffset, Math.min(maxOffset, potentialOffset)));
        
        const currentPriceRange = (interactionRef.current.startRange.max - interactionRef.current.startRange.min);
        const pixelToPriceRatio = currentPriceRange / chartHeight;
        const priceDiff = dy * pixelToPriceRatio;
        setVerticalOffset(interactionRef.current.startVerticalOffset + priceDiff);
    }
  };

  const handleEnd = () => { setIsDragging(false); interactionRef.current.mode = 'NONE'; };

  const currentPrice = data.length > 0 ? data[data.length-1].close + priceOffset : 0;
  const currentPriceY = getY(currentPrice);
  const gridLines = []; const gridCount = Math.floor(height / 50);
  for(let i=0; i<=gridCount; i++) gridLines.push(minPrice + (priceRange * (i/gridCount)));

  if (data.length === 0) return (
      <div ref={containerRef} className="w-full h-full flex flex-col items-center justify-center text-zinc-500 gap-2">
         <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
         <span className="text-[10px] tracking-[0.2em] font-black animate-pulse">CONNECTING ENGINE...</span>
      </div>
  );

  if (isMarketClosed) {
      return (
          <div ref={containerRef} className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-[#050505] overflow-hidden select-none">
             {/* ... Market Closed UI ... */}
             <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(220,38,38,0.05),_transparent_70%)]"></div>
             <div className="absolute inset-0 bg-[linear-gradient(to_right,#111_1px,transparent_1px),linear-gradient(to_bottom,#111_1px,transparent_1px)] bg-[size:24px_24px] opacity-20"></div>
             <div className="z-10 flex flex-col items-center animate-in zoom-in-95 duration-500">
                <div className="relative mb-6">
                   <div className="absolute inset-0 bg-red-500 blur-[50px] opacity-20 rounded-full"></div>
                   <div className="text-[72px] relative z-10 drop-shadow-[0_10px_10px_rgba(0,0,0,0.5)] animate-[bounce_3s_infinite]">🗿</div>
                </div>
                <div className="relative">
                    <h2 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-br from-red-500 via-orange-500 to-red-600 tracking-[0.25em] drop-shadow-[0_0_25px_rgba(239,68,68,0.4)] uppercase text-center leading-tight">MARKET<br/>TUTUP</h2>
                </div>
             </div>
          </div>
      );
  }

  return (
    <div ref={containerRef} className={`w-full h-full ${isDragging ? 'cursor-grabbing' : 'cursor-grab'} select-none`}
      onMouseDown={e => handleStart(e.clientX, e.clientY, e)} onMouseMove={e => handleMove(e.clientX, e.clientY, e)} onMouseUp={handleEnd} onMouseLeave={handleEnd}
      onTouchStart={e => handleStart(e.touches[0].clientX, e.touches[0].clientY, e)} onTouchMove={e => handleMove(e.touches[0].clientX, e.touches[0].clientY, e)} onTouchEnd={handleEnd}
      onWheel={e => {
        setCandleWidth(prev => {
            const step = prev < 5 ? 0.5 : (prev < 10 ? 1 : 2);
            return Math.max(MIN_CANDLE_WIDTH, Math.min(MAX_CANDLE_WIDTH, prev + -Math.sign(e.deltaY) * step));
        });
      }}
      onDoubleClick={() => { setManualRange(null); setOffsetX(0); setVerticalOffset(0); setCandleWidth(INITIAL_CANDLE_WIDTH); }}
    >
      <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="overflow-visible bg-[#050505]">
        <defs>
          <clipPath id="chart-area"><rect x="0" y="0" width={chartWidth} height={chartHeight} /></clipPath>
          <filter id="neon-cyan"><feGaussianBlur stdDeviation="2" result="c"/><feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <filter id="neon-purple"><feGaussianBlur stdDeviation="2" result="c"/><feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        {gridLines.map((p, i) => (
           <g key={i}><line x1="0" y1={getY(p)} x2={width} y2={getY(p)} stroke="#151515" strokeWidth="1"/><text x={width-45} y={getY(p)+3} fill="#444" fontSize="9" fontFamily="monospace">{p.toFixed(2)}</text></g>
        ))}
        <line x1={chartWidth} y1={0} x2={chartWidth} y2={height} stroke="#222" />
        <g clipPath="url(#chart-area)">
          {showIndicators && fullIndicators && (
             <>
                <path d={createPath(fullIndicators.ema50)} stroke="#00f3ff" strokeWidth="2" fill="none" filter="url(#neon-cyan)" opacity="0.9"/>
                <path d={createPath(fullIndicators.ema200)} stroke="#a855f7" strokeWidth="2" fill="none" filter="url(#neon-purple)" opacity="0.9"/>
                <path d={createPath(fullIndicators.bb.upper)} stroke="#555" strokeWidth="1" strokeDasharray="3 3" fill="none"/>
                <path d={createPath(fullIndicators.bb.lower)} stroke="#555" strokeWidth="1" strokeDasharray="3 3" fill="none"/>
             </>
          )}
          {visibleItems.map(item => {
             if(!item.candle) return null;
             const c = item.candle; const isUp = c.close >= c.open; const color = isUp ? '#089981' : '#f23645';
             const y1 = getY(c.open); const y2 = getY(c.close);
             
             // --- RENDERING ENGINE v2 (TradingView Style) ---

             // 1. BAR MODE (Zoomed Out / Width < 3px)
             if (renderWidth < 3) {
                 const snappedX = Math.floor(item.x) + 0.5;
                 return (
                     <line 
                        key={item.index}
                        x1={snappedX}
                        y1={getY(c.high)}
                        x2={snappedX}
                        y2={getY(c.low)}
                        stroke={color}
                        strokeWidth={renderWidth}
                        shapeRendering={renderWidth === 1 ? "crispEdges" : "auto"}
                     />
                 );
             }

             // 2. CANDLE MODE (Zoomed In / Width >= 3px)
             const wickX = Math.floor(item.x + renderWidth / 2) + 0.5;
             const bodyHeight = Math.max(Math.abs(y1-y2), 1);
             const bodyY = Math.min(y1, y2);

             return (
                 <g key={item.index}>
                     <line x1={wickX} y1={getY(c.high)} x2={wickX} y2={getY(c.low)} stroke={color} strokeWidth={1} shapeRendering="crispEdges"/>
                     <rect x={item.x} y={bodyY} width={renderWidth} height={bodyHeight} fill={color} shapeRendering="crispEdges" />
                 </g>
             );
          })}
        </g>
        <line x1="0" y1={currentPriceY} x2={width} y2={currentPriceY} stroke={isMarketClosed ? '#444' : (data[data.length-1].close >= data[data.length-2].close ? '#089981' : '#f23645')} strokeDasharray="4 4" opacity="0.8"/>
        <rect x={width-Y_AXIS_WIDTH} y={currentPriceY-9} width={Y_AXIS_WIDTH} height={18} fill={isMarketClosed ? '#333' : (data[data.length-1].close >= data[data.length-2].close ? '#089981' : '#f23645')} rx="2"/>
        <text x={width-Y_AXIS_WIDTH+4} y={currentPriceY+4} fill="white" fontSize="10" fontWeight="bold" fontFamily="monospace">{currentPrice.toFixed(2)}</text>
        {showIndicators && (
            <g transform="translate(15, 20)">
                <text fill="#00f3ff" fontSize="10" fontWeight="bold" fontFamily="sans-serif">EMA 50</text>
                <text x="50" fill="#a855f7" fontSize="10" fontWeight="bold" fontFamily="sans-serif">EMA 200</text>
                <text x="110" fill="#666" fontSize="10" fontWeight="bold" fontFamily="sans-serif">BB(20,2)</text>
            </g>
        )}
        <g transform={`translate(${chartWidth - 30}, 15)`} onClick={(e) => {e.stopPropagation(); setShowIndicators(!showIndicators);}} style={{cursor:'pointer'}}>
            {showIndicators ? (
               <path d="M1 6s2.5-4 6-4 6 4 6 4 6 4-2.5 4-6 4-6-4-6-4z" stroke="#a855f7" strokeWidth="1.5" fill="none" />
            ) : (
               <g stroke="#444"><path d="M1 6s2.5-4 6-4 6 4 6 4-2.5 4-6 4-6-4-6-4z" fill="none"/><line x1="1" y1="11" x2="13" y2="1"/></g>
            )}
            <circle cx="7" cy="6" r="2" fill="#a855f7" opacity={showIndicators ? 1 : 0} />
        </g>
        <text x={chartWidth-10} y={chartHeight+9} fill="#777" fontSize="9" textAnchor="end" fontFamily="monospace" fontWeight="bold">NEXT CANDLE: <tspan fill="#00ff9d">{nextCandleLabel}</tspan></text>
      </svg>
    </div>
  );
};

export default React.memo(CandleStickChart);
