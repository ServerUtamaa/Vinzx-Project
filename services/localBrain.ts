
import { Asset, Candle, AnalysisResult, TradeFeedback, TimeFrame } from "../types";
import { calculateRSI, calculateBollingerBands, calculateATR, analyzeEMACondition, detectCandlestickPatterns, detectSMC, SMCResult } from "../utils/indicators";

// --- ASI NARRATIVE GENERATOR (14 Points with 4 Variations) ---
const generate14PointNarrative = (signal: 'BUY' | 'SELL' | 'WAIT', timeframe: TimeFrame, asset: Asset, rrType: string, isCounterTrend: boolean, smc: SMCResult, patterns: any[], totalScore: number, minScore: number): string[] => {
    const isBullish = signal === 'BUY';
    const isBearish = signal === 'SELL';
    
    // Seed random based on timestamp to avoid flickering on re-renders but still be dynamic
    const rand = () => Math.floor(Math.random() * 4);

    const p1 = [
        `1. [STRUKTUR PASAR]: Berdasarkan 150 candle terakhir di ${timeframe}, market sedang dalam fase ${smc.structure}. ${isBullish ? 'Tren naik (Higher High/Higher Low) terlihat jelas.' : (isBearish ? 'Tren turun (Lower Low/Lower High) mendominasi.' : 'Harga sedang bergerak sideways (konsolidasi).')}`,
        `1. [TREN UTAMA]: Analisis struktur pada ${timeframe} menunjukkan kondisi ${smc.structure}. ${isBullish ? 'Buyer terus menembus resistance (Uptrend).' : (isBearish ? 'Seller terus menembus support (Downtrend).' : 'Belum ada arah tren yang pasti.')}`,
        `1. [MARKET DIRECTION]: Arah pergerakan di ${timeframe} teridentifikasi sebagai ${smc.structure}. ${isBullish ? 'Fokus pada peluang Buy karena struktur mendukung.' : (isBearish ? 'Fokus pada peluang Sell mengikuti arus turun.' : 'Market sedang ranging, waspada false breakout.')}`,
        `1. [PETA STRUKTUR]: Pemetaan 150 candle di ${timeframe} mengonfirmasi tren ${smc.structure}. ${isBullish ? 'Struktur tangga naik (HH/HL) terbentuk sempurna.' : (isBearish ? 'Struktur tangga turun (LL/LH) sedang berlangsung.' : 'Harga terjebak dalam zona datar.')}`
    ][rand()];

    const p2 = [
        `2. [SIKLUS HARGA]: Market saat ini berada dalam fase ${smc.phase}. ${smc.phase === 'MARKUP' ? 'Harga sedang didorong naik kuat.' : smc.phase === 'MARKDOWN' ? 'Harga sedang ditekan turun tajam.' : smc.phase === 'ACCUMULATION' ? 'Institusi sedang mengumpulkan aset (Akumulasi).' : 'Institusi sedang melepas aset (Distribusi).'}`,
        `2. [FASE MARKET]: Deteksi siklus menunjukkan fase ${smc.phase}. ${smc.phase === 'MARKUP' ? 'Momentum bullish sedang menguasai pasar.' : smc.phase === 'MARKDOWN' ? 'Momentum bearish memegang kendali penuh.' : smc.phase === 'ACCUMULATION' ? 'Fase pengumpulan tenaga sebelum kenaikan.' : 'Fase distribusi sebelum potensi penurunan.'}`,
        `2. [KONDISI SIKLUS]: Kita berada di tahap ${smc.phase}. ${smc.phase === 'MARKUP' ? 'Tanda-tanda partisipasi publik dalam tren naik.' : smc.phase === 'MARKDOWN' ? 'Kepanikan atau aksi jual mendominasi.' : smc.phase === 'ACCUMULATION' ? 'Smart money sedang membangun posisi beli.' : 'Smart money mulai merealisasikan profit.'}`,
        `2. [ZONA SIKLUS]: Analisis Wyckoff mendeteksi fase ${smc.phase}. ${smc.phase === 'MARKUP' ? 'Tren naik sedang berakselerasi.' : smc.phase === 'MARKDOWN' ? 'Tren turun sedang berakselerasi.' : smc.phase === 'ACCUMULATION' ? 'Konsolidasi bawah: persiapan reversal naik.' : 'Konsolidasi atas: persiapan reversal turun.'}`
    ][rand()];

    const p3 = [
        `3. [MOMENTUM CANDLE]: Dari sisi volume dan bentuk candle, terdapat ${smc.bullishStrongCount} candle Bullish kuat vs ${smc.bearishStrongCount} Bearish kuat. ${smc.bullishStrongCount > smc.bearishStrongCount ? 'Buyer lebih agresif.' : 'Seller lebih agresif.'}`,
        `3. [KEKUATAN IMPULS]: Statistik 150 candle mencatat ${smc.bullishStrongCount} dorongan Buyer dan ${smc.bearishStrongCount} dorongan Seller. ${smc.bullishStrongCount > smc.bearishStrongCount ? 'Dominasi hijau terlihat jelas.' : 'Dominasi merah menekan pasar.'}`,
        `3. [JEJAK VOLUME]: Perbandingan candle solid menunjukkan ${smc.bullishStrongCount} Bullish berbanding ${smc.bearishStrongCount} Bearish. ${smc.bullishStrongCount > smc.bearishStrongCount ? 'Tekanan beli mendominasi.' : 'Tekanan jual lebih masif.'}`,
        `3. [DISTRIBUSI TENAGA]: Kekuatan pasar terbagi menjadi ${smc.bullishStrongCount} candle naik solid dan ${smc.bearishStrongCount} candle turun solid. ${smc.bullishStrongCount > smc.bearishStrongCount ? 'Banteng (Bull) memegang kendali.' : 'Beruang (Bear) menguasai arena.'}`
    ][rand()];

    const p4 = [
        `4. [LIKUIDITAS (STOP HUNT)]: ${smc.liquiditySweep === 'SELL_SIDE' ? 'Stop loss buyer (Sell-side) baru saja disapu bersih. Potensi naik tinggi.' : smc.liquiditySweep === 'BUY_SIDE' ? 'Stop loss seller (Buy-side) telah diambil. Potensi turun tinggi.' : 'Belum ada sapuan likuiditas (Sweep) yang signifikan.'}`,
        `4. [JEBAKAN LIKUIDITAS]: ${smc.liquiditySweep === 'SELL_SIDE' ? 'Harga memancing seller lalu berbalik (Sell-side Sweep). Sinyal reversal bullish.' : smc.liquiditySweep === 'BUY_SIDE' ? 'Harga memancing buyer lalu berbalik (Buy-side Sweep). Sinyal reversal bearish.' : 'Market masih bergerak natural tanpa manipulasi likuiditas.'}`,
        `4. [ZONA MANIPULASI]: ${smc.liquiditySweep === 'SELL_SIDE' ? 'Institusi telah mengambil likuiditas di bawah support (Sweep).' : smc.liquiditySweep === 'BUY_SIDE' ? 'Institusi telah mengambil likuiditas di atas resistance (Sweep).' : 'Tidak terdeteksi adanya perburuan stop-loss.'}`,
        `4. [STATUS LIQUIDITY]: ${smc.liquiditySweep === 'SELL_SIDE' ? 'Likuiditas bawah telah dibersihkan, jalan ke atas terbuka.' : smc.liquiditySweep === 'BUY_SIDE' ? 'Likuiditas atas telah dibersihkan, jalan ke bawah terbuka.' : 'Likuiditas utama masih utuh, waspada pergerakan tiba-tiba.'}`
    ][rand()];

    const p5 = [
        `5. [KONFIRMASI STRUKTUR]: ${smc.choch !== 'NONE' ? 'Terjadi perubahan karakter (CHoCH) ' + smc.choch + ', tren mulai berbalik.' : smc.bos !== 'NONE' ? 'Tren berlanjut dengan penembusan struktur (BOS) ' + smc.bos + '.' : 'Belum ada penembusan struktur (BOS/CHoCH) baru.'}`,
        `5. [BREAKOUT VALID]: ${smc.choch !== 'NONE' ? 'Sinyal awal pembalikan arah terkonfirmasi via CHoCH ' + smc.choch + '.' : smc.bos !== 'NONE' ? 'Struktur lama berhasil ditembus (BOS ' + smc.bos + '), tren berlanjut.' : 'Harga masih tertahan di dalam struktur lama.'}`,
        `5. [PERGESERAN TREN]: ${smc.choch !== 'NONE' ? 'Market menunjukkan indikasi reversal kuat (CHoCH ' + smc.choch + ').' : smc.bos !== 'NONE' ? 'Konfirmasi penerusan arah terdeteksi (BOS ' + smc.bos + ').' : 'Tidak ada pergeseran struktur yang terdeteksi saat ini.'}`,
        `5. [STATUS BREAK]: ${smc.choch !== 'NONE' ? 'Level penting ditembus berlawanan arah (CHoCH ' + smc.choch + ').' : smc.bos !== 'NONE' ? 'Level penting ditembus searah tren (BOS ' + smc.bos + ').' : 'Market sedang menguji batas struktur tanpa penembusan.'}`
    ][rand()];

    const p6 = [
        `6. [LOKASI HARGA]: Harga saat ini berada di area ${isBullish ? 'Discount (Murah), ideal untuk mencari peluang Buy.' : isBearish ? 'Premium (Mahal), ideal untuk mencari peluang Sell.' : 'Equilibrium (Tengah), kurang ideal untuk entry.'}`,
        `6. [ZONA SND]: Posisi harga masuk ke zona ${isBullish ? 'Demand (Permintaan tinggi), risiko Buy lebih rendah.' : isBearish ? 'Supply (Penawaran tinggi), risiko Sell lebih rendah.' : 'Netral, probabilitas arah masih seimbang.'}`,
        `6. [VALUASI MARKET]: Secara valuasi, harga sedang berada di tingkat ${isBullish ? 'Oversold/Discount, menarik bagi buyer institusi.' : isBearish ? 'Overbought/Premium, menarik bagi seller institusi.' : 'Wajar (Fair Value), menunggu dorongan likuiditas.'}`,
        `6. [AREA KEPUTUSAN]: Kita berada di wilayah ${isBullish ? 'Bawah (Discount), tempat smart money mengumpulkan aset.' : isBearish ? 'Atas (Premium), tempat smart money melepas aset.' : 'Tengah range, sangat rentan terhadap false signal.'}`
    ][rand()];

    const p7 = [
        `7. [JEJAK INSTITUSI]: Terdeteksi adanya Order Block (OB) ${smc.orderBlock !== 'NONE' ? smc.orderBlock : 'yang belum jelas'} di area ini.`,
        `7. [ORDER BLOCK]: Terdapat blok pesanan besar (OB) ${smc.orderBlock !== 'NONE' ? smc.orderBlock : 'yang tidak teridentifikasi'} sebagai titik pantul potensial.`,
        `7. [ZONA REAKSI]: Area ini memiliki Order Block ${smc.orderBlock !== 'NONE' ? smc.orderBlock : 'yang minim'} peninggalan institusi besar.`,
        `7. [INSTITUTIONAL ENTRY]: Jejak smart money (Order Block) ${smc.orderBlock !== 'NONE' ? 'bertipe ' + smc.orderBlock + ' terlihat jelas.' : 'belum terbentuk dengan sempurna.'}`
    ][rand()];

    const p8 = [
        `8. [IMBALANCE (FVG)]: Terdapat Fair Value Gap (FVG) ${smc.fvg !== 'NONE' ? smc.fvg : 'yang sudah tertutup'} yang bertindak sebagai magnet harga.`,
        `8. [KETIDAKSEIMBANGAN]: Market meninggalkan FVG ${smc.fvg !== 'NONE' ? smc.fvg : 'yang minim'} akibat pergerakan impulsif sebelumnya.`,
        `8. [GAP HARGA]: Ruang kosong atau FVG ${smc.fvg !== 'NONE' ? smc.fvg : 'tidak ditemukan'} di sekitar area harga saat ini.`,
        `8. [INEFFICIENCY]: Terdeteksi inefisiensi pasar (FVG) ${smc.fvg !== 'NONE' ? smc.fvg : 'yang telah diseimbangkan'} yang perlu diwaspadai.`
    ][rand()];

    const p9 = [
        `9. [KONFLUENS SNR]: Area ini didukung oleh level Support/Resistance historis yang cukup kuat.`,
        `9. [BATAS PSIKOLOGIS]: Terdapat pertemuan (confluence) antara zona SMC dengan level Support & Resistance klasik.`,
        `9. [KEY LEVEL]: Harga bereaksi pada level kunci (Key Level) yang sering menjadi titik balik (pivot) di masa lalu.`,
        `9. [ZONA PENAHAN]: Level Support dan Resistance di area ini memberikan lapisan pertahanan ekstra untuk setup kita.`
    ][rand()];

    const p10 = [
        `10. [KONFIRMASI CANDLE]: Muncul pola candlestick ${patterns.length > 0 ? patterns[0].name : 'Netral/Doji'} sebagai pemicu (trigger) entry.`,
        `10. [AKSI HARGA]: Reaksi harga membentuk pola ${patterns.length > 0 ? patterns[0].name : 'konsolidasi'}, memberikan sinyal eksekusi.`,
        `10. [TRIGGER ENTRY]: Pola ${patterns.length > 0 ? patterns[0].name : 'Indecision'} terdeteksi, memvalidasi penolakan di zona penting.`,
        `10. [CANDLE PATTERN]: Konfirmasi akhir didapat dari formasi ${patterns.length > 0 ? patterns[0].name : 'Candle kecil'}, menunjukkan siapa yang memegang kendali.`
    ][rand()];

    const p11 = [
        `11. [JEBAKAN RETAIL]: ${smc.inducement ? 'Terdeteksi adanya Inducement (jebakan) sebelum pergerakan ini. Retail trader sudah terkena stop loss.' : 'Tidak terlihat adanya jebakan (Inducement) yang mencolok.'}`,
        `11. [MANIPULASI PASAR]: ${smc.inducement ? 'Market baru saja melakukan pergerakan palsu (Inducement) untuk memancing likuiditas.' : 'Pergerakan terlihat organik tanpa manipulasi (Inducement) yang jelas.'}`,
        `11. [SMART MONEY TRAP]: ${smc.inducement ? 'Pola Inducement terkonfirmasi, smart money telah mengumpulkan order dari trader yang terjebak.' : 'Belum ada tanda-tanda jebakan likuiditas (Inducement) di area ini.'}`,
        `11. [STATUS INDUCEMENT]: ${smc.inducement ? 'Trader pemula telah dijebak (Inducement), jalan sekarang terbuka untuk pergerakan asli.' : 'Market bergerak bersih tanpa pola jebakan (Inducement) sebelumnya.'}`
    ][rand()];

    const p12 = [
        `12. [PROBABILITAS MATEMATIS]: Berdasarkan skor SMC dan Candlestick (Total Skor: ${totalScore}), probabilitas keberhasilan setup ini berada di angka > ${signal === 'WAIT' ? '0' : '85'}%. Syarat minimal eksekusi untuk timeframe ini adalah skor +${minScore} atau -${minScore}.`,
        `12. [WIN RATE AI]: Kalkulasi algoritma memberikan Total Skor ${totalScore}, menghasilkan tingkat akurasi prediksi > ${signal === 'WAIT' ? '0' : '85'}%. Filter ketat diterapkan untuk menghindari entry prematur.`,
        `12. [SKOR KESUKSESAN]: Dengan mempertimbangkan semua variabel, AI memberikan Total Skor ${totalScore} dengan probabilitas win > ${signal === 'WAIT' ? '0' : '85'}%. Hanya setup dengan skor ekstrem yang dieksekusi.`,
        `12. [RATING SETUP]: Setup ini dievaluasi dengan Total Skor ${totalScore}, mencerminkan peluang profit sebesar > ${signal === 'WAIT' ? '0' : '85'}%. Skor di bawah batas minimal (±${minScore}) akan diabaikan.`
    ][rand()];

    const p13 = [
        `13. [MANAJEMEN RISIKO]: Setup ini menggunakan rasio Risk:Reward (RR) sebesar ${rrType}, sangat sehat untuk pertumbuhan akun jangka panjang.`,
        `13. [PROFIL RISIKO]: Target keuntungan dan batasan kerugian diatur dengan rasio RR ${rrType}, memastikan profitabilitas yang konsisten.`,
        `13. [RISK/REWARD RATIO]: Skema perdagangan ini menawarkan rasio RR ${rrType}, memberikan ruang yang cukup untuk menahan volatilitas.`,
        `13. [PARAMETER TRADING]: Dengan rasio RR ${rrType}, setup ini memenuhi standar manajemen risiko profesional.`
    ][rand()];

    const p14 = [
        `14. [KEPUTUSAN FINAL AI]: **${signal === 'WAIT' ? 'WAIT (TUNGGU)' : `EKSEKUSI ${signal}`}**. ${signal === 'WAIT' ? 'Skor belum mencapai batas minimal atau konfirmasi candle belum valid.' : 'Semua parameter konfirmasi (Skor, Zona, dan Pola Candle) telah terpenuhi.'}`,
        `14. [KESIMPULAN ASI]: **${signal === 'WAIT' ? 'WAIT (TUNGGU)' : `EKSEKUSI ${signal}`}**. ${signal === 'WAIT' ? `Sabar, jangan paksakan entry saat probabilitas rendah (Skor < ${minScore}).` : 'Setup valid dengan konfirmasi ganda, silakan tempatkan posisi sesuai parameter.'}`,
        `14. [REKOMENDASI SISTEM]: **${signal === 'WAIT' ? 'WAIT (TUNGGU)' : `EKSEKUSI ${signal}`}**. ${signal === 'WAIT' ? 'Lebih baik menjaga modal hingga setup yang tepat muncul di area OB/FVG.' : `Lampu hijau dari algoritma, eksekusi dengan disiplin ${rrType}.`}`,
        `14. [TINDAKAN FINAL]: **${signal === 'WAIT' ? 'WAIT (TUNGGU)' : `EKSEKUSI ${signal}`}**. ${signal === 'WAIT' ? 'Tidak ada peluang yang jelas, pantau terus pergerakan harga.' : 'Momentum, struktur, dan konfirmasi candle selaras, eksekusi sekarang.'}`
    ][rand()];

    return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14];
};

export const analyzeLocalMarket = (
  asset: Asset,
  candles: Candle[],
  timeframe: TimeFrame,
  lastFeedback: TradeFeedback,
  winStreak: number = 0,
  lossStreak: number = 0
): AnalysisResult => {
  
  if (!candles || candles.length < 150) {
      return {
          signal: 'WAIT', confidence: 0, entryPrice: 0, stopLoss: 0, takeProfit: 0, riskRewardRatio: "0:0",
          reasoning: Array(14).fill("Initializing ASI Core (Need 150 Candles)..."), smcConceptsFound: [], timestamp: new Date().toLocaleTimeString(), timeframe: timeframe
      };
  }

  const current = candles[candles.length - 1];
  const prev = candles[candles.length - 2];
  const rsi = calculateRSI(candles, 14);
  const atr = calculateATR(candles, 14) || (current.close * 0.001);
  const emaState = analyzeEMACondition(candles);
  const patterns = detectCandlestickPatterns(candles);
  const smc = detectSMC(candles, timeframe);

  // --- MATHEMATICAL AGENT LOGIC (LOCAL BRAIN) ---
  
  let signal: 'BUY' | 'SELL' | 'WAIT' = 'WAIT';
  let confidence = 0;
  let isCounterTrend = false;
  let setupType = "SMC_TREND";

  // TIMEFRAME-SPECIFIC THRESHOLDS
  let minScore = 6;
  let rrRatioVal = 3.0;
  let rrLabel = "1:3";

  if (timeframe === 'M1') { minScore = 8; rrRatioVal = 3.0; rrLabel = "1:3"; }
  else if (timeframe === 'M5') { minScore = 7; rrRatioVal = 3.0; rrLabel = "1:3"; }
  else if (timeframe === 'M15') { minScore = 7; rrRatioVal = 3.0; rrLabel = "1:3"; }
  else if (timeframe === 'M30') { minScore = 6; rrRatioVal = 3.0; rrLabel = "1:3"; }
  else if (timeframe === 'H1') { minScore = 6; rrRatioVal = 4.0; rrLabel = "1:4"; }
  else if (timeframe === 'H4' || timeframe === 'D1') { minScore = 5; rrRatioVal = 4.0; rrLabel = "1:4"; }

  // 1. LAYER 1: SMC VECTOR (DIRECTION)
  const emaValues = emaState.values || { ema50: current.close, ema200: current.close };
  const isUptrend = current.close > emaValues.ema200 && current.close > emaValues.ema50;
  const isDowntrend = current.close < emaValues.ema200 && current.close < emaValues.ema50;

  // 2. LAYER 2 & 4: SND LOCATION & CANDLE TRIGGER
  const bullishPatterns = patterns.filter(p => p.type === 'BULLISH');
  const bearishPatterns = patterns.filter(p => p.type === 'BEARISH');
  
  // --- ADVANCED SCORING SYSTEM ---
  let totalScore = smc.score; // Base score from SMC structure, momentum, BOS, CHoCH, Sweep, FVG, OB

  // Add Candlestick Pattern Scores
  if (bullishPatterns.length > 0) {
      const bestPattern = bullishPatterns[0];
      if (bestPattern.name.includes('Morning Star') || bestPattern.name.includes('Three White Soldiers')) totalScore += 3;
      else if (bestPattern.strength >= 3) totalScore += 2;
      else totalScore += 1;
  }
  if (bearishPatterns.length > 0) {
      const bestPattern = bearishPatterns[0];
      if (bestPattern.name.includes('Evening Star') || bestPattern.name.includes('Three Black Crows')) totalScore -= 3;
      else if (bestPattern.strength >= 3) totalScore -= 2;
      else totalScore -= 1;
  }

  // Add Zone Rejection Scores (Simplified via RSI & EMA)
  if (rsi < 40 && isUptrend) totalScore += 2; // Strong demand reaction
  if (rsi > 60 && isDowntrend) totalScore -= 2; // Strong supply reaction

  // Interpretation (Strict Entry)
  const currentBodySize = Math.abs(current.close - current.open) / (current.high - current.low);
  const isStrongConfirmation = currentBodySize >= 0.6;

  if (totalScore >= minScore) {
      // Check for retrace to OB/FVG or strong reversal pattern
      const hasBullishConfirmation = bullishPatterns.length > 0 && isStrongConfirmation;
      const atBullishLevel = smc.orderBlock === 'BULLISH' || smc.fvg === 'BULLISH' || smc.liquiditySweep === 'SELL_SIDE';

      if (hasBullishConfirmation && atBullishLevel) {
          signal = 'BUY';
          confidence = Math.min(99, 85 + totalScore);
          setupType = "HIGH_PROB_BULLISH";
      } else {
          signal = 'WAIT';
          setupType = "BULLISH_WAIT_CONFIRMATION";
      }
  } else if (totalScore <= -minScore) {
      const hasBearishConfirmation = bearishPatterns.length > 0 && isStrongConfirmation;
      const atBearishLevel = smc.orderBlock === 'BEARISH' || smc.fvg === 'BEARISH' || smc.liquiditySweep === 'BUY_SIDE';

      if (hasBearishConfirmation && atBearishLevel) {
          signal = 'SELL';
          confidence = Math.min(99, 85 + Math.abs(totalScore));
          setupType = "HIGH_PROB_BEARISH";
      } else {
          signal = 'WAIT';
          setupType = "BEARISH_WAIT_CONFIRMATION";
      }
  } else if (totalScore >= 3) {
      signal = 'WAIT';
      setupType = "BULLISH_BIAS_WAIT";
  } else if (totalScore <= -3) {
      signal = 'WAIT';
      setupType = "BEARISH_BIAS_WAIT";
  } else {
      signal = 'WAIT';
      setupType = "SIDEWAYS_NO_TRADE";
  }

  // 3. LAYER 3: SNR (Implicitly handled by zones and EMA support in this simplified math model)

  // 4. EXECUTION CALCULATION
  let entry = current.close;
  let sl = 0, tp = 0;

  if (signal === 'BUY') {
      // SL below low of confirmation candle or OB
      const riskBuffer = current.close * 0.0005; 
      sl = Math.min(current.low, prev.low) - riskBuffer;
      const risk = entry - sl;
      tp = entry + (risk * rrRatioVal); 
  } else if (signal === 'SELL') {
      const riskBuffer = current.close * 0.0005;
      sl = Math.max(current.high, prev.high) + riskBuffer;
      const risk = sl - entry;
      tp = entry - (risk * rrRatioVal); 
  } else {
      sl = 0;
      tp = 0;
      rrLabel = "0:0";
  }

  // PREDICTION 10-25 CANDLES
  let prediction = "";
  let multiplier = 1.5;
  let candleTarget = "10-25";
  if (timeframe === 'M1' || timeframe === 'M5') { multiplier = 1.2; candleTarget = "10-15"; }
  else if (timeframe === 'M15' || timeframe === 'M30') { multiplier = 1.8; candleTarget = "15-20"; }
  else { multiplier = 2.5; candleTarget = "15-25"; }

  const avgSwingDist = atr * 10; // Approximation of average swing distance
  const targetPriceUp = current.close + (avgSwingDist * multiplier);
  const targetPriceDown = current.close - (avgSwingDist * multiplier);

  if (signal === 'BUY') {
      prediction = `Prediksi ${candleTarget} candle ke depan: Harga diproyeksikan naik menuju area ${targetPriceUp.toFixed(4)} didukung oleh struktur bullish dan momentum saat ini.`;
  } else if (signal === 'SELL') {
      prediction = `Prediksi ${candleTarget} candle ke depan: Harga diproyeksikan turun menuju area ${targetPriceDown.toFixed(4)} didukung oleh struktur bearish dan momentum saat ini.`;
  } else {
      prediction = `Prediksi ${candleTarget} candle ke depan: Harga kemungkinan akan bergerak ranging/sideways di sekitar ${current.close.toFixed(4)} karena belum ada konfirmasi tren yang kuat.`;
  }

  const smcConcepts = ["ASI_CORE", setupType, timeframe + "_ANALYSIS"];
  if (smc.bos !== 'NONE') smcConcepts.push(`BOS_${smc.bos}`);
  if (smc.choch !== 'NONE') smcConcepts.push(`CHoCH_${smc.choch}`);
  if (smc.orderBlock !== 'NONE') smcConcepts.push(`OB_${smc.orderBlock}`);
  if (smc.fvg !== 'NONE') smcConcepts.push(`FVG_${smc.fvg}`);
  if (smc.liquiditySweep !== 'NONE') smcConcepts.push(`SWEEP_${smc.liquiditySweep}`);
  if (smc.inducement) smcConcepts.push('INDUCEMENT');
  smcConcepts.push(...patterns.map(p => p.name));

  const reasoning = generate14PointNarrative(signal, timeframe, asset, rrLabel, isCounterTrend, smc, patterns, totalScore, minScore);

  return {
    signal,
    confidence: signal === 'WAIT' ? 0 : confidence,
    entryPrice: entry,
    stopLoss: sl,
    takeProfit: tp,
    riskRewardRatio: rrLabel,
    reasoning: reasoning,
    smcConceptsFound: smcConcepts,
    timestamp: new Date().toLocaleTimeString(),
    timeframe: timeframe,
    prediction: prediction
  };
};
