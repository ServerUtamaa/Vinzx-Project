
/**
 * SECURITY PROTOCOL ACTIVE
 * 
 * MARKET LOGIC ENGINE - UNAUTHORIZED MODIFICATION PROHIBITED.
 * REQUIRED KEY: 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a
 */

import { Asset } from "../types";

/**
 * Checks if the market for a specific asset is currently open.
 * Crypto is 24/7.
 * Forex & Commodities are closed from Friday 22:00 UTC to Monday 00:00 UTC.
 * (Saturday 05:00 WIB to Monday 07:00 WIB)
 */
export const isMarketOpen = (asset: Asset): boolean => {
    const cryptoAssets = [Asset.BTCUSD, Asset.ETHUSD, Asset.SOLUSD, Asset.BNBUSD];
    if (cryptoAssets.includes(asset)) return true;

    const now = new Date();
    const day = now.getUTCDay(); // 0 = Sunday, 5 = Friday, 6 = Saturday
    const hour = now.getUTCHours();

    // Friday after 22:00 UTC (Saturday 05:00 WIB)
    if (day === 5 && hour >= 22) return false;
    
    // Saturday all day
    if (day === 6) return false;
    
    // Sunday all day
    if (day === 0) return false;

    return true;
};

export const getMarketStatusMessage = (asset: Asset): string => {
    if (isMarketOpen(asset)) return "MARKET OPEN";
    return "MARKET CLOSED (WEEKEND)";
};
