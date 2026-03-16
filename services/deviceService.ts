
/**
 * Device Identification Service
 * Generates and persists a unique identifier for the browser/device.
 */

const DEVICE_ID_KEY = 'NVR_DEVICE_FINGERPRINT';

/**
 * Generates a simple fingerprint based on browser properties.
 * In a real app, you'd use a library like FingerprintJS.
 */
const generateFingerprint = (): string => {
  const navigator_info = window.navigator;
  const screen_info = window.screen;
  let uid = navigator_info.mimeTypes.length.toString();
  uid += navigator_info.userAgent.replace(/\D+/g, '');
  uid += navigator_info.plugins.length.toString();
  uid += screen_info.height || '';
  uid += screen_info.width || '';
  uid += screen_info.pixelDepth || '';
  
  // Simple hash function
  let hash = 0;
  for (let i = 0; i < uid.length; i++) {
    const char = uid.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return 'DEV-' + Math.abs(hash).toString(16).toUpperCase() + '-' + Math.random().toString(36).substr(2, 9).toUpperCase();
};

/**
 * Gets the device ID from storage or generates a new one.
 */
export const getDeviceId = (): string => {
  let deviceId = localStorage.getItem(DEVICE_ID_KEY);
  
  if (!deviceId) {
    // Try to get from cookie as fallback
    const cookies = document.cookie.split(';');
    const deviceCookie = cookies.find(c => c.trim().startsWith(DEVICE_ID_KEY + '='));
    if (deviceCookie) {
      deviceId = deviceCookie.split('=')[1];
    }
  }

  if (!deviceId) {
    deviceId = generateFingerprint();
    // Save to multiple places for persistence
    localStorage.setItem(DEVICE_ID_KEY, deviceId);
    // Set cookie for 10 years
    const expires = new Date();
    expires.setFullYear(expires.getFullYear() + 10);
    document.cookie = `${DEVICE_ID_KEY}=${deviceId};expires=${expires.toUTCString()};path=/;SameSite=Strict`;
  }

  return deviceId;
};
