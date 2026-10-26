// Resolve from this module so all requests retain the reverse-proxy prefix.
export const serviceBase = new URL('../', import.meta.url);
export function serviceURL(path) { return new URL(path, serviceBase); }
export function websocketURL() {
  const url = serviceURL('translate/ws');
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url;
}
