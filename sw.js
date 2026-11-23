/* ML Atlas – service worker.
 * App files (same origin): network first, falling back to the cache — new GitHub Pages deploys
 * show up immediately when online, and the app still opens offline.
 * Cross-origin requests (the Pyodide runtime on cdn.jsdelivr.net) are deliberately NOT intercepted:
 * piping its 10 MB WebAssembly file through cache.put(response.clone()) can stall the stream and
 * deadlock the cache. jsDelivr serves those versioned files as immutable for a year, so the browser's
 * own HTTP cache keeps Python available offline after the first load. */
// Old versions of this cache are deleted on update, but only this app's own: other sites on the same domain (for
// example another course at you.github.io/other-course) share the browser's cache storage and keep theirs.
const CACHE_PREFIX = 'ml-atlas-app-';
const APP_CACHE = CACHE_PREFIX + 'v2';
const CORE = ['./', './index.html', './css/app.css', './vendor/vue.global.prod.js', './manifest.webmanifest', './icons/icon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(APP_CACHE).then(c => c.addAll(CORE)).catch(() => {}).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== APP_CACHE && k.startsWith(CACHE_PREFIX)).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  if (new URL(req.url).origin !== self.location.origin) return; // let the browser handle CDN requests
  e.respondWith(
    fetch(req).then(res => {
      if (res && res.ok && res.type === 'basic') { const copy = res.clone(); caches.open(APP_CACHE).then(c => c.put(req, copy)).catch(() => {}); }
      return res;
    }).catch(() => caches.match(req).then(hit => hit || (req.mode === 'navigate' ? caches.match('./index.html') : undefined)))
  );
});
