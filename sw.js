/* Service worker.
 * App files (same origin): network first, falling back to the cache — new GitHub Pages deploys
 * show up immediately when online, and the app still opens offline.
 * Cross-origin requests (the Pyodide runtime on cdn.jsdelivr.net) are deliberately NOT intercepted:
 * piping its 10 MB WebAssembly file through cache.put(response.clone()) can stall the stream and
 * deadlock the cache. jsDelivr serves those versioned files as immutable for a year, so the browser's
 * own HTTP cache keeps Python available offline after the first load. */
// One cache per published address: several courses on one site (you.github.io/one, you.github.io/two) keep separate
// offline copies, and an update deletes only this course's older caches, never another course's.
const SCOPE = self.registration.scope;
const APP_CACHE = 'course-app-v1:' + SCOPE;
const CORE = ['./', './index.html', './css/app.css', './vendor/vue.global.prod.js', './manifest.webmanifest', './icons/icon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(APP_CACHE).then(c => c.addAll(CORE)).catch(() => {}).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== APP_CACHE && k.endsWith(':' + SCOPE)).map(k => caches.delete(k)))).then(() => self.clients.claim()));
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
