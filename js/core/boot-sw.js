/* Offline support and install-as-app, only over http/https (GitHub Pages or `node serve.js`). */
(function () {
  if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
    window.addEventListener('load', function () { navigator.serviceWorker.register('sw.js').catch(function () {}); });
  }
})();
