#!/usr/bin/env node
/* Tiny zero-dependency static server for local use.
 * Usage: node serve.js [port]   then open http://localhost:<port>
 * (GitHub Pages serves the same files in production; opening index.html directly also works,
 * but a server enables the offline service worker and the install-as-app option.) */
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = process.argv[3] ? path.resolve(process.argv[3]) : __dirname;   // node serve.js [port] [folder]
const port = Number(process.argv[2]) || 8080;
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.md': 'text/markdown; charset=utf-8', '.py': 'text/x-python; charset=utf-8'
};
// Only files inside the app folder are served. A plain prefix check would also accept a sibling folder whose name
// starts the same way ("LLM-learning-old"), so containment is tested with path.relative. Hidden files and folders
// (.git, .claude, .env …) and installed packages are never served: the server can be reached from the local network.
function resolveInside(urlPath) {
  const fp = path.resolve(root, '.' + path.posix.normalize('/' + urlPath));
  const rel = path.relative(root, fp);
  if (rel === '' || rel.startsWith('..') || path.isAbsolute(rel)) return null;
  const parts = rel.split(path.sep);
  if (parts.some(s => s.startsWith('.')) || parts[0] === 'node_modules') return null;
  return fp;
}
const HEADERS = { 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'SAMEORIGIN' };

const server = http.createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405, { Allow: 'GET, HEAD' }); res.end(); return; }
  let p;
  try { p = decodeURIComponent(req.url.split('?')[0]); } catch (e) { res.writeHead(400); res.end('Bad request'); return; }
  if (p.includes('\0')) { res.writeHead(400); res.end('Bad request'); return; }
  if (p === '/' || p === '') p = '/index.html';
  const fp = resolveInside(p);
  let ok = false;
  try { ok = !!fp && fs.statSync(fp).isFile(); } catch (e) { ok = false; }
  if (!ok) { res.writeHead(404, Object.assign({ 'Content-Type': 'text/plain; charset=utf-8' }, HEADERS)); res.end('Not found'); return; }
  res.writeHead(200, Object.assign({ 'Content-Type': TYPES[path.extname(fp).toLowerCase()] || 'application/octet-stream' }, HEADERS));
  if (req.method === 'HEAD') { res.end(); return; }
  fs.createReadStream(fp).pipe(res);
});

server.listen(port, () => {
  console.log('\n  The course is running.\n');
  console.log('  On this computer:   http://localhost:' + port);
  try {
    const nets = require('os').networkInterfaces();
    for (const name of Object.keys(nets)) for (const ni of nets[name]) {
      if (ni.family === 'IPv4' && !ni.internal) console.log('  On your network:    http://' + ni.address + ':' + port + '   (open on a phone or tablet on the same Wi-Fi)');
    }
  } catch (e) { /* ignore */ }
  console.log('\n  Press Ctrl+C to stop.\n');
});
server.on('error', e => {
  if (e.code === 'EADDRINUSE') { console.error('Port ' + port + ' is in use. Try:  node serve.js ' + (port + 1)); process.exit(1); }
  throw e;
});
