/* serve.js must only serve files inside its folder: no traversal (including into a sibling folder whose name starts
 * the same way), no hidden files, no node_modules, only GET/HEAD. Run: node tests/serve.test.js */
'use strict';
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

let passed = 0, failed = 0;
const ok = (name, cond, extra) => { if (cond) passed++; else { failed++; console.log('  FAIL ' + name + (extra ? ' — ' + extra : '')); } };

const base = fs.mkdtempSync(path.join(os.tmpdir(), 'mla-serve-'));
const site = path.join(base, 'site'), sibling = path.join(base, 'site-secret');
for (const d of [site, sibling, path.join(site, '.hidden'), path.join(site, 'node_modules'), path.join(site, 'sub')]) fs.mkdirSync(d, { recursive: true });
fs.writeFileSync(path.join(site, 'index.html'), '<!doctype html><title>ok</title>');
fs.writeFileSync(path.join(site, 'sub', 'a.js'), 'console.log(1)');
fs.writeFileSync(path.join(site, '.hidden', 'secret.txt'), 'hidden');
fs.writeFileSync(path.join(site, 'node_modules', 'pkg.js'), 'pkg');
fs.writeFileSync(path.join(sibling, 'secret.txt'), 'sibling secret');

const port = 40000 + Math.floor(Math.random() * 20000);
const server = spawn(process.execPath, [path.join(__dirname, '..', 'serve.js'), String(port), site], { stdio: 'ignore' });

function request(method, rawPath) {
  return new Promise(resolve => {
    const req = http.request({ host: '127.0.0.1', port, method, path: rawPath }, res => {
      let body = ''; res.on('data', c => { body += c; }); res.on('end', () => resolve({ status: res.statusCode, body, headers: res.headers }));
    });
    req.on('error', e => resolve({ status: 0, body: String(e) }));
    req.end();
  });
}
const wait = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  let up = false;
  for (let i = 0; i < 50 && !up; i++) { const r = await request('GET', '/'); up = r.status === 200; if (!up) await wait(100); }
  ok('server starts and serves index.html', up);
  const home = await request('GET', '/');
  ok('index content', home.body.includes('<title>ok</title>'));
  ok('nosniff header', home.headers && home.headers['x-content-type-options'] === 'nosniff');
  ok('serves nested files', (await request('GET', '/sub/a.js')).status === 200);
  for (const p of ['/../site-secret/secret.txt', '/..%2fsite-secret%2fsecret.txt', '/%2e%2e/site-secret/secret.txt', '/..%5csite-secret%5csecret.txt',
                   '/sub/../../site-secret/secret.txt', '/.hidden/secret.txt', '/%2ehidden/secret.txt', '/node_modules/pkg.js', '/sub', '/a%00.js']) {
    const r = await request('GET', p);
    ok('blocked: ' + p, r.status === 404 || r.status === 400, 'status ' + r.status);
    ok('no leak: ' + p, !/secret|hidden|pkg/.test(r.body), r.body.slice(0, 60));
  }
  ok('HEAD allowed', (await request('HEAD', '/')).status === 200);
  ok('POST refused', (await request('POST', '/')).status === 405);
  server.kill();
  fs.rmSync(base, { recursive: true, force: true });
  console.log('serve: ' + passed + ' passed, ' + failed + ' failed');
  process.exit(failed ? 1 : 0);
})();
