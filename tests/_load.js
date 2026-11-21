/* Shared test loader: runs the whole app (every <script> in index.html, in order) inside Node with
 * minimal browser stubs, without mounting it. Returns the global MLA namespace. */
'use strict';
const path = require('path');
const fs = require('fs');
const vm = require('vm');
const root = path.join(__dirname, '..');

function decodeEntities(s) {
  return String(s).replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (m, e) => {
    const named = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", times: '×', nbsp: ' ', hellip: '…', mdash: '—', rarr: '→', larr: '←' };
    if (e[0] === '#') { const n = e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10); return isNaN(n) ? m : String.fromCodePoint(n); }
    return named[e.toLowerCase()] != null ? named[e.toLowerCase()] : m;
  });
}
function el() {
  let html = '';
  const node = { style: {}, classList: { add() {}, remove() {}, toggle() {} }, setAttribute() {}, getAttribute() { return null; }, removeAttribute() {}, hasAttribute() { return false }, appendChild() {}, removeChild() {}, remove() {}, addEventListener() {}, removeEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; }, click() {}, focus() {}, select() {}, scrollIntoView() {}, getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 700 }; }, children: [], value: '', nodeType: 1 };
  Object.defineProperty(node, 'innerHTML', { get() { return html; }, set(v) { html = v; } });
  Object.defineProperty(node, 'textContent', { get() { return decodeEntities(html.replace(/<[^>]*>/g, '')); }, set(v) { html = v; } });
  Object.defineProperty(node, 'children', { get() { const m = html.match(/foo="([^"]*)"/); return [{ getAttribute(n) { return n === 'foo' && m ? decodeEntities(m[1]) : null; } }]; } });
  return node;
}

function load(opts) {
  opts = opts || {};
  if (global.MLA && global.MLA.__loaded) return global.MLA;
  const store = {};
  const listeners = {};
  global.window = global;
  global.document = { documentElement: el(), createElement: el, getElementById() { return null; }, querySelector() { return null; }, querySelectorAll() { return []; }, addEventListener(t, f) { (listeners[t] = listeners[t] || []).push(f); }, removeEventListener() {}, body: el(), readyState: 'loading', visibilityState: 'visible' };
  global.location = { hash: '#/home', protocol: 'http:' };
  global.history = { length: 1, back() {} };
  global.localStorage = { getItem(k) { return store[k] || null; }, setItem(k, v) { store[k] = String(v); }, removeItem(k) { delete store[k]; } };
  try { Object.defineProperty(global, 'navigator', { value: { serviceWorker: null, platform: 'Node', userAgent: 'node' }, configurable: true }); } catch (e) { /* ignore */ }
  global.addEventListener = function (t, f) { (listeners[t] = listeners[t] || []).push(f); };
  global.removeEventListener = function () {};
  global.matchMedia = function () { return { matches: false }; };
  global.requestAnimationFrame = function () { return 0; };
  global.scrollTo = function () {};
  global.getComputedStyle = function () { return { getPropertyValue() { return ''; } }; };
  if (opts.vue !== false) {
    vm.runInThisContext(fs.readFileSync(path.join(root, 'vendor/vue.global.prod.js'), 'utf8'), { filename: 'vue.global.prod.js' });
    if (!global.Vue || !global.Vue.createApp) throw new Error('Vue did not load');
  }
  const index = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  // The page loads js/data/catalog.js and fetches lesson files on demand; Node loads the lesson files themselves, in
  // the catalog's order (tools/content-files.json), and records which file each lesson came from.
  const lessonFiles = JSON.parse(fs.readFileSync(path.join(root, 'tools/content-files.json'), 'utf8')).map(f => 'js/data/' + f);
  const scripts = [...index.matchAll(/<script src="([^"]+)"><\/script>/g)].map(m => m[1]).filter(s => !s.includes('vendor/vue'))
    .flatMap(s => s === 'js/data/catalog.js' ? lessonFiles : [s]);
  const errors = [];
  for (const s of scripts) {
    const file = path.join(root, s);
    if (!fs.existsSync(file)) { errors.push('missing script ' + s); continue; }
    const before = global.MLA && global.MLA.data && global.MLA.data.concepts ? global.MLA.data.concepts.length : 0;
    try { require(file); } catch (e) { errors.push(s + ': ' + (e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e)); }
    if (lessonFiles.includes(s)) for (const c of global.MLA.data.concepts.slice(before)) c.file = path.basename(s);
  }
  const MLA = global.MLA;
  MLA.__loaded = true;
  MLA.__loadErrors = errors;
  MLA.__scripts = scripts;
  return MLA;
}

// Tiny test helper
function suite(name) {
  let pass = 0, fail = 0;
  const failures = [];
  return {
    ok(label, cond, detail) { if (cond) pass++; else { fail++; failures.push(label + (detail ? ' — ' + detail : '')); } },
    done() {
      for (const f of failures) console.log('  FAIL ' + f);
      console.log(name + ': ' + pass + ' passed, ' + fail + ' failed');
      process.exit(fail ? 1 : 0);
    },
    get failed() { return fail; }
  };
}

module.exports = { load, suite, root };
