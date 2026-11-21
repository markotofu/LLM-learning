/* ML Atlas – datasets that code in the in-browser Python can open as data/<name>, for example
 * pd.read_csv("data/bike_rentals_hourly.csv") or pd.read_excel("data/sales.xlsx"). Before code runs,
 * js/engines/python.js writes every dataset the code mentions into Python's file system.
 * Two kinds:
 *   - bundled practice datasets in data/, synthetic (tools/make-datasets.py mimics the real datasets of the projects);
 *   - files the learner uploads (CSV, TSV, Excel, JSON, Parquet, text). Their contents live in this browser's
 *     IndexedDB (localStorage is far too small); their names, sizes and dates live in state.datasets, so they show
 *     in the progress. Exports include the contents (MLA.store.exportFull), and imports restore them.
 *   MLA.datasets.list() -> [{ name, title, desc, rows, size, type, bundled, updatedAt }]
 *   MLA.datasets.find(name), read(name) -> Promise<Uint8Array>, version(name), referenced(code) -> [names]
 *   MLA.datasets.add(file, { replace }) -> Promise<meta>, remove(name), download(name), usage() -> bytes uploaded
 *   MLA.datasets.exportAll() -> Promise<[{ name, type, updatedAt, data: base64 }]>, importAll(list) -> Promise<count>
 *   MLA.datasets.readCode(name) -> a line of pandas that loads it */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  const BUNDLED = [
    { name: 'bike_rentals_hourly.csv', title: 'Bike rentals, hourly (synthetic)', rows: 2184, lesson: 'time-series',
      desc: 'Thirteen weeks of hourly bike rentals with daily and weekly cycles, weather effects, rainy days, two holidays and a slow upward trend. Columns: datetime, hour, weekday (0 is Monday), holiday, temp_c, humidity, rentals.' },
    { name: 'telco_churn_synthetic.csv', title: 'Telecom customer churn (synthetic)', rows: 2000, lesson: 'ensembles',
      desc: 'Two thousand customers of a phone and internet company and whether they left. Columns: customer_id, tenure_months, contract, monthly_charges, total_charges (blank for brand-new customers), internet, support_calls, paperless_billing, churned.' },
    { name: 'card_transactions_synthetic.csv', title: 'Card payments with rare fraud (synthetic)', rows: 5000, lesson: 'anomaly-detection',
      desc: 'Five thousand card payments over 14 days, about 0.9% of them fraudulent. Columns: transaction_id, day, hour, merchant_category, amount, distance_from_home_km, online, new_merchant, fraud.' }
  ].map(d => Object.assign(d, { bundled: true, type: 'csv' }));

  const MAX_FILE = 20 * 1024 * 1024, MAX_TOTAL = 60 * 1024 * 1024;
  const KINDS = ['csv', 'tsv', 'txt', 'json', 'jsonl', 'xlsx', 'xls', 'parquet'];
  const state = () => (MLA.store && MLA.store.state) || { datasets: {} };
  const metas = () => { const s = state(); if (!s.datasets || typeof s.datasets !== 'object') s.datasets = {}; return s.datasets; };
  const extOf = name => (String(name).toLowerCase().match(/\.([a-z0-9]+)$/) || [])[1] || '';
  // names become paths in Python: keep letters, digits, dots, dashes, underscores and brackets
  const safeName = name => String(name || '').trim().replace(/\s+/g, '_').replace(/[^A-Za-z0-9_.()-]/g, '').replace(/^\.+/, '').slice(0, 80);

  // ---- storage for the file contents: IndexedDB, or memory where it is missing (Node tests, some private modes)
  const memory = new Map();
  let dbp = null;
  function db() {
    if (!global.indexedDB) return Promise.resolve(null);
    if (!dbp) {
      dbp = new Promise(resolve => {
        try {
          const r = global.indexedDB.open('ml-atlas-datasets', 1);
          r.onupgradeneeded = () => r.result.createObjectStore('files', { keyPath: 'name' });
          r.onsuccess = () => resolve(r.result);
          r.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
      });
    }
    return dbp;
  }
  function request(mode, fn) {
    return db().then(d => {
      if (!d) return fn(null);
      return new Promise((resolve, reject) => {
        const t = d.transaction('files', mode), req = fn(t.objectStore('files'));
        t.oncomplete = () => resolve(req && 'result' in req ? req.result : undefined);
        t.onerror = () => reject(t.error || new Error('Browser storage refused the file.'));
        t.onabort = () => reject(t.error || new Error('Browser storage is full: delete a dataset or free some space.'));
      });
    });
  }
  const putBlob = (name, bytes, updatedAt) => request('readwrite', s => s ? s.put({ name, data: bytes, updatedAt }) : (memory.set(name, bytes), null));
  const getBlob = name => request('readonly', s => s ? s.get(name) : null).then(r => r === undefined || r === null ? (memory.get(name) ? { data: memory.get(name) } : null) : r);
  const delBlob = name => request('readwrite', s => s ? s.delete(name) : (memory.delete(name), null));

  function uploaded() { return Object.values(metas()).map(m => Object.assign({ bundled: false, title: m.name }, m)); }
  const all = () => BUNDLED.concat(uploaded());
  function find(name) { return all().find(d => d.name === name) || null; }
  function version(name) { const d = find(name); return d ? (d.bundled ? 'bundled' : String(d.updatedAt || '') + ':' + (d.size || 0)) : ''; }
  function usage() { return uploaded().reduce((n, d) => n + (d.size || 0), 0); }

  const cache = {};
  function read(name) {
    const d = find(name);
    if (!d) return Promise.reject(new Error('No dataset named ' + name));
    if (!d.bundled) return getBlob(name).then(r => { if (!r) throw new Error(name + ' is listed but its contents are not in this browser any more. Upload it again.'); return new Uint8Array(r.data); });
    if (!cache[name]) {
      cache[name] = global.fetch('data/' + name).then(r => { if (!r.ok) throw new Error('Could not download ' + name); return r.arrayBuffer(); })
        .then(b => new Uint8Array(b), e => { delete cache[name]; throw e; });
    }
    return cache[name];
  }
  function referenced(code) {
    const names = new Set(), re = /data\/([A-Za-z0-9_.()-]+)/g;
    let m;
    while ((m = re.exec(String(code || '')))) names.add(m[1]);
    return [...names].filter(n => find(n));
  }

  // A quick look at a text table: how many rows and which columns (for the list; Python does the real parsing).
  function peek(bytes, kind) {
    if (kind !== 'csv' && kind !== 'tsv') return {};
    const text = new TextDecoder('utf-8').decode(bytes.subarray(0, Math.min(bytes.length, 2 * 1024 * 1024)));
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    const sep = kind === 'tsv' ? '\t' : ',';
    const rows = bytes.length <= 2 * 1024 * 1024 ? Math.max(0, lines.length - 1) : null;
    return { rows, columns: lines.length ? lines[0].split(sep).length : 0 };
  }

  async function add(file, opts) {
    opts = opts || {};
    const name = safeName(file && file.name), kind = extOf(name);
    if (!name || !KINDS.includes(kind)) throw new Error('Upload a CSV, TSV, Excel (.xlsx, .xls), JSON, JSON Lines, Parquet or text file.');
    if (BUNDLED.some(d => d.name === name)) throw new Error(name + ' is the name of a built-in dataset: rename your file first.');
    if (file.size > MAX_FILE) throw new Error(name + ' is ' + MLA.util.formatBytes(file.size) + '; files up to ' + MLA.util.formatBytes(MAX_FILE) + ' can be uploaded.');
    const existing = metas()[name];
    if (existing && !opts.replace) { const e = new Error(name + ' already exists.'); e.exists = true; throw e; }
    if (usage() - (existing ? existing.size : 0) + file.size > MAX_TOTAL) throw new Error('Uploaded datasets would take more than ' + MLA.util.formatBytes(MAX_TOTAL) + ': delete one first.');
    const bytes = new Uint8Array(await file.arrayBuffer());
    const updatedAt = new Date().toISOString();
    await putBlob(name, bytes, updatedAt);
    const meta = Object.assign({ name, type: kind, size: bytes.length, updatedAt, uploadedAt: existing ? existing.uploadedAt : updatedAt }, peek(bytes, kind));
    metas()[name] = meta;
    return meta;
  }
  async function remove(name) { await delBlob(name); delete metas()[name]; delete cache[name]; }
  function clearAll() {                          // "Erase all progress": the uploaded files go too
    memory.clear();
    return request('readwrite', s => (s ? s.clear() : null)).catch(() => { /* nothing stored */ });
  }
  async function download(name) {
    const bytes = await read(name);
    return MLA.util.download(name, new Blob([bytes]), 'application/octet-stream');
  }
  function readCode(name) {
    const p = '"data/' + name + '"', k = extOf(name);
    if (k === 'tsv') return 'df = pd.read_csv(' + p + ', sep="\\t")';
    if (k === 'xlsx' || k === 'xls') return 'df = pd.read_excel(' + p + ')';
    if (k === 'json') return 'df = pd.read_json(' + p + ')';
    if (k === 'jsonl') return 'df = pd.read_json(' + p + ', lines=True)';
    if (k === 'parquet') return 'df = pd.read_parquet(' + p + ')';
    if (k === 'txt') return 'text = open(' + p + ', encoding="utf-8").read()';
    return 'df = pd.read_csv(' + p + ')';
  }

  // ---- export and import: the contents travel as base64 inside the progress file
  function toBase64(bytes) {
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return global.btoa ? global.btoa(s) : Buffer.from(bytes).toString('base64');
  }
  function fromBase64(b64) {
    if (!global.atob) return new Uint8Array(Buffer.from(b64, 'base64'));
    const s = global.atob(b64), out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }
  async function exportAll() {
    const out = [];
    for (const m of uploaded()) {
      try { const bytes = await read(m.name); out.push({ name: m.name, type: m.type, updatedAt: m.updatedAt, data: toBase64(bytes) }); } catch (e) { /* contents lost: nothing to export */ }
    }
    return out;
  }
  async function importAll(list) {
    let n = 0;
    for (const f of Array.isArray(list) ? list : []) {
      const name = safeName(f && f.name), kind = extOf(name);
      if (!name || !KINDS.includes(kind) || typeof f.data !== 'string' || BUNDLED.some(d => d.name === name)) continue;
      const mine = metas()[name];
      if (mine && (mine.updatedAt || '') >= (f.updatedAt || '')) continue;            // keep the newer copy
      const bytes = fromBase64(f.data);
      if (bytes.length > MAX_FILE) continue;
      const updatedAt = f.updatedAt || new Date().toISOString();
      await putBlob(name, bytes, updatedAt);
      metas()[name] = Object.assign({ name, type: kind, size: bytes.length, updatedAt, uploadedAt: updatedAt }, peek(bytes, kind));
      n++;
    }
    return n;
  }

  MLA.datasets = { list: all, find, read, version, referenced, uploaded, usage, add, remove, clearAll, download, readCode, exportAll, importAll, readUploaded: read, MAX_FILE, MAX_TOTAL, KINDS };
})(typeof globalThis !== 'undefined' ? globalThis : window);
