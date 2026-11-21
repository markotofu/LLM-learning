/* ML Atlas – in-browser Python (Pyodide in a Web Worker) + LeetCode-style test harness.
 *
 * - The runtime loads lazily from the jsDelivr CDN the first time code is run (~15 MB, then cached
 *   by the browser and the service worker so it also works offline afterwards).
 * - Code runs in a Web Worker: the page never freezes, and a run that exceeds the time limit is
 *   stopped by terminating the worker (the runtime restarts on the next run).
 * - The same HARNESS source is executed by tests/run_problems.py under CPython to verify every
 *   problem's reference solution, so browser and CI grading are identical. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  const reactive = (global.Vue && global.Vue.reactive) ? global.Vue.reactive : (x => x);

  const PYODIDE_VERSION = '0.27.2';
  const INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v' + PYODIDE_VERSION + '/full/';

  // ------------------------------------------------------------------ Python side
  const HARNESS = String.raw`
import sys, io, json, math, time, random, traceback, ast, contextlib
try:
    import numpy as np
except Exception:
    np = None

_MLA_SRC = ''
_MLA_CAP = 20000

def _mla_setup_printing():
    if np is None:
        return
    try:
        np.set_printoptions(legacy='1.25')
    except Exception:
        pass
    np.set_printoptions(precision=4, suppress=True, linewidth=88, threshold=60, edgeitems=3)

_mla_setup_printing()

def _mla_cap(s):
    if not s:
        return ''
    if len(s) > _MLA_CAP:
        return s[:_MLA_CAP] + '\n... (output truncated)'
    return s

def _mla_plain(v):
    if np is not None and isinstance(v, np.generic):
        try:
            return v.item()
        except Exception:
            return v
    return v

def _mla_repr(v, limit=700):
    try:
        s = repr(_mla_plain(v))
    except Exception:
        s = '<unprintable ' + type(v).__name__ + '>'
    if len(s) > limit:
        s = s[:limit] + ' ...'
    return s

def _mla_is_bool(x):
    return isinstance(x, bool) or (np is not None and isinstance(x, np.bool_))

def _mla_is_num(x):
    if _mla_is_bool(x):
        return False
    if isinstance(x, (int, float)):
        return True
    return np is not None and isinstance(x, np.number)

def _mla_equal(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    if np is not None and (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        try:
            A = np.asarray(a)
            B = np.asarray(b)
        except Exception:
            return False
        if A.shape != B.shape:
            return False
        try:
            if A.dtype.kind in 'biuf' and B.dtype.kind in 'biuf':
                if A.dtype.kind == 'b' or B.dtype.kind == 'b':
                    return bool(np.array_equal(A, B))
                return bool(np.allclose(A.astype(float), B.astype(float), rtol=tol, atol=tol, equal_nan=True))
            if A.dtype.kind == 'O' or B.dtype.kind == 'O':
                return all(_mla_equal(x, y, tol) for x, y in zip(A.ravel().tolist(), B.ravel().tolist()))
            return bool(np.array_equal(A, B))
        except Exception:
            return False
    if _mla_is_bool(a) or _mla_is_bool(b):
        return _mla_is_bool(a) and _mla_is_bool(b) and bool(a) == bool(b)
    if _mla_is_num(a) and _mla_is_num(b):
        try:
            fa, fb = float(a), float(b)
        except Exception:
            return a == b
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)
    if isinstance(b, (list, tuple)) and isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_mla_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(b, dict) and isinstance(a, dict):
        return set(a.keys()) == set(b.keys()) and all(_mla_equal(a[k], b[k], tol) for k in b)
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    try:
        return bool(a == b)
    except Exception:
        return False

def _mla_hint(got, exp):
    if got is None and exp is not None:
        return 'Your function returned None. Did you forget a return statement?'
    arrays = np is not None and (isinstance(got, np.ndarray) or isinstance(exp, np.ndarray))
    if arrays:
        try:
            gs, es = np.shape(got), np.shape(exp)
            if gs != es:
                return 'Shape mismatch: yours is ' + str(gs) + ', expected ' + str(es) + '.'
        except Exception:
            pass
        return 'Same shape, but some values differ.'
    if _mla_is_num(exp) and isinstance(got, (list, tuple)):
        return 'Expected a single number but got a ' + type(got).__name__ + '.'
    if isinstance(exp, (list, tuple)) and _mla_is_num(got):
        return 'Expected a ' + type(exp).__name__ + ' but got a single number.'
    if isinstance(exp, (list, tuple)) and isinstance(got, (list, tuple)) and len(exp) != len(got):
        return 'Length mismatch: yours has ' + str(len(got)) + ' items, expected ' + str(len(exp)) + '.'
    if isinstance(exp, dict) and isinstance(got, dict) and set(exp) != set(got):
        missing = [k for k in exp if k not in got][:5]
        extra = [k for k in got if k not in exp][:5]
        parts = []
        if missing:
            parts.append('missing keys ' + repr(missing))
        if extra:
            parts.append('unexpected keys ' + repr(extra))
        return 'Dictionary keys differ: ' + ', '.join(parts) + '.'
    if _mla_is_num(got) and _mla_is_num(exp):
        return 'Numbers differ (tolerance is about 1e-6 unless stated).'
    if type(got) is not type(exp) and not (isinstance(got, (list, tuple)) and isinstance(exp, (list, tuple))):
        return 'Type mismatch: yours is ' + type(got).__name__ + ', expected ' + type(exp).__name__ + '.'
    return ''

def _mla_err(e):
    lines = _MLA_SRC.splitlines()
    out = []
    line_no = None
    tb = traceback.extract_tb(e.__traceback__) if e.__traceback__ is not None else []
    for f in [f for f in tb if f.filename == '<your code>'][-6:]:
        code = lines[f.lineno - 1].strip() if f.lineno and 0 < f.lineno <= len(lines) else ''
        out.append('  Line ' + str(f.lineno) + ', in ' + f.name + ((':\n    ' + code) if code else ''))
        line_no = f.lineno
    if isinstance(e, SyntaxError):
        if e.filename in ('<your code>', None):
            line_no = e.lineno
            code = (e.text or '').rstrip('\n').strip()
            if code:
                out.append('  Line ' + str(e.lineno) + ':\n    ' + code)
        msg = type(e).__name__ + ': ' + str(e.msg)
    else:
        msg = ''.join(traceback.format_exception_only(type(e), e)).strip()
    return {'type': type(e).__name__, 'message': msg, 'line': line_no, 'trace': '\n'.join(out)}

def _mla_run_tests(user_code, payload):
    global _MLA_SRC
    P = json.loads(payload)
    _MLA_SRC = user_code
    prelude = P.get('prelude') or ''
    solution = P.get('solution') or ''
    tests = P.get('tests') or []
    res = {'error': None, 'stdout': '', 'tests': [], 'ms': 0}
    t_start = time.perf_counter()
    user_ns = {'__name__': '__main__'}
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            if prelude:
                exec(compile(prelude, '<prelude>', 'exec'), user_ns)
            exec(compile(user_code, '<your code>', 'exec'), user_ns)
    except BaseException as e:
        res['error'] = _mla_err(e)
        res['stdout'] = _mla_cap(out.getvalue())
        return json.dumps(res)
    res['stdout'] = _mla_cap(out.getvalue())
    ref = {}
    def get_ref():
        if 'ns' not in ref:
            ns = {'__name__': '__ref__', 'np': np, 'math': math, 'random': random}
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if prelude:
                        exec(compile(prelude, '<prelude>', 'exec'), ns)
                    if solution:
                        exec(compile(solution, '<reference>', 'exec'), ns)
            except BaseException as e:
                res['refError'] = 'Reference solution failed to load: ' + repr(e)
            ref['ns'] = ns
        return ref['ns']
    for i, t in enumerate(tests):
        r = {'name': t.get('name') or ('Test ' + str(i + 1)), 'hidden': bool(t.get('hidden')), 'passed': False,
             'call': t.get('call'), 'got': None, 'expected': None, 'error': None, 'hint': '', 'stdout': '', 'ms': 0}
        tout = io.StringIO()
        t0 = time.perf_counter()
        try:
            env = dict(user_ns)
            if np is not None:
                env.setdefault('np', np)
            env.setdefault('math', math)
            with contextlib.redirect_stdout(tout), contextlib.redirect_stderr(tout):
                if t.get('setup'):
                    exec(compile(t['setup'], '<test setup>', 'exec'), env)
                if t.get('code'):
                    exec(compile(t['code'], '<test>', 'exec'), env)
                    r['passed'] = True
                else:
                    got = eval(compile(t['call'], '<test>', 'eval'), env)
            if not t.get('code'):
                r['got'] = _mla_repr(got)
                renv = dict(get_ref())
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if t.get('setup'):
                        exec(compile(t['setup'], '<test setup>', 'exec'), renv)
                    expr = t.get('expect') if t.get('expect') not in (None, '') else t['call']
                    expected = eval(compile(expr, '<expected>', 'eval'), renv)
                r['expected'] = _mla_repr(expected)
                ok = _mla_equal(got, expected, float(t.get('tol') or 1e-6))
                r['passed'] = bool(ok)
                if not ok:
                    r['hint'] = _mla_hint(got, expected)
        except BaseException as e:
            r['error'] = _mla_err(e)
            if isinstance(e, AssertionError) and not str(e):
                r['error']['message'] = 'AssertionError: a check in this test failed.'
        r['ms'] = round((time.perf_counter() - t0) * 1000, 1)
        r['stdout'] = _mla_cap(tout.getvalue())
        res['tests'].append(r)
    res['ms'] = round((time.perf_counter() - t_start) * 1000, 1)
    return json.dumps(res)

def _mla_exec(code):
    global _MLA_SRC
    _MLA_SRC = code
    ns = {'__name__': '__main__'}
    res = {'error': None, 'value': None, 'figures': [], 'ms': 0}
    t0 = time.perf_counter()
    try:
        tree = ast.parse(code, '<your code>', 'exec')
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, '<your code>', 'exec'), ns)
        if last is not None:
            v = eval(compile(last, '<your code>', 'eval'), ns)
            if v is not None:
                res['value'] = _mla_repr(v, 6000)
    except BaseException as e:
        res['error'] = _mla_err(e)
    res['ms'] = round((time.perf_counter() - t0) * 1000, 1)
    plt = sys.modules.get('matplotlib.pyplot')
    if plt is not None:
        try:
            import base64
            for num in plt.get_fignums():
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=96, bbox_inches='tight')
                res['figures'].append(base64.b64encode(buf.getvalue()).decode('ascii'))
            plt.close('all')
        except Exception as e:
            res['figError'] = repr(e)
    return json.dumps(res)
`;

  // ------------------------------------------------------------------ Worker side (stringified)
  function workerMain() {
    var py = null, loading = null, queue = Promise.resolve();
    var STALL_MS = 15000;
    function post(m) { self.postMessage(m); }
    function mb(n) { return (n / 1048576).toFixed(1); }
    // Fetch a URL while watching for stalls (no bytes for STALL_MS → abort). Reading the body fills
    // the browser's HTTP cache, so Pyodide's own requests for the same files are then served locally.
    async function fetchWatched(url, onBytes, keep) {
      var ctrl = new AbortController(), last = Date.now();
      var timer = setInterval(function () { if (Date.now() - last > STALL_MS) ctrl.abort(); }, 1000);
      try {
        var res = await fetch(url, { signal: ctrl.signal });
        if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url);
        var reader = res.body.getReader(), chunks = [], size = 0;
        while (true) {
          var r = await reader.read();
          if (r.done) break;
          last = Date.now(); size += r.value.length;
          if (keep) chunks.push(r.value);
          if (onBytes) onBytes(r.value.length);
        }
        if (!keep) return null;
        var out = new Uint8Array(size), off = 0;
        chunks.forEach(function (c) { out.set(c, off); off += c.length; });
        return out;
      } catch (e) {
        throw new Error(e && e.name === 'AbortError' ? 'download stalled' : String((e && e.message) || e));
      } finally { clearInterval(timer); }
    }
    async function prefetch(root) {
      var lock = JSON.parse(new TextDecoder().decode(await fetchWatched(root + 'pyodide-lock.json', null, true)));
      var files = ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip'];
      var np = lock.packages && lock.packages.numpy;
      if (np && np.file_name) files.push(np.file_name);
      var total = 14.6 * 1048576, got = 0, lastPost = 0;
      for (var i = 0; i < files.length; i++) {
        await fetchWatched(root + files[i], function (n) {
          got += n; var now = Date.now();
          if (now - lastPost > 250) { lastPost = now; post({ type: 'status', state: 'loading', text: 'Downloading Python… ' + mb(Math.min(got, total)) + ' / ' + mb(Math.max(got, total)) + ' MB (first time only)' }); }
        }, false);
      }
    }
    // First visit: race a 256 KB slice of the big file on every mirror and use the fastest one.
    async function pickMirror(mirrors, fallback) {
      var ctrls = mirrors.map(function () { return new AbortController(); });
      var probes = mirrors.map(function (root, i) {
        return fetch(root + 'pyodide.asm.wasm', { headers: { Range: 'bytes=0-262143' }, signal: ctrls[i].signal })
          .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.arrayBuffer(); })
          .then(function () { return i; });
      });
      var timeout = new Promise(function (res) { setTimeout(function () { res(fallback); }, 8000); });
      var winner;
      try { winner = await Promise.race([Promise.any(probes), timeout]); } catch (e) { winner = fallback; }
      ctrls.forEach(function (c) { c.abort(); });
      return winner;
    }
    function init() {
      if (!loading) {
        loading = (async function () {
          var mirrors = self.MLA_MIRRORS, start = self.MLA_START || 0, base = null, lastErr = null;
          if (!self.MLA_HAS_PREF && typeof Promise.any === 'function') {
            post({ type: 'status', state: 'loading', text: 'Finding the fastest download server…' });
            start = await pickMirror(mirrors, start);
          }
          for (var k = 0; k < mirrors.length && !base; k++) {
            var idx = (start + k) % mirrors.length;
            post({ type: 'status', state: 'loading', text: k === 0 ? 'Downloading Python (first time only, ~15 MB)…' : 'Download stalled — trying another CDN (' + new URL(mirrors[idx]).host + ')…' });
            try { await prefetch(mirrors[idx]); base = mirrors[idx]; post({ type: 'mirror', index: idx }); }
            catch (e) { lastErr = e; }
          }
          if (!base) throw new Error('Could not download Python (' + ((lastErr && lastErr.message) || 'network error') + '). Check your internet connection and try again.');
          post({ type: 'status', state: 'loading', text: 'Starting Python…' });
          importScripts(base + 'pyodide.js');
          py = await loadPyodide({ indexURL: base });
          post({ type: 'status', state: 'loading', text: 'Loading NumPy…' });
          await py.loadPackage(['numpy']);
          py.runPython("import os\nos.environ['MPLBACKEND'] = 'AGG'");
          py.runPython(self.MLA_HARNESS);
          post({ type: 'ready', version: py.runPython("import sys\n'.'.join(map(str, sys.version_info[:3]))"), pyodide: py.version });
          return py;
        })();
        loading.catch(function () { loading = null; });
      }
      return loading;
    }
    async function loadImports(srcs) {
      for (var i = 0; i < srcs.length; i++) {
        if (!srcs[i]) continue;
        try {
          await py.loadPackagesFromImports(srcs[i], {
            messageCallback: function (m) { if (/^Loading /.test(m)) post({ type: 'status', state: 'busy', text: m.length > 90 ? m.slice(0, 90) + '…' : m }); },
            errorCallback: function () {}
          });
        } catch (e) { /* unknown package: the import error surfaces when the code runs */ }
      }
      var all = srcs.join('\n');
      if (/matplotlib/.test(all)) {
        try { py.runPython("import matplotlib\nmatplotlib.use('agg')\nimport matplotlib.pyplot as _mla_plt\n_mla_plt.show = lambda *a, **k: None\n_mla_plt.close('all')"); } catch (e) { /* ignore */ }
      }
    }
    async function handle(msg) {
      var id = msg.id;
      try {
        await init();
        if (msg.type === 'init') { post({ id: id, type: 'done', result: { ok: true } }); return; }
        (msg.files || []).forEach(function (f) {         // datasets the code reads, as data/<name> under the working directory
          var full = '/home/pyodide/' + f.path;
          try { py.FS.mkdirTree(full.slice(0, full.lastIndexOf('/'))); } catch (e) { /* exists */ }
          py.FS.writeFile(full, f.data);
        });
        var out;
        if (msg.type === 'exec') {
          await loadImports([msg.code]);
          post({ id: id, type: 'started' });
          py.setStdout({ batched: function (s) { post({ id: id, type: 'stdout', text: s + '\n' }); } });
          py.setStderr({ batched: function (s) { post({ id: id, type: 'stdout', text: s + '\n', err: true }); } });
          py.globals.set('_mla_code', msg.code);
          out = py.runPython('_mla_exec(_mla_code)');
        } else {
          var p = msg.problem || {};
          await loadImports([msg.code, p.prelude, p.solution, (p.tests || []).map(function (t) { return [t.setup, t.call, t.code].filter(Boolean).join('\n'); }).join('\n')]);
          post({ id: id, type: 'started' });
          py.setStdout({ batched: function () {} });
          py.setStderr({ batched: function () {} });
          py.globals.set('_mla_code', msg.code);
          py.globals.set('_mla_payload', JSON.stringify(p));
          out = py.runPython('_mla_run_tests(_mla_code, _mla_payload)');
        }
        post({ id: id, type: 'done', result: JSON.parse(out) });
      } catch (e) {
        post({ id: id, type: 'fail', error: String((e && e.message) || e) });
      }
    }
    self.onmessage = function (e) { queue = queue.then(function () { return handle(e.data); }); };
  }

  // ------------------------------------------------------------------ Main thread API
  const status = reactive({ state: 'idle', message: '', version: null, error: null, ready: false, busy: false });
  let worker = null, seq = 0, watch = null, lastProgress = 0;
  const pending = new Map();
  // Identical copies of the Pyodide distribution on jsDelivr's different CDN providers
  const MIRRORS = ['cdn', 'testingcf', 'fastly', 'gcore'].map(h => 'https://' + h + '.jsdelivr.net/pyodide/v' + PYODIDE_VERSION + '/full/');
  const MIRROR_KEY = 'mla.pyMirror';
  const NO_PROGRESS_MS = 75000;

  function preferredMirror() { try { const v = localStorage.getItem(MIRROR_KEY); const i = Number(v); return v != null && i >= 0 && i < MIRRORS.length ? i : -1; } catch (e) { return -1; } }
  function clearLoadTimers() { clearInterval(watch); watch = null; }
  function spawn() {
    sentFiles = new Set();
    const pref = preferredMirror();
    const src = 'self.MLA_MIRRORS=' + JSON.stringify(MIRRORS) + ';\nself.MLA_START=' + Math.max(0, pref) + ';\nself.MLA_HAS_PREF=' + (pref >= 0) + ';\nself.MLA_HARNESS=' + JSON.stringify(HARNESS) + ';\n(' + workerMain.toString() + ')();\n';
    const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    worker = new Worker(url);
    worker.onmessage = onMessage;
    worker.onerror = (e) => { failAll('crash', 'The Python worker crashed: ' + (e.message || 'unknown error')); };
    if (!status.ready) {
      status.state = 'loading'; status.message = 'Starting Python…';
      // watchdog: the worker reports download progress continuously; if it goes silent for too long, give up
      // cleanly (the next Run starts a fresh attempt) instead of spinning forever.
      clearLoadTimers(); lastProgress = Date.now();
      watch = setInterval(() => { if (status.ready) { clearLoadTimers(); return; } if (Date.now() - lastProgress > NO_PROGRESS_MS) failAll('load', 'Python stopped making download progress. Check your internet connection, then press Run again to retry.'); }, 3000);
    }
  }
  function onMessage(e) {
    const m = e.data || {};
    if (m.type === 'status') { lastProgress = Date.now(); status.message = m.text; if (!status.ready) status.state = 'loading'; return; }
    if (m.type === 'mirror') { try { localStorage.setItem(MIRROR_KEY, String(m.index)); } catch (err) { /* ignore */ } return; }
    if (m.type === 'ready') { clearLoadTimers(); status.ready = true; status.version = m.version; status.pyodide = m.pyodide; status.state = pending.size ? 'busy' : 'ready'; status.message = 'Python ' + m.version + ' ready'; status.error = null; return; }
    const job = pending.get(m.id); if (!job) return;
    if (m.type === 'started') { status.state = 'busy'; status.message = 'Running…'; if (job.timeoutMs) job.timer = setTimeout(() => timeoutJob(m.id), job.timeoutMs); return; }
    if (m.type === 'stdout') { if (job.onStdout) job.onStdout(m.text, !!m.err); return; }
    if (m.type === 'done' || m.type === 'fail') {
      clearTimeout(job.timer); pending.delete(m.id);
      if (!pending.size) { status.state = status.ready ? 'ready' : 'idle'; status.message = status.ready ? 'Python ' + status.version + ' ready' : ''; }
      if (m.type === 'done') job.resolve({ ok: true, result: m.result });
      else {
        const loadFail = !status.ready;
        if (loadFail) { status.state = 'error'; status.error = m.error; status.message = 'Could not load Python'; kill(); }
        job.resolve({ ok: false, kind: loadFail ? 'load' : 'crash', message: loadFail ? 'Could not download the Python runtime from cdn.jsdelivr.net. Check your internet connection (the first run needs it; afterwards it is cached). Details: ' + m.error : m.error });
      }
    }
  }
  function kill() { clearLoadTimers(); if (worker) { try { worker.terminate(); } catch (e) { /* ignore */ } } worker = null; status.ready = false; }
  function failAll(kind, message) {
    kill();
    for (const [id, job] of pending) { clearTimeout(job.timer); job.resolve({ ok: false, kind, message }); pending.delete(id); }
    const isError = kind === 'crash' || kind === 'load';
    status.state = isError ? 'error' : 'idle'; status.message = isError ? message : 'Python stopped'; if (isError) status.error = message;
  }
  function timeoutJob(id) {
    const job = pending.get(id); if (!job) return;
    failAll('timeout', 'Time limit exceeded (' + Math.round(job.timeoutMs / 1000) + ' s). Your code may have an infinite loop or be too slow. Python was restarted.');
  }
  function send(type, payload, opts) {
    opts = opts || {};
    return new Promise(resolve => {
      if (typeof Worker === 'undefined') { resolve({ ok: false, kind: 'unsupported', message: 'This browser does not support Web Workers, so Python cannot run here.' }); return; }
      if (!worker) spawn();
      const id = ++seq;
      pending.set(id, { resolve, onStdout: opts.onStdout, timeoutMs: opts.timeoutMs || 0, timer: null });
      if (status.ready) { status.state = 'busy'; status.message = 'Running…'; }
      worker.postMessage(Object.assign({ id, type }, payload));
    });
  }
  function timeoutMs() { try { return Math.max(3, Number(MLA.store.state.profile.timeoutSec) || 20) * 1000; } catch (e) { return 20000; } }

  function warmup() { return send('init', {}); }
  // Datasets the code mentions as data/<name> (js/core/datasets.js) are written into Python's file system before it
  // runs. Each worker gets each version of a file once; a restarted worker starts with an empty file system.
  let sentFiles = new Set();
  async function filesFor(codes) {
    if (!MLA.datasets) return [];
    const names = new Set();
    for (const c of codes) for (const n of MLA.datasets.referenced(c)) names.add(n);
    const out = [];
    for (const n of names) {
      const key = n + '|' + MLA.datasets.version(n);
      if (sentFiles.has(key)) continue;
      try { out.push({ path: 'data/' + n, data: await MLA.datasets.read(n) }); sentFiles.add(key); } catch (e) { /* Python reports the missing file */ }
    }
    return out;
  }
  async function exec(code, opts) {
    opts = opts || {};
    const files = await filesFor([code]);
    return send('exec', { code: String(code || ''), files }, { onStdout: opts.onStdout, timeoutMs: opts.timeoutMs || timeoutMs() });
  }
  // mode 'run' = visible examples only, 'submit' = every test (including hidden ones)
  function runTests(code, problem, mode, opts) {
    opts = opts || {};
    const tests = (problem.tests || []).filter(t => mode === 'submit' || !t.hidden).map(t => ({ name: t.name, setup: t.setup || '', call: t.call || '', expect: t.expect == null ? null : t.expect, code: t.code || '', tol: t.tol || null, hidden: !!t.hidden }));
    return filesFor([code, problem.prelude].concat(tests.map(t => t.setup + '\n' + t.code))).then(files =>
      send('test', { code: String(code || ''), files, problem: { prelude: problem.prelude || '', solution: problem.solution || '', tests } }, { timeoutMs: opts.timeoutMs || timeoutMs() }));
  }
  function stop() { if (pending.size) failAll('stopped', 'Stopped. Python was restarted.'); else kill(); status.state = 'idle'; status.message = ''; }

  // Friendly explanations for the errors beginners hit most
  function explain(err) {
    if (!err) return '';
    const m = (err.message || '') + ' ' + (err.type || '');
    if (/No module named '(torch|tensorflow|transformers|jax|keras|torchvision)/.test(m)) return 'That library is not available in the in-browser Python runtime. These exercises use NumPy instead — the ideas are identical. Run PyTorch/TensorFlow code on your own machine or in Google Colab.';
    if (/No module named/.test(m)) return 'That package is not available in the browser runtime. NumPy, pandas, scikit-learn, SciPy and matplotlib are.';
    if (/IndentationError|TabError/.test(m)) return 'Python uses indentation to group code. Indent the body of every def / if / for / while consistently (4 spaces), and do not mix tabs and spaces.';
    if (/name 'np' is not defined/.test(m)) return 'Add `import numpy as np` at the top of your code.';
    if (/NameError/.test(m)) return 'A name is used before it is defined. Check spelling and capitalisation, and that the variable or function is defined above where it is used.';
    if (/'NoneType' object/.test(m)) return 'Something is None. A common cause: a function without a `return` statement returns None.';
    if (/shapes? .* (not aligned|mismatch)|could not be broadcast|matmul: Input operand/.test(m)) return 'Array shapes do not fit together. Print `.shape` of each array: for A @ B the inner dimensions must match, (n, k) @ (k, m) → (n, m).';
    if (/ZeroDivisionError/.test(m)) return 'Division by zero. Guard against empty inputs or add a small epsilon.';
    if (/IndexError/.test(m)) return 'An index is out of range. Remember Python indexes start at 0 and the last valid index is len(x) - 1.';
    if (/SyntaxError/.test(m)) return 'Python could not parse the code. Look for a missing colon after def/if/for, an unclosed bracket or quote, or a typo near the line shown.';
    if (/RecursionError/.test(m)) return 'A function keeps calling itself without stopping. Check your base case.';
    return '';
  }

  MLA.python = { VERSION: PYODIDE_VERSION, INDEX_URL, HARNESS, status, warmup, exec, runTests, stop, explain, _workerMain: workerMain };
})(typeof globalThis !== 'undefined' ? globalThis : window);
