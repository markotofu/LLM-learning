/* Executes every practice problem under CPython using the exact HARNESS the browser runs:
 *   - the reference solution must pass every test,
 *   - the starter code must fail at least one test (otherwise the tests are too weak),
 *   - every Python code sample must compile, and runnable samples must run without errors.
 * Needs Python 3 with numpy (pandas / scikit-learn optional: problems that need a missing package are skipped).
 * Run: node tests/problems.test.js [problem-id-or-concept-id ...] */
'use strict';
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { load, suite, root } = require('./_load.js');
const MLA = load({ vue: false });
const T = suite('problems');
T.ok('scripts load without errors', MLA.__loadErrors.length === 0, MLA.__loadErrors.join('\n'));

const only = process.argv.slice(2);
const pick = x => !only.length || only.includes(x.id) || only.includes(x.concept);
const problems = MLA.data.problems.filter(pick).map(p => ({ id: p.id, concept: p.concept, prelude: p.prelude || '', starter: p.starter, solution: p.solution, tests: p.tests }));
const samples = [];
for (const c of MLA.data.concepts) {
  if (only.length && !only.includes(c.id)) continue;
  c.sections.forEach((s, i) => { if (s.type === 'code' && (!s.lang || s.lang === 'python' || s.lang === 'pytorch')) samples.push({ id: c.id + '#' + (i + 1), code: s.code, run: s.lang !== 'pytorch' && s.run !== false, compileOnly: s.lang === 'pytorch', expect: s.predict ? [].concat(s.predict.expect || []) : null }); });
}
const tmp = path.join(root, 'tests', '.tmp'); fs.mkdirSync(tmp, { recursive: true });
const payloadFile = path.join(tmp, 'payload.json');
fs.writeFileSync(payloadFile, JSON.stringify({ harness: MLA.python.HARNESS, problems, samples }));

let py = null;
for (const cand of ['python', 'python3', 'py']) { const r = spawnSync(cand, ['-c', 'import numpy; print(numpy.__version__)'], { encoding: 'utf8' }); if (r.status === 0) { py = cand; break; } }
if (!py) { console.log('SKIP: Python 3 with numpy not found — install it to verify problem solutions (CI does).'); process.exit(0); }

const r = spawnSync(py, ['-X', 'utf8', path.join(__dirname, 'run_problems.py'), payloadFile], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout: 30 * 60 * 1000, env: Object.assign({}, process.env, { MPLBACKEND: 'Agg', PYTHONIOENCODING: 'utf-8' }) });
if (r.status !== 0 || !r.stdout) { console.log(r.stderr || r.error || 'python runner failed'); process.exit(1); }
const out = JSON.parse(r.stdout.slice(r.stdout.indexOf('{')));
let skipped = 0;
const dump = {};
for (const p of out.problems) {
  if (p.skipped) { skipped++; console.log('  skip ' + p.id + ' (' + p.skipped + ')'); continue; }
  T.ok(p.id + ': reference solution passes all ' + p.total + ' tests', p.solOk, p.solDetail);
  T.ok(p.id + ': starter code fails at least one test', p.starterFails, 'starter passes everything — add a stronger test');
}
for (const s of out.samples) {
  if (s.skipped) { skipped++; continue; }
  T.ok('sample ' + s.id + ' compiles', s.compiles, s.detail);
  if (s.ran != null) T.ok('sample ' + s.id + ' runs', s.ran, s.detail);
  if (s.expectMissing) T.ok('sample ' + s.id + ': output confirms the prediction answer', s.expectMissing.length === 0, 'missing from output: ' + JSON.stringify(s.expectMissing));
  if (process.env.MLA_SAMPLE_OUTPUT && s.output != null) dump[s.id] = s.output;
}
if (process.env.MLA_SAMPLE_OUTPUT) fs.writeFileSync(process.env.MLA_SAMPLE_OUTPUT, JSON.stringify(dump, null, 1));
console.log(out.problems.length + ' problems and ' + out.samples.length + ' code samples checked with ' + out.python + (skipped ? ' (' + skipped + ' skipped: missing packages)' : '') + ' in ' + out.seconds + ' s');
T.done();
