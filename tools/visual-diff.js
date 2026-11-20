#!/usr/bin/env node
/* Visual comparison of every demo and diagram (dark and light themes) between a git ref and the working tree.
 * 1. checks the ref out into a temporary git worktree and serves it on port 4174;
 * 2. screenshots every demo and diagram there: those pictures become the baseline (tests/e2e/.baseline/);
 * 3. serves the working tree on port 4173 and compares the same screenshots against the baseline.
 * Both sets are rendered on the same machine, so fonts and anti-aliasing match. Differences, with side-by-side
 * and diff images, are in the Playwright report: npx playwright show-report
 * Usage: node tools/visual-diff.js [ref]        (default ref: HEAD, i.e. your uncommitted changes) */
'use strict';
const { execFileSync, spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');

const root = path.join(__dirname, '..');
const ref = process.argv[2] || 'HEAD';
const cli = require.resolve('@playwright/test/cli');
const snapDir = path.join(root, 'tests/e2e/.baseline');
const tmp = path.join(os.tmpdir(), 'mla-visual-' + Date.now());

const git = args => execFileSync('git', args, { cwd: root, stdio: 'pipe', encoding: 'utf8' });
const serve = (port, dir) => spawn(process.execPath, [path.join(root, 'serve.js'), String(port), dir], { cwd: root, stdio: 'ignore' });
const ready = port => new Promise((resolve, reject) => {
  const t0 = Date.now();
  (function poll() {
    http.get('http://localhost:' + port + '/index.html', res => { res.resume(); resolve(); }).on('error', () => Date.now() - t0 > 20000 ? reject(new Error('server on ' + port + ' did not start')) : setTimeout(poll, 200));
  })();
});
const playwright = (url, update) => new Promise(resolve => {
  const env = Object.assign({}, process.env, { MLA_URL: url, MLA_UPDATE_SNAPSHOTS: update });
  const p = spawn(process.execPath, [cli, 'test', '--project=visual'], { cwd: root, env, stdio: 'inherit' });
  p.on('exit', code => resolve(code));
});

(async () => {
  let base, cur, status = 1;
  try {
    git(['worktree', 'add', '--detach', tmp, ref]);
    fs.rmSync(snapDir, { recursive: true, force: true });
    base = serve(4174, tmp); cur = serve(4173, root);
    await Promise.all([ready(4174), ready(4173)]);
    console.log('\nRendering the baseline (' + ref + ')…');
    await playwright('http://localhost:4174', 'all');
    console.log('\nComparing the working tree against it…');
    status = await playwright('http://localhost:4173', 'missing');
    console.log(status === 0 ? '\nNo visual differences.' : '\nVisual differences found: run `npx playwright show-report` to see them side by side.');
  } catch (e) {
    console.error(e.message || e);
  } finally {
    if (base) base.kill(); if (cur) cur.kill();
    try { git(['worktree', 'remove', '--force', tmp]); } catch (e) { /* already gone */ }
  }
  process.exit(status);
})();
