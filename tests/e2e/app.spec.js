/* End-to-end tests in a real browser. They cover what the Node tests cannot see: rendering, layout on a phone,
 * interactions, saved progress and the in-browser Python. Run: npm run test:e2e */
'use strict';
const { test, expect } = require('@playwright/test');
const { boot, go, lessons } = require('./helpers');
const { audit } = require('./a11y');                       // a small built-in accessibility check, run in the page

test('every lesson renders, without errors, fits a phone screen and passes the accessibility check', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const errors = await boot(page, '/home');
  const list = await lessons(page);
  expect(list.length).toBeGreaterThan(100);
  const bad = [];
  for (const c of list) {
    await go(page, '/concept/' + c.id, c.title);
    const r = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      unknownDemo: document.body.textContent.includes('Unknown demo'),
      sections: document.querySelectorAll('.lesson-section').length,
    }));
    if (r.overflow > 1 || r.unknownDemo || r.sections < 3) bad.push(c.id + ' ' + JSON.stringify(r));
    const a11y = await page.evaluate(audit);
    if (a11y.length) bad.push(c.id + ' accessibility: ' + a11y.join('; '));
  }
  expect(bad).toEqual([]);
  expect(errors).toEqual([]);
});

test('the other pages render, fit a phone screen and pass the accessibility check', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const errors = await boot(page, '/home');
  const routes = [['/home', 'Dashboard'], ['/map', 'Concept Map'], ['/problems', 'Practice Problems'], ['/problem/dot-product', 'Dot Product'], ['/playground', 'Python Playground'], ['/tracks', 'Learning Tracks'], ['/track/build-llm', ''], ['/review', 'Review'], ['/quiz', 'Custom Quiz'], ['/glossary', 'Glossary'], ['/projects', 'Projects'], ['/achievements', 'Achievements'], ['/profile', 'Progress'], ['/selftest', 'Self-test'], ['/terms', 'Terms'], ['/privacy', 'Privacy']];
  const bad = [];
  for (const [r, t] of routes) {
    await go(page, r, t || undefined);
    const o = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (o > 1) bad.push(r + ' overflows by ' + o + 'px');
    const a11y = await page.evaluate(audit);
    if (a11y.length) bad.push(r + ' accessibility: ' + a11y.join('; '));
  }
  expect(bad).toEqual([]);
  expect(errors).toEqual([]);
});

test('demos keep their proportions and diagram labels fit their boxes', async ({ page }) => {
  const errors = await boot(page, '/home');
  const list = (await lessons(page)).filter(c => c.demos.length);
  const bad = [];
  for (const c of list) {
    await go(page, '/concept/' + c.id, c.title);
    await page.waitForTimeout(150);
    const issues = await page.evaluate(() => {
      const out = [];
      // A canvas drawn at W x H pixels must be displayed with the same shape (catches squashed or stretched demos).
      for (const cv of document.querySelectorAll('.demo canvas')) {
        const r = cv.getBoundingClientRect(), b = parseFloat(getComputedStyle(cv).borderTopWidth) || 0;
        const shown = (r.width - 2 * b) / (r.height - 2 * b);
        if (!cv.width || !cv.height || Math.abs(cv.width / cv.height - shown) > 0.03) out.push('canvas ' + cv.width + 'x' + cv.height + ' shown ' + Math.round(r.width) + 'x' + Math.round(r.height));
      }
      // Static diagrams: every label inside its box, no overlapping labels.
      for (const svg of document.querySelectorAll('.demo svg.diagram, .demo svg.walk-flow')) {
        const texts = [...svg.querySelectorAll('text')].filter(t => t.textContent.trim().length > 1).map(t => ({ t: t.textContent, b: t.getBBox() }));
        const boxes = [...svg.querySelectorAll('rect')].map(r => r.getBBox());
        for (const { t, b } of texts) for (const r of boxes) {
          const ox = Math.min(b.x + b.width, r.x + r.width) - Math.max(b.x, r.x), oy = Math.min(b.y + b.height, r.y + r.height) - Math.max(b.y, r.y);
          const inside = b.x >= r.x + 2 && b.x + b.width <= r.x + r.width - 2 && b.y >= r.y - 1 && b.y + b.height <= r.y + r.height + 1;
          if (ox > 1 && oy > 1 && !inside) out.push('label "' + t + '" crosses a box edge');
        }
        for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
          const a = texts[i].b, c = texts[j].b;
          if (Math.min(a.x + a.width, c.x + c.width) - Math.max(a.x, c.x) > 1 && Math.min(a.y + a.height, c.y + c.height) - Math.max(a.y, c.y) > 3) out.push('labels overlap: "' + texts[i].t + '" / "' + texts[j].t + '"');
        }
      }
      return out;
    });
    for (const i of issues) bad.push(c.id + ': ' + i);
  }
  expect(bad).toEqual([]);
  expect(errors).toEqual([]);
});

test('a step-through walkthrough steps, highlights its code and restarts when a setting changes', async ({ page }) => {
  const errors = await boot(page, '/home');
  await go(page, '/concept/knn', 'k-Nearest');
  const walk = page.locator('.walk').first();
  await expect(walk.locator('.walk-count')).toHaveText('step 1 of 10');
  for (let i = 0; i < 3; i++) await walk.getByRole('button', { name: 'Next step' }).click();
  await expect(walk.locator('.walk-count')).toHaveText('step 4 of 10');
  await expect(walk.locator('.walk-node.on')).toHaveCount(1);
  await expect(walk.locator('.walk-code li.on', { hasText: 'np.linalg.norm' })).toHaveCount(1);
  await expect(walk.locator('.walk-formula')).toContainText('d = √(');
  await expect(walk.locator('.walk-where')).toContainText('the straight-line distance');   // the symbols of this step
  await walk.locator('select[aria-label="k"]').selectOption('1');
  await expect(walk.locator('.walk-count')).toHaveText('step 1 of 10');
  for (let i = 0; i < 9; i++) await walk.getByRole('button', { name: 'Next step' }).click();
  await expect(walk.locator('.walk-say')).toContainText('labelled orange (1 of 1 votes)');
  expect(errors).toEqual([]);
});

test('switching themes recolours the app', async ({ page }) => {
  await boot(page, '/profile');
  await go(page, '/profile', 'Progress');
  await page.getByRole('button', { name: 'Vault', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe('rgb(22, 17, 11)');
  await page.getByRole('button', { name: 'Lapis', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'darker');
  await page.getByRole('button', { name: 'Fresco', exact: true }).click();
  expect(await page.evaluate(() => document.documentElement.getAttribute('data-theme'))).toBeNull();
  // the Classic look brings back the original palette, with its own theme names
  await page.getByRole('button', { name: 'Classic', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-style', 'classic');
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe('rgb(246, 247, 249)');
  await page.getByRole('button', { name: 'Graphite', exact: true }).click();
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe('rgb(15, 16, 18)');
  // the palette button in the top bar switches back
  await page.getByRole('button', { name: 'Switch to the Baroque look' }).click();
  expect(await page.evaluate(() => document.documentElement.getAttribute('data-style'))).toBeNull();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});

test('legal pages show the owner, the year and the stored data', async ({ page }) => {
  await boot(page, '/terms');
  await go(page, '/terms', 'Terms');
  await expect(page.locator('main')).toContainText('markotofu');
  await go(page, '/privacy', 'Privacy');
  await expect(page.locator('main')).toContainText('mla.state.v1');
});

test('progress saved by older versions (section positions) is migrated to section ids', async ({ page }) => {
  await boot(page, '/home', { version: 1, concepts: { 'transformer-block': { read: [0, 1], checks: { 3: true }, quizBest: 0, quizAttempts: 0, quizPassed: false } } });
  const cs = await page.evaluate(() => JSON.parse(JSON.stringify(MLA.store.state.concepts['transformer-block'])));
  expect(cs.read).toEqual(['one-block', 'the-residual-stream']);
  expect(Object.keys(cs.checks)).toHaveLength(1);
  expect(Object.keys(cs.checks)[0]).toMatch(/^check-/);
  await go(page, '/concept/transformer-block', 'The Transformer Block');
  await expect(page.locator('.lesson-section.read')).toHaveCount(2);
});

test('a wrong "predict first" answer is explained and queued for review', async ({ page }) => {
  await boot(page, '/home');
  await go(page, '/concept/probability', 'Probability');
  // the lesson has several predictions; pick the Bayes one by its section heading
  const section = page.locator('.lesson-section').filter({ has: page.getByRole('heading', { name: 'Bayes by counting', exact: true }) });
  const box = section.locator('.predict-box');
  await box.getByText('About 50%', { exact: true }).click();
  await expect(box.locator('.feedback')).toContainText('Not what happens');
  await go(page, '/review?tab=mistakes', 'Review');
  await expect(page.locator('.stat', { hasText: 'Coming back later' })).toContainText('1');
});

test('sync code: export, then merge it back without losing anything', async ({ page }) => {
  await boot(page, '/home', { xp: 120, problems: { 'dot-product': { solved: true, solvedAt: '2026-01-01T00:00:00Z', attempts: 1 } } });
  await go(page, '/profile', 'Progress');
  await page.getByRole('button', { name: 'Create & copy sync code' }).click();
  const code = await page.locator('textarea.sync-box[readonly]').inputValue();
  expect(code).toMatch(/^MLA1/);
  await page.getByPlaceholder('Paste a sync code').fill(code);
  await page.getByRole('button', { name: 'Apply sync code' }).click();
  await expect(page.locator('.toast', { hasText: 'Sync code applied' })).toBeVisible();
  expect(await page.evaluate(() => MLA.store.state.problems['dot-product'].solved)).toBe(true);
});

test('read aloud: the lesson plays part by part with an on-device voice, and the player stops', async ({ page }) => {
  // A stand-in for the browser's speech synthesis: one on-device voice, every utterance "ends" after 300 ms.
  await page.addInitScript(() => {
    const voice = { voiceURI: 'test-local', name: 'Test voice', lang: 'en-US', localService: true, default: true };
    window.__spoken = [];
    window.SpeechSynthesisUtterance = function (text) { this.text = text; };
    Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: {
      speaking: false, paused: false, pending: false, getVoices: () => [voice], addEventListener() {}, removeEventListener() {},
      cancel() {}, pause() {}, resume() {},
      speak(u) { window.__spoken.push(u.text); setTimeout(() => u.onend && u.onend(), 300); }
    } });
  });
  const errors = await boot(page, '/home');
  await go(page, '/concept/linear-regression', 'Linear');
  await page.getByRole('button', { name: 'Listen to the lesson' }).click();
  const player = page.getByRole('region', { name: 'Read aloud' });
  await expect(player).toContainText('part 1 of');
  await expect(page.locator('.concept-hero.speaking')).toHaveCount(1);
  await player.getByRole('button', { name: 'Next part' }).click();
  await expect(player).toContainText('part 2 of');
  expect(await page.evaluate(() => window.__spoken.length)).toBeGreaterThan(0);
  await player.getByRole('button', { name: 'Stop reading' }).click();
  await expect(player).toHaveCount(0);
  // one box alone: the takeaways play and the player closes after them
  await page.locator('[data-narr="takeaways"] .sec-listen').click();
  await expect(page.getByRole('region', { name: 'Read aloud' })).toContainText('Key takeaways');
  await expect(page.getByRole('region', { name: 'Read aloud' })).toHaveCount(0, { timeout: 20000 });
  expect(errors).toEqual([]);
});

test('a walkthrough runs on a practice dataset, and on a phone shows only the code of the current step', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const errors = await boot(page, '/home');
  await go(page, '/concept/svm', 'Support Vector');
  const walk = page.locator('.walk').first();
  await walk.locator('select[aria-label="Data"]').selectOption('card_transactions_synthetic.csv');
  await expect(walk.locator('.walk-data p')).toContainText('Blue is fraud = 0, orange is fraud = 1');
  await expect(walk.locator('select[aria-label="feature 1"]')).toHaveValue('amount');
  for (let i = 0; i < 4; i++) await walk.getByRole('button', { name: 'Next step' }).click();
  await expect(walk.locator('.walk-count')).toHaveText(/step 5 of \d+/);
  const shown = await walk.locator('.walk-code li').evaluateAll(els => els.filter(e => getComputedStyle(e).display !== 'none').length);
  expect(shown).toBeLessThan(8);
  await walk.getByRole('button', { name: /Show all 8 lines/ }).click();
  await expect(walk.locator('.walk-code li:visible')).toHaveCount(8);
  expect(errors).toEqual([]);
});

test('a walkthrough read aloud: its speaker button reads the steps and the card follows the voice', async ({ page }) => {
  await page.addInitScript(() => {                // the same silent stand-in: every utterance "ends" after 150 ms
    const voice = { voiceURI: 'test-local', name: 'Test voice', lang: 'en-US', localService: true, default: true };
    window.__spoken = [];
    window.SpeechSynthesisUtterance = function (text) { this.text = text; };
    Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: {
      speaking: false, paused: false, pending: false, getVoices: () => [voice], addEventListener() {}, removeEventListener() {},
      cancel() {}, pause() {}, resume() {},
      speak(u) { window.__spoken.push(u.text); setTimeout(() => u.onend && u.onend(), 150); }
    } });
  });
  const errors = await boot(page, '/home');
  await go(page, '/concept/knn', 'k-Nearest');
  const walk = page.locator('.walk').first();
  await expect(walk.locator('select[aria-label="Speed"] option')).toHaveText(['Sluggish', 'Super slow', 'Slow', 'Normal', 'Fast', 'With the voice']);
  await page.locator('[data-narr="step-through-how-k-nn-labels-a-new-point"] .sec-listen').click();
  await expect(walk.locator('select[aria-label="Speed"]')).toHaveValue('voice');
  await expect(walk.locator('.walk-count')).toHaveText('step 3 of 10', { timeout: 15000 });     // moved on with the voice
  expect(await page.evaluate(() => window.__spoken.some(t => t.includes('A new point arrives')))).toBe(true);
  await walk.getByRole('button', { name: 'Pause' }).click();
  expect(await page.evaluate(() => MLA.speech.state.playing)).toBe(false);
  // a timed speed is remembered for the next walkthrough
  await walk.locator('select[aria-label="Speed"]').selectOption('sluggish');
  expect(await page.evaluate(() => MLA.store.state.profile.walkSpeed)).toBe('sluggish');
  expect(errors).toEqual([]);
});

test.describe('in-browser Python (downloads Pyodide from the CDN)', () => {
  test('a lesson code sample runs and prints its output', async ({ page }) => {
    await boot(page, '/home');
    await go(page, '/concept/probability', 'Probability');
    const sample = page.locator('.lesson-section', { hasText: 'Bayes by counting' });
    await sample.getByRole('button', { name: /Run/ }).first().click();
    await expect(sample.locator('.ro-stdout')).toContainText('P(condition | positive) ~ 0.167', { timeout: 150 * 1000 });
  });

  test('an uploaded dataset is readable from Python and travels in the progress export', async ({ page }) => {
    await boot(page, '/home');
    await go(page, '/playground', 'Python Playground');
    await page.locator('.pg-data input[type=file]').setInputFiles({ name: 'shop sales.csv', mimeType: 'text/csv', buffer: Buffer.from('city,sales\nA,10\nB,5\nA,20\n') });
    await expect(page.locator('.pg-ds-name', { hasText: 'shop_sales.csv' })).toBeVisible();
    const out = await page.evaluate(async () => {
      let o = '';
      await MLA.python.exec('import pandas as pd\nprint(pd.read_csv("data/shop_sales.csv").sales.sum())', { onStdout: t => { o += t; }, timeoutMs: 150000 });
      return o.trim();
    });
    expect(out).toBe('35');
    const exported = await page.evaluate(async () => JSON.parse(await MLA.store.exportFull()).datasetFiles.map(f => f.name));
    expect(exported).toContain('shop_sales.csv');
  });

  test('a practice problem: the starter fails, the reference solution is accepted', async ({ page }) => {
    await boot(page, '/home');
    await go(page, '/problem/dot-product', 'Dot Product');
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page.getByText(/tests passed/).first()).toBeVisible({ timeout: 150 * 1000 });
    await expect(page.getByText('Wrong Answer').first()).toBeVisible();
    await page.getByRole('button', { name: /^Solution/ }).click();
    await page.getByRole('button', { name: 'Reveal solution' }).first().click();
    await page.locator('[role=dialog]').getByRole('button', { name: 'Reveal solution' }).click();
    await page.getByRole('button', { name: 'Copy into editor' }).first().click();
    await page.locator('[role=dialog]').getByRole('button', { name: 'Copy into editor' }).click();
    await page.getByRole('button', { name: 'Submit', exact: true }).click();
    await expect(page.getByText('Accepted').first()).toBeVisible({ timeout: 60 * 1000 });
  });
});

// Map at scale: zoom to a branch, hide finished branches and the progress heat map, in all three views.
test('the concept map zooms to a branch, hides finished branches and shows a heat map, in all three views', async ({ page }) => {
  const errors = await boot(page, '/home');
  // progress: Start and the whole Python & Data branch finished, one Math lesson mastered, one in progress
  await page.evaluate(() => {
    const at = '2026-01-01T00:00:00Z', S = MLA.store.state.concepts;
    const done = extra => Object.assign({ read: [], checks: {}, quizBest: 90, quizAttempts: 1, quizPassed: true, completedAt: at, masteredAt: null, visitedAt: at }, extra);
    for (const c of MLA.data.concepts) if (c.branch === 'start' || c.branch === 'code') S[c.id] = done();
    S['vectors-matrices'] = done({ masteredAt: at });
    S.probability = { read: [], checks: {}, quizBest: 0, quizAttempts: 0, quizPassed: false, completedAt: null, masteredAt: null, visitedAt: at };
  });
  await go(page, '/map', 'Concept Map');
  // what the engine says a zoom to Math shows: everything, and without the finished branches
  const math = await page.evaluate(() => {
    const f = MLA.graph.branchFocus(MLA.content.graph, 'math'), finished = ['start', 'code'];
    return { all: f.ids.size, unfinished: [...f.ids].filter(id => f.inside.has(id) || !finished.includes(MLA.content.byId[id].branch)).length };
  });

  // Mind map: the zoomed branch's hub in the middle, the lessons it builds on from other branches on the left
  await page.getByLabel('Zoom to a branch').selectOption('math');
  await page.waitForFunction(() => location.hash === '#/map?branch=math');
  await expect(page.locator('.focus-banner')).toContainText('Zoomed to Math Foundations');
  await expect(page.locator('.hub.hcenter')).toHaveAttribute('aria-label', /^Math, /);
  await expect(page.locator('.mnode')).toHaveCount(math.all);
  await expect(page.locator('.mnode.ctx').first()).toContainText('needed for');

  // Heat map: one level per status, filled from the theme's --heat tokens
  await page.getByLabel('Heat map').check();
  await expect(page.locator('.mnode.h3')).toHaveCount(1);       // mastered
  await expect(page.locator('.mnode.h1')).toHaveCount(1);       // in progress
  const fill = await page.evaluate(() => {
    const probe = document.createElement('div'); probe.style.background = 'var(--heat-3)'; document.body.appendChild(probe);
    const want = getComputedStyle(probe).backgroundColor; probe.remove();
    return { want, got: getComputedStyle(document.querySelector('.mnode.h3 .mnode-bg')).fill };
  });
  expect(fill.got).toBe(fill.want);

  // Hide finished branches: Start and Python & Data leave the map; the zoomed branch always stays
  await page.getByLabel('Hide finished branches').check();
  await expect(page.locator('.mnode')).toHaveCount(math.unfinished);
  await expect(page.locator('.focus-banner')).toContainText('from finished branches');

  // Prerequisite tree: the zoomed lane stands out, and a lane name zooms to its branch from the keyboard
  await page.getByRole('button', { name: 'Prerequisite tree' }).click();
  await expect(page.locator('.lane.focus .lane-label')).toContainText('Math Foundations');
  await expect(page.locator('.mnode')).toHaveCount(math.unfinished);
  await page.getByRole('button', { name: 'Zoom to Classical Machine Learning' }).focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => location.hash === '#/map?branch=ml');
  await expect(page.locator('.focus-banner')).toContainText('Zoomed to Classical Machine Learning');
  // a recommended prerequisite (Optimization Theory, for support vector machines) is dotted; required links are solid
  expect(await page.locator('path.edge.soft').count()).toBeGreaterThan(0);
  expect(await page.locator('path.edge.soft').first().evaluate(e => getComputedStyle(e).strokeDasharray)).not.toBe('none');
  expect(await page.locator('path.edge.tree').first().evaluate(e => getComputedStyle(e).strokeDasharray)).toBe('none');

  // Outline: the zoomed branch first, then what it builds on (finished branches hidden), rows coloured by progress
  await page.getByRole('button', { name: 'Outline' }).click();
  await expect(page.locator('.outline-branch h3').first()).toHaveText('Classical Machine Learning');
  await expect(page.locator('.outline-branch.ctx h3')).toHaveText(['Math Foundations']);
  await expect(page.locator('.outline-item.h3')).toHaveCount(1);
  await page.locator('.focus-banner').getByRole('button', { name: 'Show every branch' }).click();
  await page.waitForFunction(() => location.hash === '#/map');
  const open = await page.evaluate(() => MLA.progress.branchStats().filter(x => x.done < x.concepts).length);
  await expect(page.locator('.outline-branch')).toHaveCount(open);
  expect(await page.evaluate(() => [MLA.store.state.profile.mapHideDone, MLA.store.state.profile.mapHeat])).toEqual([true, true]);

  // a phone: every view, zoomed and with the options on, fits without scrolling sideways
  await page.setViewportSize({ width: 375, height: 812 });
  await page.getByLabel('Zoom to a branch').selectOption('llm');
  await page.waitForFunction(() => location.hash === '#/map?branch=llm');
  for (const view of ['Mind map', 'Prerequisite tree', 'Outline']) {
    await page.getByRole('button', { name: view }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
  }
  expect(errors).toEqual([]);
});

test('the map lists recommended prerequisites apart from required ones, and its side panel works from the keyboard', async ({ page }) => {
  const errors = await boot(page, '/home');
  await go(page, '/map?focus=svm', 'Concept Map');
  const panel = page.locator('.map-drawer');
  await expect(panel.locator('h3')).toHaveText('Support Vector Machines');
  const soft = panel.locator('.drawer-sec', { hasText: 'Recommended, not required' });
  await expect(soft.locator('.chip')).toHaveCount(1);
  await expect(soft.locator('.chip')).toContainText('Optimization Theory');
  await expect(panel.locator('.drawer-sec', { hasText: 'Prerequisites' }).locator('.chip', { hasText: 'Optimization Theory' })).toHaveCount(0);
  await expect(page.locator('path.edge.soft.hl-soft')).toHaveCount(1);     // the dotted link into the selected lesson
  // a chip opens its lesson's panel from the keyboard, which links back the other way
  await soft.locator('.chip').focus();
  await page.keyboard.press('Enter');
  await expect(panel.locator('h3')).toContainText('Optimization Theory');
  await expect(panel.locator('.drawer-sec', { hasText: 'Also helps with' })).toContainText('Support Vector Machines');
  // a lesson on the map opens with Space, and its panel zooms to its branch, keeping the selection
  await page.locator('.mnode[aria-label="Support Vector Machines"]').focus();
  await page.keyboard.press('Space');
  await expect(panel.locator('h3')).toHaveText('Support Vector Machines');
  await panel.getByRole('button', { name: 'Zoom to Classical ML' }).click();
  await page.waitForFunction(() => location.hash === '#/map?branch=ml');
  await expect(panel.locator('h3')).toHaveText('Support Vector Machines');
  expect(errors).toEqual([]);
});

// Placement test: started from the onboarding dialog and answered right throughout; then each mark's reason, undo of
// one mark, the marks on the map, a track, the dashboard and a lesson page, and undo of all of them.
test('the placement test marks the lessons a learner knows, explains each mark, and every mark can be undone', async ({ page }) => {
  const errors = await boot(page, '/home', { profile: { onboarded: false } });
  await page.getByRole('dialog').getByRole('button', { name: 'Take the placement test' }).click();
  await page.waitForFunction(() => location.hash === '#/placement');
  await page.getByRole('button', { name: 'Start the test' }).click();
  const card = page.locator('.placement-card');
  await expect(card).toHaveAttribute('data-n', '1');
  expect(await page.evaluate(audit)).toEqual([]);
  // the right answer to the question on screen, found through the lesson's own quiz (the options are shuffled)
  const rightAnswer = () => page.evaluate(() => {
    const c = document.querySelector('.placement-card'), q = MLA.placement.questions(c.dataset.lesson)[+c.dataset.k];
    if (q.type === 'fill') return { fill: String([].concat(q.answer).find(a => !/^\/.*\/[a-z]*$/.test(String(a)))) };
    const text = s => { const d = document.createElement('div'); d.innerHTML = MLA.util.inline(s); return d.textContent.trim(); };
    const want = (q.type === 'mcq' ? [q.answer] : q.answer).map(i => text(q.options[i]));
    return { pick: [...c.querySelectorAll('.pq .option')].map((b, i) => want.includes(b.textContent.trim().slice(1).trim()) ? i : -1).filter(i => i >= 0) };
  });
  let asked = 0;
  while (asked < 16 && await card.count()) {
    const a = await rightAnswer();
    if (a.fill) await card.getByRole('textbox', { name: 'Your answer' }).fill(a.fill);
    else for (const i of a.pick) await card.locator('.pq .option').nth(i).click();
    await card.getByRole('button', { name: 'Answer', exact: true }).click();
    asked++;
    await page.waitForFunction(n => { const c = document.querySelector('.placement-card'); return !c || +c.dataset.n === n; }, asked + 1);
  }
  expect(asked).toBeGreaterThanOrEqual(10);
  expect(asked).toBeLessThanOrEqual(15);
  await expect(page.getByRole('heading', { name: 'Your results' })).toBeVisible();
  await expect(page.locator('main')).toContainText('You answered ' + asked + ' questions, ' + asked + ' of them right.');
  const known = await page.evaluate(() => MLA.progress.knownList().map(k => k.id));
  expect(known.length).toBeGreaterThanOrEqual(20);
  expect(await page.evaluate(() => MLA.store.state.xp)).toBe(0);                // known is not completed: no XP
  const rows = page.locator('.placement-list').first().locator('li');
  await expect(rows).toHaveCount(known.length);
  await expect(rows.first()).toContainText('You answered');
  await expect(page.locator('.placement-next')).toContainText('Suggested next');
  expect(await page.evaluate(audit)).toEqual([]);
  // undo one lesson, then mark it again
  const first = await rows.first().locator('a').textContent();
  await rows.first().getByRole('button', { name: 'Undo the mark on ' + first }).click();
  await expect(rows.first()).toContainText('Not marked any more.');
  expect(await page.evaluate(id => MLA.progress.isKnown(id), known[0])).toBe(false);
  await rows.first().getByRole('button', { name: 'Mark again: ' + first }).click();
  expect(await page.evaluate(id => MLA.progress.isKnown(id), known[0])).toBe(true);
  // the map, a track and the dashboard show the marks
  await go(page, '/map', 'Concept Map');
  await expect(page.locator('.mnode.known')).toHaveCount(known.length);
  await expect(page.locator('.map-legend')).toContainText('known');
  await go(page, '/track/foundations', 'Machine Learning Foundations');
  expect(await page.locator('.track-step.known').count()).toBeGreaterThan(5);
  await expect(page.locator('main')).toContainText('known, so the path skips');
  await go(page, '/home', 'Dashboard');
  await expect(page.locator('main')).toContainText('marked as known');
  // a lesson page says why it is marked, and unmarks it
  const lesson = known.includes('numpy') ? 'numpy' : known[known.length - 1];
  await go(page, '/concept/' + lesson, await page.evaluate(id => MLA.progress.concept(id).title, lesson));
  await expect(page.locator('.callout.known-note')).toContainText('It counts as done');
  await page.getByRole('button', { name: 'Not known after all: unmark it' }).click();
  await expect(page.locator('.callout.known-note')).toHaveCount(0);
  expect(await page.evaluate(id => MLA.progress.status(id), lesson)).not.toBe('known');
  // undo all, from the results page
  await go(page, '/placement', 'Placement test');
  await page.getByRole('button', { name: 'Undo all' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Unmark all' }).click();
  expect(await page.evaluate(() => MLA.progress.knownList().length)).toBe(0);
  await go(page, '/map', 'Concept Map');
  await expect(page.locator('.mnode.known')).toHaveCount(0);
  expect(errors).toEqual([]);
});

// Custom quiz (#/quiz): lessons chosen by link, the lessons they build on, every question of two lessons (their own quiz
// plus the extra question bank), one wrong answer of each kind, results by lesson, and Review receiving only the missed
// question from a lesson's own quiz.
test('a custom quiz mixes lesson and bank questions, reports by lesson, and sends only lesson questions to review', async ({ page }) => {
  const errors = await boot(page, '/quiz?ids=knn,svm&deps=0');
  await page.waitForFunction(() => document.title.startsWith('Custom Quiz'));
  await expect(page.locator('.qb-lesson input:checked')).toHaveCount(2);
  const start = page.locator('.qb-start .btn.primary');
  await expect(start).toContainText('Start: 2 lessons');
  await page.getByLabel('Also ask about the lessons these build on').check();
  expect(Number((await start.textContent()).match(/Start: (\d+)/)[1])).toBeGreaterThan(5);
  await page.getByLabel('Also ask about the lessons these build on').uncheck();
  await page.getByLabel('Number of questions').selectOption({ label: 'All of them' });
  expect(await page.evaluate(audit)).toEqual([]);
  await start.click();
  await expect(page.locator('.qb-head')).toBeVisible();
  const want = await page.evaluate(() => {
    const own = ['knn', 'svm'].reduce((s, id) => s + MLA.content.byId[id].quiz.length, 0);
    const bank = ['knn', 'svm'].reduce((s, id) => s + MLA.data.quizBank[id].length, 0);
    return { total: own + bank, bank };
  });
  expect(want.bank).toBeGreaterThanOrEqual(6);
  await expect(page.locator('.quiz-q')).toHaveCount(want.total);
  await expect(page.locator('.qb-head')).toContainText(want.total + ' questions from 2 lessons, ' + want.bank + ' of them from the extra question bank');
  expect(await page.evaluate(audit)).toEqual([]);
  // answer every question from the data (options are shuffled): all right, except the first true-or-false question of
  // each source, a lesson's own quiz and the bank
  const plan = await page.evaluate(() => {
    const text = s => { const d = document.createElement('div'); d.innerHTML = MLA.util.inline(String(s)); return d.textContent.replace(/\s+/g, ' ').trim(); };
    const byTitle = {}; MLA.data.concepts.forEach(c => { byTitle[c.title] = c; });
    const missed = {};
    return [...document.querySelectorAll('.quiz-q')].map(el => {
      const c = byTitle[el.querySelector('.q-num .dim').textContent.replace(/^\s*·\s*/, '').trim()];
      const stem = el.querySelector('.q-text').textContent.replace(/\s+/g, ' ').trim();
      const own = c.quiz.find(x => text(x.q) === stem), q = own || MLA.data.quizBank[c.id].find(x => text(x.q) === stem);
      const src = own ? 'lesson' : 'bank';
      if (q.type === 'fill') return { fill: String([].concat(q.answer).find(a => !/^\/.*\/[a-z]*$/.test(String(a)))) };
      let want, miss = null;
      if (q.type === 'tf') { if (!missed[src]) { missed[src] = true; miss = src; } want = [(q.answer !== !!miss) ? 'True' : 'False']; }
      else want = (q.type === 'mcq' ? [q.answer] : q.answer).map(i => text(q.options[i]));
      // option indices on screen, matched on the whole option text (after its letter key)
      const pick = [...el.querySelectorAll('.option')].map((b, i) => want.includes(b.textContent.replace(/\s+/g, ' ').trim().slice(1).trim()) ? i : -1).filter(i => i >= 0);
      return { pick, miss };
    });
  });
  const cards = page.locator('.quiz-q');
  for (let i = 0; i < plan.length; i++) {
    const p = plan[i], card = cards.nth(i);
    if (p.fill) { await card.getByRole('textbox', { name: 'Your answer' }).fill(p.fill); continue; }
    expect(p.pick.length).toBeGreaterThan(0);
    for (const k of p.pick) await card.locator('.option').nth(k).click();
  }
  expect(plan.filter(p => p.miss).map(p => p.miss).sort()).toEqual(['bank', 'lesson']);
  await page.getByRole('button', { name: 'Submit answers' }).click();
  await expect(page.locator('.result-card')).toContainText((want.total - 2) + ' of ' + want.total + ' correct');
  const rows = page.locator('.qb-break');
  await expect(rows).toHaveCount(2);
  const right = await rows.locator('.small').allTextContents();
  const rate = s => { const m = s.match(/(\d+) of (\d+)/); return m[1] / m[2]; };
  expect(rate(right[0])).toBeLessThanOrEqual(rate(right[1]));                 // weakest lesson first
  expect(await page.evaluate(() => MLA.mistakes.stats().total)).toBe(1);      // only the lesson question is queued
  // a new draw of the same lessons, and the builder again
  await page.getByRole('button', { name: 'New questions, same lessons' }).click();
  await expect(page.locator('.result-card')).toHaveCount(0);
  await page.getByRole('button', { name: 'Change the lessons' }).click();
  await expect(page.locator('.qb-branch')).toHaveCount(8);
  expect(errors).toEqual([]);
});
