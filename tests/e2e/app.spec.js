/* End-to-end tests in a real browser. They cover what the Node tests cannot see: rendering, layout on a phone,
 * interactions, saved progress and the in-browser Python. Run: npm run test:e2e */
'use strict';
const { test, expect } = require('@playwright/test');
const { boot, go, lessons } = require('./helpers');

test('every lesson renders, without errors, and fits a phone screen', async ({ page }) => {
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
  }
  expect(bad).toEqual([]);
  expect(errors).toEqual([]);
});

test('the other pages render and fit a phone screen', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const errors = await boot(page, '/home');
  const routes = [['/home', 'Dashboard'], ['/map', 'Concept Map'], ['/problems', 'Practice Problems'], ['/problem/dot-product', 'Dot Product'], ['/playground', 'Python Playground'], ['/tracks', 'Learning Tracks'], ['/track/build-llm', ''], ['/review', 'Review'], ['/glossary', 'Glossary'], ['/projects', 'Projects'], ['/achievements', 'Achievements'], ['/profile', 'Progress'], ['/selftest', 'Self-test'], ['/terms', 'Terms'], ['/privacy', 'Privacy']];
  const bad = [];
  for (const [r, t] of routes) {
    await go(page, r, t || undefined);
    const o = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (o > 1) bad.push(r + ' overflows by ' + o + 'px');
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
      for (const svg of document.querySelectorAll('.demo svg.diagram')) {
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

test.describe('in-browser Python (downloads Pyodide from the CDN)', () => {
  test('a lesson code sample runs and prints its output', async ({ page }) => {
    await boot(page, '/home');
    await go(page, '/concept/probability', 'Probability');
    const sample = page.locator('.lesson-section', { hasText: 'Bayes by counting' });
    await sample.getByRole('button', { name: /Run/ }).first().click();
    await expect(sample.locator('.ro-stdout')).toContainText('P(condition | positive) ~ 0.167', { timeout: 150 * 1000 });
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
