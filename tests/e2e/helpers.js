/* Shared helpers for the browser tests. */
'use strict';
require('../../js/course.js');                      // MLA.course: the storage key and id of this course
const COURSE = globalThis.MLA.course;

// Seed localStorage before the app starts (once per page), so tests skip onboarding and start from a known state.
async function seed(page, extra) {
  await page.addInitScript(([extra, key, app]) => {
    if (sessionStorage.getItem('mla-seeded')) return;
    sessionStorage.setItem('mla-seeded', '1');
    const base = { app, version: 2, xp: 0, concepts: {}, problems: {}, profile: { name: 'Tester', onboarded: true, theme: 'dark' } };
    const s = Object.assign({}, base, extra || {});
    s.profile = Object.assign({}, base.profile, (extra && extra.profile) || {});
    localStorage.setItem(key, JSON.stringify(s));
    // Deterministic randomness, so demos look the same on every run.
    let a = 42;
    Math.random = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }, [extra || null, COURSE.storageKey, COURSE.id]);
}

// Open the app at a route and wait until the content registry is ready. Returns the list of uncaught page errors.
async function boot(page, route, extra) {
  const errors = [];
  page.on('pageerror', e => errors.push(String(e && e.message || e)));
  await seed(page, extra);
  await page.goto('/#' + (route || '/home'));
  await page.waitForFunction(() => window.MLA && MLA.content && MLA.content.ready && MLA.store && document.querySelector('#app .app, #app > *'));
  return errors;
}

// Client-side navigation (no reload) and a wait for the page title the router sets.
async function go(page, route, titleStart) {
  await page.evaluate(r => { location.hash = '#' + r; }, route);
  if (titleStart) await page.waitForFunction(t => document.title.startsWith(t), titleStart);
  await page.waitForFunction(() => !document.querySelector('.content-loading'));     // lesson files load on demand
  await page.waitForTimeout(80);
}

async function lessons(page) {
  return page.evaluate(() => MLA.data.concepts.map(c => ({ id: c.id, title: c.title, demos: c.sections.filter(s => s.type === 'demo').map(s => s.demo) })));
}

module.exports = { seed, boot, go, lessons };
