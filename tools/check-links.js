#!/usr/bin/env node
/* Checks that every external link in the lessons still works: the "Go deeper" references, case-study sources,
 * and the links inside lesson text and glossary definitions. Needs a network connection, so it is not part of
 * npm test; .github/workflows/links.yml runs it once a week and on demand.
 *   node tools/check-links.js            prints every broken link and exits with 1 if any link is gone
 * Sites that refuse automated requests (403, 429) are listed as "blocked" without failing the run. */
'use strict';
const MLA = require('../tests/_load.js').load({ vue: false });
if (!MLA.content.ready) MLA.content.init();

const links = new Map();                                    // url -> where it is used
const add = (url, where) => { if (/^https?:[/][/]/.test(url)) { if (!links.has(url)) links.set(url, []); links.get(url).push(where); } };
const scan = (text, where) => { for (const m of String(text || '').matchAll(/[(<](https?:[/][/][^)\s>]+)[)>]/g)) add(m[1], where); };
for (const c of MLA.data.concepts) {
  for (const r of c.refs || []) add(r.url, c.id + ' (references)');
  for (const k of c.cases || []) if (k.source && k.source.url) add(k.source.url, c.id + ' (case study)');
  for (const f of ['why', 'when', 'pitfalls']) scan(c[f], c.id);
  for (const s of c.sections) { scan(s.md, c.id); scan(s.after, c.id); }
}
for (const g of MLA.data.glossary || []) scan(g.def, 'glossary: ' + g.term);

async function check(url) {
  const opts = { redirect: 'follow', headers: { 'user-agent': 'Mozilla/5.0 (compatible; ML-Atlas-link-check)' }, signal: AbortSignal.timeout(20000) };
  try {
    let res = await fetch(url, Object.assign({ method: 'HEAD' }, opts));
    if (res.status === 405 || res.status === 403 || res.status === 400) res = await fetch(url, Object.assign({ method: 'GET' }, opts));
    return res.status;
  } catch (e) { return 'error: ' + ((e && e.cause && e.cause.code) || (e && e.name) || e); }
}

(async () => {
  const urls = [...links.keys()];
  const results = [];
  let next = 0;
  await Promise.all(Array.from({ length: 6 }, async () => {
    while (next < urls.length) { const url = urls[next++]; results.push({ url, status: await check(url) }); }
  }));
  // Network errors and server errors are often temporary: try those once more after a pause.
  const flaky = r => typeof r.status !== 'number' || r.status >= 500;
  if (results.some(flaky)) await new Promise(r => setTimeout(r, 5000));
  for (const r of results.filter(flaky)) r.status = await check(r.url);
  const blocked = results.filter(r => r.status === 401 || r.status === 403 || r.status === 429 || r.status === 999);
  const unreachable = results.filter(flaky);
  const broken = results.filter(r => typeof r.status === 'number' && r.status >= 400 && r.status < 500 && !blocked.includes(r));   // 404, 410: gone
  const where = r => '  (' + links.get(r.url).join(', ') + ')';
  for (const r of broken) console.log('BROKEN ' + r.status + '  ' + r.url + where(r));
  for (const r of unreachable) console.log('unreachable (may be temporary) ' + r.status + '  ' + r.url + where(r));
  for (const r of blocked) console.log('blocked ' + r.status + '  ' + r.url);
  console.log(urls.length + ' links checked: ' + (urls.length - broken.length - blocked.length - unreachable.length) + ' fine, ' + blocked.length + ' refuse automated requests, ' +
    unreachable.length + ' unreachable right now, ' + broken.length + ' broken');
  process.exit(broken.length ? 1 : 0);
})();
