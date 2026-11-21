#!/usr/bin/env node
/* Writes a static, readable page for every lesson (lessons/<id>.html), a lesson index (lessons/index.html),
 * sitemap.xml and robots.txt. The app itself lives behind hash routes (#/concept/<id>), which search engines do not
 * index and chat apps cannot preview; these pages carry each lesson's title, description and OpenGraph tags, its
 * text and code, and charts described in words, with a link into the interactive lesson.
 * The deploy workflow runs it before publishing (.github/workflows/deploy.yml); the output is not committed.
 *   node tools/snapshots.js            SITE_URL overrides the published address. */
'use strict';
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const SITE = (process.env.SITE_URL || 'https://markotofu.github.io/LLM-learning').replace(/\/+$/, '');

const MLA = require('../tests/_load.js').load();
if (!MLA.content.ready) MLA.content.init();
const esc = MLA.util.escapeHtml;
const md = s => MLA.util.md(s || '');

const STYLE = `
:root { color-scheme: light dark; --bg: #fbf8f1; --text: #2b2418; --muted: #6b5f4b; --accent: #8a5a12; --panel: #f3ecdd; --border: #e0d4bb; }
@media (prefers-color-scheme: dark) { :root { --bg: #17140f; --text: #efe6d2; --muted: #b9ab90; --accent: #e3b45f; --panel: #221d16; --border: #3a3226; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 17px/1.65 Georgia, "Times New Roman", serif; }
main, header, footer { max-width: 820px; margin: 0 auto; padding: 0 18px; }
header { padding-top: 18px; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; align-items: center; }
a { color: var(--accent); }
.open { display: inline-block; padding: 8px 14px; border: 1px solid var(--accent); border-radius: 8px; text-decoration: none; font-family: system-ui, sans-serif; font-size: 15px; }
h1 { font-size: 2rem; line-height: 1.2; margin: 18px 0 6px; } h2 { margin-top: 2em; font-size: 1.35rem; } h3 { font-size: 1.1rem; }
.lead { color: var(--muted); font-size: 1.05rem; }
pre, code { font: 14px/1.5 ui-monospace, Consolas, monospace; } pre { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow-x: auto; }
table { border-collapse: collapse; display: block; overflow-x: auto; } td, th { border: 1px solid var(--border); padding: 4px 8px; }
.note { background: var(--panel); border-left: 3px solid var(--accent); padding: 8px 12px; border-radius: 0 8px 8px 0; font-size: 0.95rem; }
.meta, footer { color: var(--muted); font-size: 0.9rem; } footer { padding: 30px 18px 40px; }
`;

function page(title, description, url, body, type) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:">
<title>${esc(title)}</title>
<meta name="description" content="${esc(description)}">
<link rel="canonical" href="${url}">
<link rel="icon" href="../icons/icon.svg" type="image/svg+xml">
<meta property="og:site_name" content="ML Atlas">
<meta property="og:type" content="${type || 'article'}">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(description)}">
<meta property="og:url" content="${url}">
<meta property="og:image" content="${SITE}/icons/icon-512.png">
<meta name="twitter:card" content="summary">
<style>${STYLE}</style>
</head>
<body>
${body}
<footer>${esc(MLA.data.legal && MLA.data.legal.copyright ? MLA.data.legal.copyright() : 'ML Atlas')} · <a href="${SITE}/">ML Atlas</a>, an interactive course from machine learning basics to LLMs, free in your browser.</footer>
</body>
</html>
`;
}

function sectionHtml(c, s) {
  const h = s.title ? `<h2 id="${esc(s.id)}">${esc(s.title)}</h2>\n` : '';
  if (s.type === 'text') return h + md(s.md);
  if (s.type === 'code') return h + md(s.md) + `<pre><code>${esc(s.code || '')}</code></pre>\n` + (s.after ? md(s.after) : '');
  if (s.type === 'chart') return h + (s.md ? md(s.md) : '') + `<p class="note">Chart, described in words: ${esc(MLA.narrate.chart(s.chart))} The interactive chart is in the app.</p>\n` + (s.after ? md(s.after) : '');
  if (s.type === 'demo') return h + `<p class="note">An interactive ${String(s.demo || '').startsWith('dg-') ? 'diagram' : 'demo'} goes here in the app.</p>\n`;
  return '';                                           // quick checks are interactive: they live in the app
}

function lessonPage(c) {
  const url = `${SITE}/lessons/${c.id}.html`, app = `${SITE}/#/concept/${c.id}`;
  const branch = MLA.content.branchById[c.branch];
  const link = id => { const x = MLA.content.byId[id]; return x ? `<a href="${x.id}.html">${esc(x.title)}</a>` : ''; };
  const next = (MLA.content.graph.children[c.id] || []).map(link).filter(Boolean);
  const parts = [
    `<header><a href="index.html">ML Atlas lessons</a><a class="open" href="${app}">Open the interactive lesson</a></header>`,
    '<main>',
    `<p class="meta">${esc(branch ? branch.name : '')} · level ${c.level} · about ${c.minutes} minutes</p>`,
    `<h1>${esc(c.title)}</h1>`,
    `<p class="lead">${esc(c.summary || '')}</p>`,
    c.prereqs.length ? `<p class="meta">Builds on: ${c.prereqs.map(link).filter(Boolean).join(', ')}</p>` : '',
    c.why ? '<h2>Why this matters</h2>' + md(c.why) : '',
    c.objectives.length ? '<h2>By the end you will be able to</h2><ul>' + c.objectives.map(o => '<li>' + MLA.util.inline(o) + '</li>').join('') + '</ul>' : '',
    ...c.sections.map(s => sectionHtml(c, s)),
    c.when ? '<h2>When to use it</h2>' + md(c.when) : '',
    c.pitfalls ? '<h2>Common pitfalls</h2>' + md(c.pitfalls) : '',
    c.takeaways.length ? '<h2>Key takeaways</h2><ul>' + c.takeaways.map(t => '<li>' + MLA.util.inline(t) + '</li>').join('') + '</ul>' : '',
    ...(c.cases || []).map(k => `<h2>Case study: ${esc(k.title)} (${esc(k.when)})</h2>` + md(k.story) + '<p><strong>What to take from it.</strong> ' + MLA.util.inline(k.lesson) + '</p>'),
    ...(c.projects || []).map(p => `<h2>Project: ${esc(p.title)}</h2><p>${MLA.util.inline(p.goal)}</p><p class="meta">${esc(p.level)} · about ${p.hours} hours · the steps, a checklist and notes are in the app</p>`),
    c.refs && c.refs.length ? '<h2>Go deeper</h2><ul>' + c.refs.map(r => `<li><a href="${esc(r.url)}">${esc(r.title)}</a></li>`).join('') + '</ul>' : '',
    `<p>${c.quiz.length} quiz questions and ${c.problems.length} practice problems with automatic checking are in the <a href="${app}">interactive lesson</a>.</p>`,
    next.length ? `<p class="meta">Next: ${next.join(', ')}</p>` : '',
    '</main>'
  ];
  return page(c.title + ' · ML Atlas', c.summary || c.title, url, parts.filter(Boolean).join('\n'));
}

function indexPage() {
  const groups = MLA.data.branches.map(b => {
    const list = MLA.data.concepts.filter(c => c.branch === b.id);
    return list.length ? `<h2>${esc(b.name)}</h2><ul>` + list.map(c => `<li><a href="${c.id}.html">${esc(c.title)}</a>: ${esc(c.summary || '')}</li>`).join('') + '</ul>' : '';
  }).join('\n');
  const body = `<header><a href="${SITE}/">ML Atlas</a><a class="open" href="${SITE}/">Open the app</a></header><main><h1>All lessons</h1>
<p class="lead">${MLA.data.concepts.length} lessons from machine learning basics to large language models. Each one has runnable Python, quick checks, a quiz and practice problems in the app.</p>
${groups}</main>`;
  return page('All lessons · ML Atlas', 'Every lesson in ML Atlas, an interactive course from machine learning basics to LLMs.', `${SITE}/lessons/`, body, 'website');
}

function main() {
  const dir = path.join(root, 'lessons');
  fs.mkdirSync(dir, { recursive: true });
  for (const c of MLA.data.concepts) fs.writeFileSync(path.join(dir, c.id + '.html'), lessonPage(c));
  fs.writeFileSync(path.join(dir, 'index.html'), indexPage());
  const today = new Date().toISOString().slice(0, 10);
  const urls = [`${SITE}/`, `${SITE}/lessons/`].concat(MLA.data.concepts.map(c => `${SITE}/lessons/${c.id}.html`));
  fs.writeFileSync(path.join(root, 'sitemap.xml'), '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' +
    urls.map(u => `  <url><loc>${u}</loc><lastmod>${today}</lastmod></url>`).join('\n') + '\n</urlset>\n');
  fs.writeFileSync(path.join(root, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${SITE}/sitemap.xml\n`);
  console.log('Wrote ' + MLA.data.concepts.length + ' lesson pages, lessons/index.html, sitemap.xml and robots.txt for ' + SITE);
}

if (require.main === module) main();
module.exports = { lessonPage, indexPage };
