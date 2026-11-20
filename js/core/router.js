/* ML Atlas – tiny hash router (works on GitHub Pages, any sub-path, and from file://) */
(function (global) {
  const MLA = global.MLA;
  const reactive = (global.Vue && global.Vue.reactive) ? global.Vue.reactive : (x => x);

  const routes = [
    { name: 'home', pattern: '/home', title: 'Dashboard' },
    { name: 'map', pattern: '/map', title: 'Concept Map', full: true },
    { name: 'concept', pattern: '/concept/:id', title: 'Concept' },
    { name: 'problems', pattern: '/problems', title: 'Practice Problems' },
    { name: 'problem', pattern: '/problem/:id', title: 'Problem', full: true },
    { name: 'playground', pattern: '/playground', title: 'Python Playground', full: true },
    { name: 'tracks', pattern: '/tracks', title: 'Learning Tracks' },
    { name: 'track', pattern: '/track/:id', title: 'Learning Track' },
    { name: 'review', pattern: '/review', title: 'Review' },
    { name: 'glossary', pattern: '/glossary', title: 'Glossary' },
    { name: 'achievements', pattern: '/achievements', title: 'Achievements' },
    { name: 'profile', pattern: '/profile', title: 'Progress & Sync' },
    { name: 'selftest', pattern: '/selftest', title: 'Self-test' },
    { name: 'terms', pattern: '/terms', title: 'Terms of Use' },
    { name: 'privacy', pattern: '/privacy', title: 'Privacy Policy' }
  ];

  function parseHash(hash) {
    let h = (hash || '').replace(/^#/, '');
    if (!h || h === '/') h = '/home';
    const [pathPart, queryPart] = h.split('?');
    const query = {};
    if (queryPart) queryPart.split('&').forEach(kv => { const [k, v] = kv.split('='); if (k) query[decodeURIComponent(k)] = decodeURIComponent(v || ''); });
    const segs = pathPart.split('/').filter(Boolean);
    for (const r of routes) {
      const psegs = r.pattern.split('/').filter(Boolean);
      if (psegs.length !== segs.length) continue;
      const params = {}; let ok = true;
      for (let i = 0; i < psegs.length; i++) {
        if (psegs[i][0] === ':') params[psegs[i].slice(1)] = decodeURIComponent(segs[i]);
        else if (psegs[i] !== segs[i]) { ok = false; break; }
      }
      if (ok) return { name: r.name, params, query, path: '/' + segs.join('/'), title: r.title, full: !!r.full };
    }
    return { name: 'notfound', params: {}, query, path: pathPart, title: 'Not found', full: false };
  }

  const route = reactive(parseHash(global.location ? global.location.hash : ''));

  function sync() {
    const r = parseHash(global.location.hash);
    const samePath = r.path === route.path;
    Object.assign(route, r);
    if (!samePath && global.scrollTo) try { global.scrollTo({ top: 0 }); } catch (e) { /* ignore */ }
  }
  function navigate(path, query) {
    let h = path;
    if (query) { const qs = Object.keys(query).filter(k => query[k] != null && query[k] !== '').map(k => encodeURIComponent(k) + '=' + encodeURIComponent(query[k])).join('&'); if (qs) h += '?' + qs; }
    if (global.location) { if (('#' + h) === global.location.hash) sync(); else global.location.hash = h; }
  }
  function back(fallback) { if (global.history && global.history.length > 1) global.history.back(); else navigate(fallback || '/home'); }
  if (global.addEventListener) global.addEventListener('hashchange', sync);

  MLA.router = { routes, route, navigate, back, parseHash, sync };
})(typeof globalThis !== 'undefined' ? globalThis : window);
