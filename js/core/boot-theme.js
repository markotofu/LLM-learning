/* ML Atlas – apply the saved theme before the first paint (avoids a flash of the wrong theme).
 * Loaded as a blocking script in <head>; kept as a file rather than inline so the Content-Security-Policy can forbid
 * inline scripts. */
(function () {
  try {
    var t = JSON.parse(localStorage.getItem('mla.state.v1') || '{}').profile;
    if (t && (t.theme === 'dark' || t.theme === 'darker')) document.documentElement.setAttribute('data-theme', t.theme);
  } catch (e) { /* storage unavailable or unreadable: keep the default theme */ }
})();
