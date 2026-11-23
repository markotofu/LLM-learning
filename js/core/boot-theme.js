/* Apply the saved theme before the first paint (avoids a flash of the wrong theme).
 * Loaded as a blocking script in <head>; kept as a file rather than inline so the Content-Security-Policy can forbid
 * inline scripts. */
(function () {
  try {
    var t = JSON.parse(localStorage.getItem(window.MLA && MLA.course ? MLA.course.storageKey : 'mla.state.v1') || '{}').profile;
    if (t && (t.theme === 'dark' || t.theme === 'darker')) document.documentElement.setAttribute('data-theme', t.theme);
    if (t && t.style === 'classic') document.documentElement.setAttribute('data-style', 'classic');
  } catch (e) { /* storage unavailable or unreadable: keep the default theme */ }
})();
