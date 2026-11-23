/* Terms of use and privacy policy pages */
(function (global) {
  const MLA = global.MLA;
  const C = (MLA.components = MLA.components || {});

  function legalView(key) {
    return {
      computed: {
        text() { const L = MLA.data.legal; return L ? L.render(L[key]) : ''; },
        copyright() { const L = MLA.data.legal; return L ? L.copyright() : ''; }
      },
      template: `
      <div class="stack legal-page">
        <div class="card"><mla-md :src="text" /></div>
        <div class="tiny dim center">{{ copyright }} · <a href="#/terms">Terms of Use</a> · <a href="#/privacy">Privacy Policy</a></div>
      </div>`
    };
  }

  C['view-terms'] = legalView('terms');
  C['view-privacy'] = legalView('privacy');
})(typeof globalThis !== 'undefined' ? globalThis : window);
