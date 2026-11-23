/* Section layouts of lessons whose sections changed after a version that saved progress by section position.
 * ML Atlas needed this once; a new course has no such older progress, so the list is empty and
 * MLA.progress.migrateSections leaves saved progress as it is. */
(function (global) {
  const MLA = global.MLA;
  MLA.data.legacySections = {};
})(typeof globalThis !== 'undefined' ? globalThis : window);
