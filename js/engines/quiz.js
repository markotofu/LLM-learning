/* Quiz engine: grading, scoring, shuffling (framework independent) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  function norm(s) { return String(s == null ? '' : s).trim().toLowerCase().replace(/\s+/g, ' ').replace(/[.]+$/, ''); }
  function fillMatches(answer, accepted) {
    const a = norm(answer); if (!a) return false;
    const list = Array.isArray(accepted) ? accepted : [accepted];
    return list.some(x => {
      if (/^\/.*\/[a-z]*$/.test(String(x))) { try { const m = String(x).match(/^\/(.*)\/([a-z]*)$/); return new RegExp(m[1], m[2]).test(String(answer).trim()); } catch (err) { return false; } }
      const e = norm(x);
      if (e === a) return true;
      const num = v => (/^-?\d*\.?\d+(e-?\d+)?$/.test(v) ? Number(v) : NaN);
      const na = num(a), ne = num(e);
      return !isNaN(na) && !isNaN(ne) && Math.abs(na - ne) <= 1e-9 + 1e-6 * Math.abs(ne);
    });
  }
  function grade(q, answer) {
    let correct = false, partial = 0, expected = '';
    switch (q.type) {
      case 'mcq': correct = Number(answer) === Number(q.answer); expected = q.options[q.answer]; partial = correct ? 1 : 0; break;
      case 'multi': {
        const ans = Array.isArray(answer) ? answer.map(Number) : [];
        correct = (q.answer || []).slice().sort().join(',') === ans.slice().sort().join(',');
        expected = (q.answer || []).map(i => q.options[i]).join('; ');
        const hits = ans.filter(i => q.answer.includes(i)).length, wrong = ans.filter(i => !q.answer.includes(i)).length;
        partial = correct ? 1 : Math.max(0, (hits - wrong) / q.answer.length); break;
      }
      case 'tf': correct = (answer === true || answer === 'true') === !!q.answer; expected = q.answer ? 'True' : 'False'; partial = correct ? 1 : 0; break;
      case 'fill': correct = fillMatches(answer, q.answer); expected = q.display || (Array.isArray(q.answer) ? q.answer.filter(x => !/^\/.*\/[a-z]*$/.test(String(x)))[0] : q.answer) || ''; partial = correct ? 1 : 0; break;
      case 'order': {
        const want = q.items.map((x, i) => i); const got = Array.isArray(answer) ? answer : [];
        correct = want.join(',') === got.join(',');
        expected = q.items.join(' → ');
        let hits = 0; want.forEach((v, i) => { if (got[i] === v) hits++; });
        partial = correct ? 1 : hits / want.length; break;
      }
      case 'match': {
        const ans = answer || {}; let hits = 0;
        q.pairs.forEach((p, i) => { if (Number(ans[i]) === i) hits++; });
        correct = hits === q.pairs.length; partial = hits / q.pairs.length;
        expected = q.pairs.map(p => p[0] + ' → ' + p[1]).join('; '); break;
      }
      default: correct = false;
    }
    return { correct, partial, expected, explanation: q.explain || '' };
  }
  function gradeAll(questions, answers) {
    const results = questions.map((q, i) => Object.assign({ index: i, question: q, answer: answers[i] }, grade(q, answers[i])));
    const points = results.reduce((a, r) => a + r.partial, 0);
    const score = questions.length ? Math.round((points / questions.length) * 100) : 0;
    return { score, correct: results.filter(r => r.correct).length, total: questions.length, results };
  }
  // Copy with shuffled options (answer indices remapped). "order" questions get a shuffled start order, and "match"
  // questions a shuffled order for the right-hand choices (rightOrder; answers still name pairs by index).
  function shuffled(q, rng) {
    if (q.type === 'order') return Object.assign({}, q, { start: MLA.util.shuffle(q.items.map((x, i) => i), rng) });
    if (q.type === 'match') return Object.assign({}, q, { rightOrder: MLA.util.shuffle(q.pairs.map((p, i) => i), rng) });
    if (!q.options || q.noShuffle) return q;
    const idx = MLA.util.shuffle(q.options.map((o, i) => i), rng);
    const copy = Object.assign({}, q, { options: idx.map(i => q.options[i]) });
    if (q.type === 'mcq') copy.answer = idx.indexOf(q.answer);
    if (q.type === 'multi') copy.answer = q.answer.map(a => idx.indexOf(a)).sort((a, b) => a - b);
    return copy;
  }
  function prepare(questions, rng) { return (questions || []).map(q => shuffled(q, rng)); }
  function describeType(t) { return { mcq: 'Multiple choice', multi: 'Select all that apply', tf: 'True or false', fill: 'Fill in the blank', order: 'Put in order', match: 'Match the pairs' }[t] || t; }
  function validQuestion(q) {
    if (!q || !q.type || !q.q) return false;
    if (q.type === 'mcq') return Array.isArray(q.options) && q.options.length >= 2 && Number.isInteger(q.answer) && q.answer >= 0 && q.answer < q.options.length;
    if (q.type === 'multi') return Array.isArray(q.options) && Array.isArray(q.answer) && q.answer.length > 0 && q.answer.every(a => a >= 0 && a < q.options.length);
    if (q.type === 'tf') return typeof q.answer === 'boolean';
    if (q.type === 'fill') return q.answer != null && (!Array.isArray(q.answer) || q.answer.length > 0);
    if (q.type === 'order') return Array.isArray(q.items) && q.items.length >= 2;
    if (q.type === 'match') return Array.isArray(q.pairs) && q.pairs.length >= 2 && q.pairs.every(p => Array.isArray(p) && p.length === 2);
    return false;
  }

  MLA.quiz = { grade, gradeAll, shuffled, prepare, describeType, norm, fillMatches, validQuestion };
})(typeof globalThis !== 'undefined' ? globalThis : window);
