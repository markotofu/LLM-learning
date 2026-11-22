/* ML Atlas – narration: turns lessons into plain spoken sentences for the read-aloud player, so a lesson can be
 * followed by ear, without looking at the page. Pure functions, no DOM; tests/narrate.test.js checks every lesson.
 *   MLA.narrate.lesson(concept) -> [{ key, title, text }]  key: 'intro' | 'why' | 'objectives' | section id |
 *                                                          'when' | 'pitfalls' | 'takeaways' | 'end'
 *   MLA.narrate.md(markdown)    -> text   markdown to sentences: formulas and their legends, tables, lists, callouts
 *   MLA.narrate.math(formula)   -> text   Unicode math to words ("Σᵢ xᵢ²" -> "the sum over i of x sub i squared")
 *   MLA.narrate.chart(spec)     -> text   a chart in words: axes, what each line does, crossings, reference lines
 *   MLA.narrate.chunks(text)    -> [text | PAUSE]  sentence-sized pieces for the speech engine
 * Section narration also says what is on screen (code, a demo, a diagram) without reading code aloud. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  const PAUSE = '\u0000pause\u0000';                         // a short silence: think before the answer is read

  // ---------------------------------------------------------------- symbols
  const GREEK = { α: 'alpha', β: 'beta', γ: 'gamma', δ: 'delta', ε: 'epsilon', ϵ: 'epsilon', ζ: 'zeta', η: 'eta', θ: 'theta', ϑ: 'theta', ι: 'iota', κ: 'kappa', λ: 'lambda', μ: 'mu', ν: 'nu', ξ: 'xi', π: 'pi', ρ: 'rho', σ: 'sigma', ς: 'sigma', τ: 'tau', υ: 'upsilon', φ: 'phi', ϕ: 'phi', χ: 'chi', ψ: 'psi', ω: 'omega',
    Γ: 'capital gamma', Δ: 'delta', Θ: 'capital theta', Λ: 'capital lambda', Ξ: 'capital xi', Φ: 'capital phi', Ψ: 'capital psi', Ω: 'capital omega' };
  const SUP = { '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
    'ⁿ': 'n', 'ⁱ': 'i', 'ᵏ': 'k', 'ˣ': 'x', 'ᵗ': 't', 'ᶻ': 'z', 'ˢ': 's', 'ᵀ': 'T', 'ᵈ': 'd', 'ᵐ': 'm', 'ʲ': 'j', 'ᵃ': 'a', 'ᵇ': 'b', 'ᶜ': 'c', 'ᵉ': 'e', 'ʰ': 'h', 'ᵒ': 'o', 'ᵖ': 'p', 'ʳ': 'r', 'ᵘ': 'u', 'ᵛ': 'v', 'ʷ': 'w', 'ʸ': 'y' };
  const SUB = { '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9', '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')',
    'ₐ': 'a', 'ₑ': 'e', 'ₒ': 'o', 'ₓ': 'x', 'ₕ': 'h', 'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₚ': 'p', 'ₛ': 's', 'ₜ': 't', 'ᵢ': 'i', 'ⱼ': 'j', 'ᵣ': 'r', 'ᵤ': 'u', 'ᵥ': 'v', 'ᵦ': 'beta', 'ᵧ': 'gamma', 'ᵨ': 'rho', 'ᵩ': 'phi', 'ᵪ': 'chi' };
  const SUP_RE = new RegExp('[' + Object.keys(SUP).join('') + ']+', 'g');
  const SUB_RE = new RegExp('[' + Object.keys(SUB).join('') + ']+', 'g');
  const SUB_CHARS = Object.keys(SUB).join('');
  const SUP_THEN_SUB = new RegExp('([' + Object.keys(SUP).join('') + ']+)(_[{][^}]*[}]|_[A-Za-z0-9α-ωΑ-Ω]+)', 'g');   // σ²_batch, Vᵀ_k
  const NUMWORD = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];

  const subText = run => run.split('').map(ch => SUB[ch] || ch).join(' ').replace(/(\d) (?=\d)/g, '$1');
  function supText(run) {
    const t = run.split('').map(ch => SUP[ch] || ch).join('');
    if (t === '2') return ' squared';
    if (t === '3') return ' cubed';
    if (t === 'T') return ' transpose';
    if (t === '-1') return ' inverse';
    if (t === '-1T' || t === 'T-1') return ' inverse transpose';
    return ' to the power ' + t.replace(/^-/, 'minus ').replace(/\(|\)/g, ' ');
  }

  // Shared rewrites for formulas and for math inside prose.
  function symbols(s, formula) {
    s = String(s).normalize('NFD');
    s = s.replace(SUP_THEN_SUB, '$2$1');                                           // σ²_batch -> σ_batch², read "sigma sub batch squared"
    s = s.replace(/(?<![A-Za-z0-9])[A-Za-z0-9]+(?:_[A-Za-z0-9]+){2,}(?![A-Za-z0-9])/g, m => m.replace(/_/g, ' '));   // hh_e_ll_lo, tokens_per_second
    if (formula) {
      s = s.replace(/([A-Za-z0-9Ͱ-Ͽ)\]])\*(?=\s|$|[,)\]ᵀ]|_)/g, '$1 star ');
      for (let k = 0; k < 3; k++) s = s.replace(/(?<![A-Za-z0-9])([A-Za-z]{2,})_([A-Za-z]{2,})(?![A-Za-z0-9])/g, '$1 $2');   // cos_sim
    }
    s = s.replace(/∇_\{([^}]*)\}|∇_([A-Za-z0-9Ͱ-Ͽℓ]+)/g, (m, a, b) => ' the gradient with respect to ' + (a || b) + ' of ')
      .replace(/gradient\s+∂\s*([^\s/(),]+)\s*\/\s*∂\s*([^\s,()\]]+)/g, 'gradient of $1 with respect to $2')
      .replace(/([A-Za-z\u0370-\u03ff])\u0302/g, '$1 hat ').replace(/([A-Za-z\u0370-\u03ff])\u0303/g, '$1 tilde ').replace(/([A-Za-z\u0370-\u03ff])\u0304/g, '$1 bar ')
      .replace(/([A-Za-z\u0370-\u03ff])\u0307/g, '$1 dot ').replace(/[\u0300-\u036f]/g, '').replace(/\b(hat|bar|tilde|dot) ?_\{?([A-Za-z0-9\u0370-\u03ff]+)\}?/g, '$1 sub $2 ')
      .replace(/\blog₂|\blog_2\b|\blog_\{2\}/g, 'log base 2').replace(/\blog₁₀|\blog_10\b|\blog_\{10\}/g, 'log base 10')
      .replace(/\(([^()‖]*?)\s*‖\s*([^()‖]*?)\)/g, '($1, $2)')
      .replace(/‖([^‖()]{1,24})‖\s*_F(²?)/g, (m, inner, sq) => ' the ' + (sq ? 'squared ' : '') + 'Frobenius norm' + (/^[\s….⋯]*$/.test(inner) ? ' ' : ' of ' + inner + ' '))
      .replace(/‖([^‖()]{1,24})‖\s*_?∞/g, ' the infinity norm of $1 ')
      .replace(/‖([^‖()]{1,24})‖\s*(?:₁|_1)/g, ' the L1 norm of $1 ').replace(/‖([^‖()]{1,24})‖\s*(?:₂|_2)/g, ' the L2 norm of $1 ').replace(/‖([^‖()]{1,24})‖²/g, ' the squared norm of $1 ').replace(/‖([^‖()]{1,24})‖/g, ' the norm of $1 ').replace(/\s*‖\s*/g, ', ')
      .replace(/⌈([^⌉]+)⌉/g, ' the ceiling of $1 ').replace(/⌊([^⌋]+)⌋/g, ' the floor of $1 ')
      .replace(/∂\s*([^\s/(),]+)\s*\/\s*∂\s*([^\s,()\]]+)/g, ' the partial derivative of $1 with respect to $2 ')
      .replace(/\bd([A-Za-z])\s*\/\s*d([A-Za-z])\b/g, ' the derivative of $1 with respect to $2 ')
      .replace(/([Σ∑Π∏])_\{([^}]*)\}/g, (m, op, over) => (op === 'Σ' || op === '∑' ? ' the sum over ' : ' the product over ') + over + ' of ')
      .replace(/([Σ∑Π∏])_([A-Za-z0-9α-ω]+)/g, (m, op, over) => (op === 'Σ' || op === '∑' ? ' the sum over ' : ' the product over ') + over + ' of ')
      .replace(new RegExp('([Σ∑Π∏])([' + SUB_CHARS + ']+)', 'g'), (m, op, run) => (op === 'Σ' || op === '∑' ? ' the sum over ' : ' the product over ') + subText(run) + ' of ')
      .replace(/[Σ∑]/g, ' the sum of ').replace(/[∏]/g, ' the product of ')
      .replace(/\barg\s?max_\{([^}]*)\}|\barg\s?max_([A-Za-z0-9α-ω]+)/g, (m, a, b) => ' the arg max over ' + (a || b) + ' of ')
      .replace(/\barg\s?min_\{([^}]*)\}|\barg\s?min_([A-Za-z0-9α-ω]+)/g, (m, a, b) => ' the arg min over ' + (a || b) + ' of ')
      .replace(/\bmax_([A-Za-z0-9α-ω]+)/g, ' the max over $1 of ').replace(/\bmin_([A-Za-z0-9α-ω]+)/g, ' the min over $1 of ')
      .replace(/\bE_([A-Za-z0-9α-ω]+)\s*\[/g, ' the expected value over $1 of [').replace(/\bE\[/g, ' the expected value of [').replace(/\bVar\[/g, ' the variance of [')
      .replace(/(?<![A-Za-z])the\s+(?=√)/g, '').replace(/√\s*\(/g, ' the square root of (')
      .replace(/√\s*([A-Za-z0-9α-ω.]+(?:_[{][^}]*[}]|_[A-Za-z0-9α-ω]+)?)/g, ' the square root of $1 ')   // √d_k keeps its subscript
      .replace(SUP_RE, run => supText(run) + ' ')
      .replace(SUB_RE, run => ' sub ' + subText(run) + ' ')
      .replace(/′′|″/g, ' double prime ').replace(/′/g, ' prime ');
    // ASCII subscripts: formulas take any x_y; prose only single-letter bases (x_i), not identifiers such as n_estimators
    let prev;
    do {
      prev = s;
      s = formula ? s.replace(/_\{([^{}]*)\}/g, ' sub $1 ').replace(/_\(([^()]*)\)/g, ' sub ($1) ').replace(/([A-Za-z0-9α-ωΑ-Ωℓ)])_([A-Za-z0-9α-ωΑ-Ωℓ]+)/g, '$1 sub $2 ')
        : s.replace(/(^|[^A-Za-z0-9_])([A-Za-zͰ-Ͽℓ])_\(([^()]*)\)/g, '$1$2 sub $3 ').replace(/(^|[^A-Za-z0-9_])([A-Za-zͰ-Ͽℓ])_\{?([A-Za-z0-9α-ωΑ-Ω]+)\}?/g, '$1$2 sub $3 ');
    } while (s !== prev);
    // powers after subscripts, so that c^s_j reads "c to the power s sub j"
    s = s.replace(/([A-Za-z0-9)\]α-ωΑ-Ω*])\s?\^\{([^}]*)\}/g, '$1 to the power $2 ').replace(/([A-Za-z0-9)\]α-ωΑ-Ω*])\s?\^\(([^)]*)\)/g, '$1 to the power $2 ')
      .replace(/([A-Za-z0-9)\]α-ωΑ-Ω*])\s?\^\s?(-?[A-Za-z0-9.α-ωΑ-Ω]+)/g, (m, b, e) => b + (e === '2' ? ' squared ' : e === '3' ? ' cubed ' : e === 'T' ? ' transpose ' : ' to the power ' + e.replace(/^-/, 'minus ') + ' '));
    s = s.replace(/(\d)\s*×\s*10(?=\s*to the power)/g, '$1 times 10')
      .replace(/(^|[\s(])(\w{1,5})\s*×\s*(\w{1,5})(?=$|[\s),.;:])/g, '$1$2 by $3')
      .replace(/×/g, ' times ').replace(/·/g, ' times ').replace(/÷/g, ' divided by ').replace(/±/g, ' plus or minus ').replace(/−/g, ' minus ')
      .replace(/≈/g, ' is approximately ').replace(/≠/g, ' is not equal to ').replace(/≤/g, ' is at most ').replace(/≥/g, ' is at least ')
      .replace(/≲/g, ' is at most about ').replace(/≪/g, ' is much smaller than ').replace(/≫/g, ' is much larger than ').replace(/∝/g, ' is proportional to ')
      .replace(/≻/g, ' is preferred over ').replace(/≡/g, ' is defined as ')
      .replace(/⇒/g, ' implies ').replace(/⇔/g, ' if and only if ').replace(/↔/g, ' to and from ').replace(/←/g, ' becomes ').replace(/→/g, ' to ').replace(/↑/g, ' up ').replace(/↓/g, ' down ')
      .replace(/∈/g, ' in ').replace(/∉/g, ' not in ').replace(/⊂/g, ' a subset of ').replace(/∪/g, ' union ').replace(/∩/g, ' intersection ').replace(/∅/g, ' the empty set ')
      .replace(/∞/g, ' infinity ').replace(/∂/g, ' partial ').replace(/∇/g, ' the gradient of ').replace(/∫/g, ' the integral of ')
      .replace(/⊙/g, ' times, element by element, ').replace(/⊕/g, ' plus ').replace(/∘/g, ' composed with ').replace(/∀/g, ' for all ').replace(/∃/g, ' there is ')
      .replace(/,\s*[…⋯]\s*,/g, ' through ').replace(/[…⋯]/g, ' and so on ')
      .replace(/½/g, ' one half ').replace(/¼/g, ' one quarter ').replace(/¾/g, ' three quarters ').replace(/⅛/g, ' one eighth ')
      .replace(/°/g, ' degrees ').replace(/ℓ/g, ' ell ').replace(/ℒ/g, ' L ')
      .replace(/[─►•✓✗]/g, ' ');
    s = s.replace(/[Ͱ-Ͽ]/g, ch => GREEK[ch] ? ' ' + GREEK[ch] + ' ' : ch);
    // underscores still left belong to names such as _backward or __init__: read the name alone
    s = s.replace(/(^|[\s(,;:])_+(?=[A-Za-z0-9])/g, '$1').replace(/(?<=[A-Za-z0-9])_+(?=$|[\s),.;:])/g, '');
    return s;
  }

  function math(f) {
    let s = String(f || '').replace(/(^|[^\w.])1\[/g, '$1 the indicator that [');           // 1[π(x) = a]
    s = s.replace(/\(\s*·\s*\|\s*([^)]+)\)/g, ' given $1');
    s = s.replace(/\s\|\s/g, ' given ');                                     // p(x | θ): conditional probability
    s = symbols(s, true);
    s = s.replace(/\|([^|\s][^|]*?)\|/g, ' the absolute value of $1 ')
      .replace(/(\d)\s*\.\.\s*(\w)/g, '$1 to $2').replace(/\.\./g, ' to ')
      .replace(/@/g, ' matrix-multiplied by ')
      .replace(/([A-Za-z0-9α-ω)\]])\*(?=\s|$|[,)\]]|transpose|sub)/g, '$1 star ').replace(/\*/g, ' times ')
      .replace(/(^|[\s(=,])-(?=[\w(])/g, '$1minus ').replace(/\s-\s/g, ' minus ')
      .replace(/\+/g, ' plus ').replace(/=/g, ' equals ').replace(/</g, ' is less than ').replace(/>/g, ' is greater than ')
      .replace(/~/g, ' is distributed as ')
      .replace(/\//g, ' over ')
      .replace(/[{}\[\]]/g, ' ').replace(/\bs\.t\./g, 'subject to').replace(/\bw\.r\.t\./g, 'with respect to');
    return tidy(s);
  }

  function tidy(s) {
    return String(s).replace(/\s+/g, ' ').replace(/\s+([,.;:!?)])/g, '$1').replace(/\(\s+/g, '(').replace(/\(\s*\)/g, '').trim();
  }

  // ---------------------------------------------------------------- prose
  // Python operators written as inline code, read as words.
  const OPS = { '**': 'to the power', '//': 'integer division', '%': 'remainder', '*': 'times', '/': 'divided by', '+': 'plus', '-': 'minus', '@': 'matrix multiplication', '==': 'equals', '!=': 'not equal', '<=': 'at most', '>=': 'at least', '+=': 'plus equals', '-=': 'minus equals' };
  function stripInline(s) {
    const codes = [];
    let t = String(s)
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/\[([^\]]+)\]\([^)\s]+\)/g, '$1')
      .replace(/`([^`]+)`/g, (m, c) => { codes.push(c); return '\u0001' + (codes.length - 1) + '\u0001'; });   // code first: it may contain * and _
    t = t.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/(^|[\s(“"'])\*([^*\n]+)\*(?=[\s).,;:!?”"']|$)/g, '$1$2').replace(/__([^_]+)__/g, '$1')
      .replace(/<\/?[a-z][^>]*>/gi, ' ');
    return t.replace(/\u0001(\d+)\u0001/g, (m, i) => {
      const c = codes[Number(i)].trim();
      if (OPS[c]) return ' ' + OPS[c] + ' ';
      return ' ' + c.replace(/_/g, ' ').replace(/\(\)/g, '').replace(/[<>]/g, ' ') + ' ';
    });
  }
  function text(s) {
    let t = stripInline(s);
    t = t.replace(/\be\.g\.,?/g, 'for example,').replace(/\bi\.e\.,?/g, 'that is,').replace(/\bvs\./g, 'versus').replace(/\bapprox\./g, 'approximately')
      .replace(/\bw\.r\.t\./g, 'with respect to').replace(/\bs\.t\./g, 'subject to').replace(/\betc\./g, 'and so on.')
      .replace(/\b(about|around|roughly)\s*~\s*/gi, '$1 ').replace(/~\s?(?=\d)/g, 'about ').replace(/\(\s*([\w.]+)\s*\/\s*([\w.]+)\s*\)/g, '($1 over $2)')
      .replace(/(\d)\s?[kK]\b(?!\w)/g, '$1 thousand').replace(/(\d)\s?M\b(?= (?:params|parameters|tokens|users|rows|images|examples|features|latents))/g, '$1 million')
      .replace(/(\d)\s?B\b(?= (?:params|parameters|tokens))/g, '$1 billion');
    t = t.replace(/\(\s*·\s*\|\s*([^)]+)\)/g, ' given $1');
    for (let k = 0; k < 4; k++) t = t.replace(/\b([A-Za-z][A-Za-z0-9]+)_([A-Za-z0-9]+)/g, '$1 $2');   // identifiers: is_missing, cross_val_score
    t = t      .replace(/\|([^|\s][^|]{0,24}?)\|/g, ' the absolute value of $1 ');                 // |x₁|
    t = symbols(t, false);
    t = t.replace(/\s-\s/g, ' minus ').replace(/ = /g, ' equals ').replace(/ \/ /g, ' over ').replace(/[{}]/g, ' ').replace(/\s\^\s?/g, ' to the power ');
    return tidy(t);
  }
  const sentence = s => { s = tidy(s); return !s ? '' : /[.!?:;]$/.test(s) ? s : s + '.'; };

  // ---------------------------------------------------------------- markdown -> speech
  const CALLOUT = { tip: 'Tip', warn: 'Common mistake', danger: 'Watch out', real: 'In the real world', note: 'Note', key: 'Key idea', analogy: 'Analogy', math: 'The math' };
  function legend(lines, label) {
    const items = [];
    for (const raw of lines) {
      if (!raw.trim()) continue;
      if (/^\s{2,}\S/.test(raw) && items.length) { items[items.length - 1].def += ' ' + raw.trim(); continue; }
      const line = raw.trim().replace(/^[-*]\s+/, '');
      let k = line.indexOf(' — '), w = 3;
      if (k < 0) { k = line.indexOf(': '); w = 2; }
      if (k > 0) items.push({ sym: line.slice(0, k).trim(), def: line.slice(k + w).trim() });
      else items.push({ note: line });
    }
    const SYMNAME = { '√': 'The root sign', 'Σ': 'Capital sigma', '∑': 'Capital sigma', 'Π': 'Capital pi', '∏': 'Capital pi', '∂': 'The curly d', '∇': 'Nabla', '‖ ‖': 'The double bars', '| |': 'The bars', '·': 'The dot', '×': 'The times sign', '⊙': 'The circled dot', '∈': 'The symbol in', '~': 'The tilde' };
    const symName = sym => { const s0 = stripInline(sym).trim(); return SYMNAME[s0] || math(s0) || s0; };
    const parts = items.map(it => it.note ? sentence(text(it.note)) : sentence(symName(it.sym) + ' is ' + text(it.def)));
    return (label ? sentence(text(label)) + ' ' : 'Where: ') + parts.join(' ');
  }
  function table(header, rows) {
    const cells = r => r.map(c => text(c));
    const h = cells(header);
    const out = ['Table, with columns ' + h.filter(Boolean).join(', ') + '.'];
    rows.slice(0, 10).forEach((r, i) => {
      const c = cells(r);
      out.push(sentence((rows.length > 1 ? 'Row ' + (i + 1) + ': ' : '') + c.map((v, j) => (h[j] && j > 0 ? h[j] + ', ' : '') + v).filter(x => x.trim() && x.trim() !== ',').join('; ')));
    });
    if (rows.length > 10) out.push(rows.length - 10 + ' more rows are on the page.');
    return out.join(' ');
  }
  function md(src) {
    if (!src) return '';
    const lines = String(src).replace(/\r\n/g, '\n').split('\n');
    const out = []; let para = [];
    const flush = () => { if (para.length) { out.push(sentence(text(para.join(' ')))); para = []; } };
    let i = 0;
    while (i < lines.length) {
      const t = lines[i].trim();
      if (!t) { flush(); i++; continue; }
      if (/^```/.test(t)) {
        flush();
        const lang = t.slice(3).trim(); i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i].trim())) i++;
        i++;
        out.push(lang === 'output' ? 'The output is shown on the page.' : 'A code example is shown on the page.');
        continue;
      }
      if (/^\$\$/.test(t)) {
        flush();
        let body = [];
        if (t.length > 4 && /\$\$$/.test(t)) { body = [t.slice(2, -2)]; i++; }
        else {
          if (t.slice(2).trim()) body.push(t.slice(2).trim());
          i++;
          while (i < lines.length && !/\$\$\s*$/.test(lines[i].trim())) { body.push(lines[i].trim()); i++; }
          if (i < lines.length) { const last = lines[i].trim().replace(/\$\$\s*$/, '').trim(); if (last) body.push(last); i++; }
        }
        const parts = body.join('  ').split(/\s{2,}/).map(x => x.trim()).filter(Boolean).map(x => sentence(math(stripInline(x))));
        out.push((parts.length > 1 ? 'Formulas: ' : 'Formula: ') + parts.join(' Next, '));
        continue;
      }
      let m = t.match(/^:::where\s*(.*)$/);
      if (m) {
        flush();
        const buf = []; i++;
        while (i < lines.length && !/^:::\s*$/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
        i++;
        out.push(legend(buf, m[1]));
        continue;
      }
      m = t.match(/^:::(tip|warn|danger|real|note|key|analogy|math)\s*(.*)$/);
      if (m) {
        flush();
        const buf = []; i++;
        while (i < lines.length && !/^:::\s*$/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
        i++;
        out.push(sentence(text(m[2] || CALLOUT[m[1]])) + ' ' + md(buf.join('\n')));
        continue;
      }
      m = t.match(/^#{1,4}\s+(.*)$/);
      if (m) { flush(); out.push(sentence(text(m[1]))); i++; continue; }
      if (/^---+$/.test(t)) { flush(); i++; continue; }
      if (/^>\s?/.test(t)) { flush(); const buf = []; while (i < lines.length && /^>\s?/.test(lines[i].trim())) { buf.push(lines[i].trim().replace(/^>\s?/, '')); i++; } out.push(md(buf.join('\n'))); continue; }
      if (/^\|.*\|$/.test(t) && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1].trim())) {
        flush();
        const header = t.slice(1, -1).split('|').map(c => c.trim()); i += 2;
        const rows = [];
        while (i < lines.length && /^\|.*\|$/.test(lines[i].trim())) { rows.push(lines[i].trim().slice(1, -1).split('|').map(c => c.trim())); i++; }
        out.push(table(header, rows));
        continue;
      }
      m = t.match(/^([-*]|\d+\.)\s+(.*)$/);
      if (m) {
        flush();
        while (i < lines.length) {
          const lm = lines[i].match(/^\s*([-*]|\d+\.)\s+(.*)$/);
          if (lm) { out.push(sentence(text(lm[2]))); i++; continue; }
          if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && out.length) { out[out.length - 1] = sentence(out[out.length - 1].replace(/[.:;]$/, '') + ' ' + text(lines[i].trim())); i++; continue; }
          break;
        }
        continue;
      }
      para.push(t); i++;
    }
    flush();
    return out.filter(Boolean).join(' ');
  }

  // ---------------------------------------------------------------- questions
  const LETTER = i => String.fromCharCode(65 + i);
  function question(q, lead) {
    if (!q) return '';
    const parts = [sentence(lead) + ' ' + md(q.q)];
    let answer = '';
    if (q.type === 'mcq' || q.type === 'multi') {
      parts.push(q.options.map((o, i) => 'Option ' + LETTER(i) + ': ' + sentence(text(o))).join(' '));
      if (q.type === 'multi') parts.push('More than one option can be right.');
      const idx = q.type === 'mcq' ? [q.answer] : q.answer;
      answer = (idx.length > 1 ? 'The answers: ' : 'The answer: ') + idx.map(i => 'option ' + LETTER(i) + ', ' + text(q.options[i])).join('; and ');
    } else if (q.type === 'tf') {
      parts.push('True or false?');
      answer = 'The answer: ' + (q.answer ? 'true' : 'false');
    } else if (q.type === 'fill') {
      answer = 'The answer: ' + text(q.display || (Array.isArray(q.answer) ? q.answer.filter(x => !/^\/.*\/[a-z]*$/.test(String(x)))[0] : q.answer) || '');
    } else if (q.type === 'order') {
      parts.push('Put these in the right order.');
      answer = 'The right order: ' + q.items.map((it, i) => (i + 1) + ', ' + text(it)).join('; ');
    } else if (q.type === 'match') {
      parts.push('Match each item on the left with one on the right. On the left: ' + q.pairs.map(p => text(p[0])).join('; ') + '. On the right: ' + q.pairs.map(p => text(p[1])).join('; ') + '.');
      answer = 'The pairs: ' + q.pairs.map(p => text(p[0]) + ' goes with ' + text(p[1])).join('; ');
    }
    return parts.join(' ') + ' ' + PAUSE + ' ' + sentence(answer) + (q.explain ? ' ' + md(q.explain) : '');
  }

  // ---------------------------------------------------------------- charts
  function sup(n) { return String(n).split('').map(ch => ({ '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' })[ch] || ch).join(''); }
  function num(v) {
    if (v == null || !isFinite(v)) return 'none';
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-3)) { const e = Math.floor(Math.log10(a)); const m = +(v / Math.pow(10, e)).toFixed(1); return (m === 1 ? '' : m === -1 ? '−' : m + ' × ') + '10' + sup(e); }
    if (a >= 1000) return Math.round(v).toLocaleString('en-US');
    return String(+v.toPrecision(3));
  }
  const clean = l => String(l || '').replace(/,?\s*log scale/gi, '').replace(/\(\s*\)/g, '').replace(/\s+/g, ' ').trim();
  const short = l => clean(l).split(/[;,]/)[0].replace(/\s*\([^)]*\)?/g, '').replace(/[\s:=/]+$/, '').trim();   // "k (number of neighbours)" -> "k"
  const seriesName = (s, k, n) => s.name ? s.name : n === 1 ? 'The line' : 'Line ' + (k + 1);
  const brief = (s, k, n) => { const nm = seriesName(s, k, n); const head = nm.split(': ')[0]; return head.length >= 3 && head.length < nm.length ? head : nm; };
  function points(s, spec) {
    const xs = s.x || spec.x || [];
    const out = [];
    s.y.forEach((y, i) => { const x = xs[i]; if (y != null && isFinite(y) && (spec.kind === 'bar' || (x != null && isFinite(x)))) out.push({ x, y, i }); });
    return out;
  }
  function trend(p) {
    const ys = p.map(q => q.y), n = ys.length;
    const lo = Math.min(...ys), hi = Math.max(...ys), range = hi - lo, scale = Math.max(Math.abs(hi), Math.abs(lo), 1e-12);
    if (range <= 0.03 * scale) return { kind: 'flat' };
    const iMax = ys.indexOf(hi), iMin = ys.indexOf(lo), first = ys[0], last = ys[n - 1];
    const peak = iMax > 0 && iMax < n - 1 && hi - Math.max(first, last) > 0.15 * range;
    const valley = iMin > 0 && iMin < n - 1 && Math.min(first, last) - lo > 0.15 * range;
    let up = 0, down = 0, flips = 0, lastSign = 0;
    for (let i = 1; i < n; i++) {
      const d = ys[i] - ys[i - 1], sg = d > 0.02 * range ? 1 : d < -0.02 * range ? -1 : 0;
      if (sg > 0) up++; else if (sg < 0) down++;
      if (sg && lastSign && sg !== lastSign) flips++;
      if (sg) lastSign = sg;
    }
    if (flips > Math.max(4, 0.25 * n)) return { kind: 'varies', lo, hi };          // a noisy trace, not a single peak
    const tail = ys[Math.floor(n * 2 / 3)], levels = n >= 6 && Math.abs(last - tail) < 0.1 * range;
    if (peak && !valley) return { kind: 'peak', at: p[iMax] };
    if (valley && !peak) return { kind: 'valley', at: p[iMin] };
    if (down === 0 || (last - first > 0.5 * range && up >= 2 * down)) return { kind: down === 0 ? 'rises' : 'mostly rises', levels };
    if (up === 0 || (first - last > 0.5 * range && down >= 2 * up)) return { kind: up === 0 ? 'falls' : 'mostly falls', levels };
    return { kind: 'varies', lo, hi };
  }
  function describeLine(s, k, n, spec) {
    const p = points(s, spec); const name = seriesName(s, k, n);
    if (!p.length) return '';
    if (p.length === 1) return name + ' is a single point: ' + num(p[0].y) + ' at ' + num(p[0].x) + '.';
    const t = trend(p), f = p[0], l = p[p.length - 1], xs = short(spec.xLabel) || 'x';
    switch (t.kind) {
      case 'flat': return name + ' stays near ' + num(p.reduce((a, q) => a + q.y, 0) / p.length) + ' throughout.';
      case 'peak': return name + ' rises from ' + num(f.y) + ' to a peak of ' + num(t.at.y) + ' at ' + xs + ' ' + num(t.at.x) + ', then falls to ' + num(l.y) + '.';
      case 'valley': return name + ' falls from ' + num(f.y) + ' to a low of ' + num(t.at.y) + ' at ' + xs + ' ' + num(t.at.x) + ', then rises to ' + num(l.y) + '.';
      case 'rises': case 'mostly rises': return name + ' ' + (t.kind === 'rises' ? 'rises' : 'rises overall') + ' from ' + num(f.y) + ' to ' + num(l.y) + (t.levels ? ', levelling off towards the end' : '') + '.';
      case 'falls': case 'mostly falls': return name + ' ' + (t.kind === 'falls' ? 'falls' : 'falls overall') + ' from ' + num(f.y) + ' to ' + num(l.y) + (t.levels ? ', levelling off towards the end' : '') + '.';
      default: return name + ' moves up and down between ' + num(t.lo) + ' and ' + num(t.hi) + ', ending at ' + num(l.y) + '.';
    }
  }
  function corr(p) {
    const n = p.length; if (n < 3) return 0;
    const mx = p.reduce((a, q) => a + q.x, 0) / n, my = p.reduce((a, q) => a + q.y, 0) / n;
    let sxy = 0, sxx = 0, syy = 0; for (const q of p) { sxy += (q.x - mx) * (q.y - my); sxx += (q.x - mx) ** 2; syy += (q.y - my) ** 2; }
    return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
  }
  function chart(spec) {
    if (!spec || !Array.isArray(spec.series)) return '';
    const kind = spec.kind || 'line';
    const S = spec.series.filter(s => s && Array.isArray(s.y));
    const out = [];
    const xl = clean(spec.xLabel), yl = clean(spec.yLabel), xs = short(spec.xLabel) || 'x';
    const refs = () => {
      for (const h of spec.hlines || []) out.push('A horizontal reference line marks ' + (h.label ? text(h.label) + ', at ' : '') + num(h.y) + '.');
      for (const v of spec.vlines || []) out.push('A vertical reference line marks ' + (v.label ? text(v.label) + ', at ' : '') + xs + ' ' + num(v.x) + '.');
    };
    if (kind === 'bar') {
      const cats = (spec.x || []).map(String);
      const catName = c => /^-?[\d.]+$/.test(c) ? xs + ' ' + c : c;
      out.push('A bar chart of ' + (yl || 'values') + ', with one ' + (S.length > 1 ? 'group of bars' : 'bar') + ' for each ' + (xs || 'category') + ' (' + cats.length + ' in all).');
      S.slice(0, 4).forEach((s, k) => {
        const name = s.name || (S.length > 1 ? 'Series ' + (k + 1) : 'The bars');
        const vals = cats.map((c, i) => ({ c, v: s.y[i] })).filter(o => o.v != null && isFinite(o.v));
        if (!vals.length) return;
        if (vals.length <= 8) out.push(name + ': ' + vals.map(o => catName(o.c) + ', ' + num(o.v)).join('; ') + '.');
        else {
          const hi = vals.reduce((a, o) => (o.v > a.v ? o : a)), lo = vals.reduce((a, o) => (o.v < a.v ? o : a));
          out.push(name + ': highest at ' + catName(hi.c) + ' (' + num(hi.v) + '), lowest at ' + catName(lo.c) + ' (' + num(lo.v) + ').');
          for (const h of spec.hlines || []) { const above = vals.filter(o => o.v > h.y).length; out.push(above + ' of the ' + vals.length + ' bars are above ' + (h.label ? text(h.label) : num(h.y)) + '.'); }
        }
      });
      if (S.length > 4) out.push('There are ' + (S.length - 4) + ' more series; the numbers are in the table under the chart.');
      if (!(cats.length > 8)) refs(); else for (const v of spec.vlines || []) out.push('A vertical reference line marks ' + text(v.label || '') + '.');
      return out.join(' ');
    }
    const all = []; S.forEach(s => points(s, spec).forEach(q => all.push(q)));
    if (!all.length) return '';
    const xsAll = all.map(q => q.x);
    out.push('Horizontal axis: ' + (xl || 'x') + (spec.xLog ? ', on a log scale' : '') + ', from ' + num(Math.min(...xsAll)) + ' to ' + num(Math.max(...xsAll)) + '.');
    out.push('Vertical axis: ' + (yl || 'y') + (spec.yLog ? ', on a log scale' : '') + '.');
    const lines = S.filter(s => !(kind === 'scatter' || s.scatter)), dots = S.filter(s => kind === 'scatter' || s.scatter);
    if (lines.length) {
      if (lines.length > 1) out.push(String(NUMWORD[lines.length] || lines.length).replace(/^./, ch => ch.toUpperCase()) + ' lines.');
      lines.slice(0, 6).forEach((s, k) => out.push(describeLine(s, k, lines.length, spec)));
      if (lines.length > 6) out.push('And ' + (lines.length - 6) + ' more lines; the numbers are in the table under the chart.');
      const shared = lines.filter(s => !s.x);
      if (shared.length >= 2) {
        const xsS = spec.x || [];
        let li = xsS.length - 1;
        while (li > 0 && shared.some(s => !(s.y[li] != null && isFinite(s.y[li])))) li--;
        const ends = shared.map((s, k) => ({ name: brief(s, k, shared.length), y: s.y[li] })).filter(o => o.y != null && isFinite(o.y)).sort((a, b) => b.y - a.y);
        if (ends.length >= 2 && ends[0].y !== ends[ends.length - 1].y) {
          out.push(ends.length === 2
            ? 'At ' + xs + ' ' + num(xsS[li]) + ', ' + ends[0].name + ' is above ' + ends[1].name + ' (' + num(ends[0].y) + ' against ' + num(ends[1].y) + ').'
            : 'At ' + xs + ' ' + num(xsS[li]) + ', ' + ends[0].name + ' is highest (' + num(ends[0].y) + ') and ' + ends[ends.length - 1].name + ' lowest (' + num(ends[ends.length - 1].y) + ').');
        }
        let said = 0;
        for (let a = 0; a < Math.min(shared.length, 4) && said < 2; a++) for (let b = a + 1; b < Math.min(shared.length, 4) && said < 2; b++) {
          const A = shared[a].y, B = shared[b].y; let prevSign = 0, prevI = -1;
          const ok = i => A[i] != null && B[i] != null && isFinite(A[i]) && isFinite(B[i]);
          const vals = []; for (let i = 0; i < xsS.length; i++) if (ok(i)) vals.push(A[i], B[i]);
          const tiny = 0.02 * (Math.max(...vals) - Math.min(...vals));      // gaps smaller than this are noise, not a crossing
          const signAt = i => { const d = A[i] - B[i]; return Math.abs(d) <= tiny ? 0 : d > 0 ? 1 : -1; };
          const lasts = (i, sg) => { let n = 0; for (let j = i; j < xsS.length && n < 3; j++) if (ok(j)) { const s2 = signAt(j); if (s2 && s2 !== sg) return false; n++; } return true; };
          for (let i = 0; i < xsS.length; i++) {
            if (!ok(i)) continue;
            const d = A[i] - B[i], sg = signAt(i);
            if (sg && prevSign && sg !== prevSign && i > 0 && lasts(i, sg)) {
              const dPrev = A[prevI] - B[prevI], t = dPrev / (dPrev - d), xc = xsS[prevI] + t * (xsS[i] - xsS[prevI]);
              out.push(brief(shared[a], a, shared.length) + ' and ' + brief(shared[b], b, shared.length) + ' cross near ' + xs + ' ' + num(xc) + '.');
              said++; break;
            }
            if (sg) { prevSign = sg; prevI = i; }
          }
        }
      }
    }
    dots.slice(0, 4).forEach((s, k) => {
      const p = points(s, spec); if (!p.length) return;
      const r = corr(p), name = s.name || (dots.length > 1 ? 'Point set ' + (k + 1) : 'The points');
      const ys = p.map(q => q.y), xsP = p.map(q => q.x);
      out.push(name + ': ' + p.length + ' point' + (p.length > 1 ? 's' : '') + ', ' + xs + ' from ' + num(Math.min(...xsP)) + ' to ' + num(Math.max(...xsP)) + ', ' + (short(spec.yLabel) || 'y') + ' from ' + num(Math.min(...ys)) + ' to ' + num(Math.max(...ys)) + (p.length >= 3 ? (r > 0.5 ? '; they trend upward' : r < -0.5 ? '; they trend downward' : '; with no clear trend') : '') + '.');
    });
    refs();
    return out.join(' ');
  }

  // ---------------------------------------------------------------- lessons
  function diagramText(name) {
    const comp = MLA.components && MLA.components['demo-' + name];
    const tpl = comp && typeof comp.template === 'string' ? comp.template : '';
    const m = tpl.match(/aria-label="([^"]+)"/);
    return m ? sentence(text(m[1])) : '';
  }
  function section(c, s, i) {
    const title = s.title || (s.type === 'check' ? 'Quick check' : 'Section ' + (i + 1));
    const parts = [];
    if (s.type === 'text') parts.push(sentence(text(title)), md(s.md));
    else if (s.type === 'code') {
      parts.push(sentence('Code example: ' + text(title)), md(s.md));
      if (s.predict) parts.push(question(s.predict, 'Before running it, predict the result.'));
      parts.push(s.lang === 'pytorch' || s.run === false ? 'The code is on the page, to run on your own computer.' : 'The code is on the page, where you can run it.');
      parts.push(md(s.after));
    } else if (s.type === 'demo') {
      const dg = String(s.demo || '').startsWith('dg-');
      const walk = String(s.demo || '').startsWith('walk-') && MLA.walks && MLA.walks[s.demo.slice(5)];
      parts.push(sentence((dg ? 'Diagram: ' : walk ? 'Step-by-step walkthrough: ' : 'Interactive demo: ') + text(title)), md(s.md));
      parts.push(dg ? diagramText(s.demo) : walk ? text(walk.describe()) + ' On the page, you can step through it, with the numbers of every step.' : 'On the page, you can try it yourself.');
      parts.push(md(s.after));
    } else if (s.type === 'chart') parts.push(sentence('Chart: ' + text(title)), md(s.md), text(chart(s.chart)), md(s.after));
    else if (s.type === 'check') parts.push(question(s.question, 'Quick check.'));
    return { key: s.id, title, text: parts.filter(Boolean).join(' ') };
  }
  function project(p) {
    return sentence(text(p.title)) + ' ' + sentence('A ' + p.level + ' project of about ' + p.hours + ' hours') + ' Goal: ' + md(p.goal) + ' Data and tools: ' + md(p.data) +
      ' The steps. ' + p.steps.map((s, i) => 'Step ' + (i + 1) + '. ' + md(s)).join(' ') + ' How you know it is done: ' + md(p.done) +
      ' To go further: ' + p.stretch.map(s => md(s)).join(' ');
  }

  function lesson(c) {
    const items = [{ key: 'intro', title: c.title, text: sentence('Lesson: ' + text(c.title)) + ' ' + sentence(text(c.summary || '')) }];
    if (c.why) items.push({ key: 'why', title: 'Why this matters', text: 'Why this matters. ' + md(c.why) });
    if (c.objectives && c.objectives.length) items.push({ key: 'objectives', title: 'By the end you will be able to', text: 'By the end of this lesson you will be able to: ' + c.objectives.map(o => text(o).replace(/[.;]$/, '')).join('; ') + '.' });
    c.sections.forEach((s, i) => items.push(section(c, s, i)));
    if (c.when) items.push({ key: 'when', title: 'When to use it', text: 'When to use it, and when not to. ' + md(c.when) });
    if (c.pitfalls) items.push({ key: 'pitfalls', title: 'Common pitfalls', text: 'Common pitfalls. ' + md(c.pitfalls) });
    if (c.takeaways && c.takeaways.length) items.push({ key: 'takeaways', title: 'Key takeaways', text: 'Key takeaways. ' + c.takeaways.map(t => sentence(text(t))).join(' ') });
    if (c.cases && c.cases.length) items.push({ key: 'cases', title: c.cases.length > 1 ? 'Case studies' : 'Case study',
      text: (c.cases.length > 1 ? 'Case studies. ' : 'Case study. ') + c.cases.map(k => sentence(text(k.title)) + ' ' + sentence(text(k.when)) + ' ' + md(k.story) + ' What to take from it: ' + md(k.lesson)).join(' ') });
    if (c.projects && c.projects.length) items.push({ key: 'projects', title: 'Sample projects',
      text: (c.projects.length > 1 ? 'Sample projects to build. ' : 'A sample project to build. ') + c.projects.map(project).join(' ') });
    const q = (c.quiz || []).length, p = (c.problems || []).length;
    items.push({ key: 'end', title: 'End of the lesson', text: 'That is the end of ' + text(c.title) + '.' + (q || p ? ' To finish it, ' + [q ? 'take the quiz of ' + q + ' questions' : '', p ? 'solve ' + (p === 1 ? 'the practice problem' : 'one of the ' + p + ' practice problems') : ''].filter(Boolean).join(' and ') + ' on the lesson page.' : '') });
    return items.filter(it => it.text && it.text.trim());
  }

  // ---------------------------------------------------------------- chunks for the speech engine
  function chunks(t, max) {
    max = max || 220;
    const out = [];
    String(t || '').split(PAUSE).forEach((part, k) => {
      if (k > 0) out.push(PAUSE);
      const sentences = tidy(part).split(/(?<=[.!?;:])\s+(?=[A-Z0-9"“(])/).filter(Boolean);
      let cur = '';
      const push = s => { if (s.trim()) out.push(s.trim()); };
      for (let s of sentences) {
        while (s.length > max) {                        // split a long sentence at a comma or a space near the limit
          let cut = s.lastIndexOf(', ', max); if (cut < max * 0.4) cut = s.lastIndexOf(' ', max); if (cut < 1) cut = max;
          push(cur); cur = '';
          push(s.slice(0, cut + 1)); s = s.slice(cut + 1).trim();
        }
        if (cur && (cur + ' ' + s).length > max) { push(cur); cur = s; } else cur = cur ? cur + ' ' + s : s;
      }
      push(cur);
    });
    return out;
  }

  MLA.narrate = { lesson, section, md, math, text, chart, question, chunks, trend, PAUSE };
})(typeof globalThis !== 'undefined' ? globalThis : window);
