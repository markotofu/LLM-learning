/* Tiny syntax highlighter (Python, C# and a generic fallback). Returns escaped HTML. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const words = s => new Set(s.split(/\s+/).filter(Boolean));

  const PY_KW = words('and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case');
  const PY_CONST = words('True False None');
  const PY_BUILTIN = words('print len range enumerate zip map filter sum min max abs round sorted reversed list dict set tuple int float str bool type isinstance issubclass super open iter next any all hasattr getattr setattr object repr format divmod pow input id hash vars dir callable staticmethod classmethod property Exception ValueError TypeError KeyError IndexError ZeroDivisionError RuntimeError NotImplementedError StopIteration AssertionError AttributeError ImportError ModuleNotFoundError NameError');
  const PY_RE = /(#[^\n]*)|([rRbBuUfF]{0,2}(?:"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?))|(\b(?:0[xXbBoO][\da-fA-F_]+|\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?j?)|\.\d+(?:[eE][+-]?\d+)?)|(@[A-Za-z_][\w.]*)|([A-Za-z_]\w*)|(\s+)|([\s\S])/g;

  function python(src) {
    src = String(src == null ? '' : src);
    let out = ''; let m; let prev = '';
    PY_RE.lastIndex = 0;
    while ((m = PY_RE.exec(src)) !== null) {
      const [, com, str, num, dec, id, ws, other] = m;
      if (com !== undefined) out += '<span class="tk-com">' + esc(com) + '</span>';
      else if (str !== undefined) out += '<span class="tk-str">' + esc(str) + '</span>';
      else if (num !== undefined) out += '<span class="tk-num">' + esc(num) + '</span>';
      else if (dec !== undefined) out += '<span class="tk-dec">' + esc(dec) + '</span>';
      else if (id !== undefined) {
        let cls = '';
        if (PY_KW.has(id)) cls = 'tk-kw';
        else if (PY_CONST.has(id)) cls = 'tk-const';
        else if (prev === 'def') cls = 'tk-fn';
        else if (prev === 'class') cls = 'tk-cls';
        else if (id === 'self' || id === 'cls') cls = 'tk-self';
        else if (PY_BUILTIN.has(id)) cls = 'tk-bi';
        else if (/^\s*\(/.test(src.slice(PY_RE.lastIndex, PY_RE.lastIndex + 12))) cls = 'tk-call';
        else if (/^[A-Z][A-Z0-9_]{1,}$/.test(id)) cls = 'tk-const';
        else if (/^[A-Z]/.test(id)) cls = 'tk-cls';
        out += cls ? '<span class="' + cls + '">' + id + '</span>' : id;
        prev = id;
        continue;
      } else if (ws !== undefined) { out += ws; continue; }
      else out += esc(other);
      prev = '';
    }
    return out;
  }

  // C# (Unity scripts): // and /* */ comments, strings (also $"…" interpolated and @"…" verbatim), numbers with
  // suffixes (5f, 2.5d), preprocessor lines (#if, #region), keywords, built-in types, and type names.
  const CS_KW = words('abstract as base break case catch checked class const continue default delegate do else enum event explicit extern finally fixed for foreach goto if implicit in interface internal is lock namespace new operator out override params private protected public readonly ref return sealed sizeof stackalloc static struct switch this throw try typeof unchecked unsafe using virtual void volatile while async await var get set value yield where when record init nameof');
  const CS_CONST = words('true false null');
  const CS_TYPES = words('bool byte char decimal double float int long object sbyte short string uint ulong ushort');
  const CS_RE = /(\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$))|(\$?@?"(?:""|\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)|(\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[fFdDmMuUlL]*\b|\.\d+[fFdDmM]?)|(^[ \t]*#[^\n]*)|([A-Za-z_]\w*)|(\n|[ \t\r]+)|([\s\S])/gm;

  function csharp(src) {
    src = String(src == null ? '' : src);
    let out = ''; let m; let prev = '';
    CS_RE.lastIndex = 0;
    while ((m = CS_RE.exec(src)) !== null) {
      const [, com, str, num, pre, id, ws, other] = m;
      if (com !== undefined) out += '<span class="tk-com">' + esc(com) + '</span>';
      else if (str !== undefined) out += '<span class="tk-str">' + esc(str) + '</span>';
      else if (num !== undefined) out += '<span class="tk-num">' + esc(num) + '</span>';
      else if (pre !== undefined) out += '<span class="tk-dec">' + esc(pre) + '</span>';
      else if (id !== undefined) {
        let cls = '';
        if (CS_KW.has(id)) cls = 'tk-kw';
        else if (CS_CONST.has(id)) cls = 'tk-const';
        else if (CS_TYPES.has(id)) cls = 'tk-bi';
        else if (['class', 'struct', 'interface', 'enum', 'new'].includes(prev)) cls = 'tk-cls';
        else if (/^\s*\(/.test(src.slice(CS_RE.lastIndex, CS_RE.lastIndex + 12))) cls = 'tk-call';
        else if (/^[A-Z]/.test(id)) cls = 'tk-cls';
        out += cls ? '<span class="' + cls + '">' + id + '</span>' : id;
        prev = id;
        continue;
      } else if (ws !== undefined) { out += ws; continue; }
      else out += esc(other);
      prev = '';
    }
    return out;
  }

  // Generic: comments (#), strings, numbers – good enough for shell / JSON / output blocks
  const GEN_RE = /(#[^\n]*)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?\b)|(\b(?:true|false|null|pip|python|conda|import)\b)|([\s\S])/g;
  function generic(src) {
    let out = ''; let m; GEN_RE.lastIndex = 0;
    while ((m = GEN_RE.exec(src)) !== null) {
      const [, com, str, num, kw, other] = m;
      if (com !== undefined) out += '<span class="tk-com">' + esc(com) + '</span>';
      else if (str !== undefined) out += '<span class="tk-str">' + esc(str) + '</span>';
      else if (num !== undefined) out += '<span class="tk-num">' + esc(num) + '</span>';
      else if (kw !== undefined) out += '<span class="tk-kw">' + esc(kw) + '</span>';
      else out += esc(other);
    }
    return out;
  }

  function code(src, lang) {
    const l = (lang || 'python').toLowerCase();
    if (l === 'python' || l === 'py' || l === 'pytorch') return python(src);
    if (l === 'csharp' || l === 'cs' || l === 'c#') return csharp(src);
    if (l === 'output' || l === 'text') return esc(String(src));
    return generic(String(src));
  }

  MLA.highlight = { python, csharp, generic, code };
})(typeof globalThis !== 'undefined' ? globalThis : window);
