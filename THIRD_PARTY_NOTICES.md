# Third-party software used by ML Atlas

ML Atlas includes or loads the open-source software below. Each component remains under its own license; the ML Atlas license (see [LICENSE](LICENSE)) does not change that.

## Included in this repository

### Vue.js 3.5.13 — `vendor/vue.global.prod.js`

The MIT License (MIT)

Copyright (c) 2018-present, Yuxi (Evan) You and Vue contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Downloaded by the browser at run time (not included in this repository)

When you run Python code, your browser downloads these from the jsDelivr CDN (`cdn.jsdelivr.net/pyodide/v0.27.2/full/`). They are distributed by their projects under their own licenses:

| Component | License | Project |
| --- | --- | --- |
| Pyodide 0.27.2 (Python compiled to WebAssembly) | Mozilla Public License 2.0 | https://pyodide.org |
| CPython 3.12 (inside Pyodide) | Python Software Foundation License | https://www.python.org |
| NumPy | BSD 3-Clause | https://numpy.org |
| pandas | BSD 3-Clause | https://pandas.pydata.org |
| SciPy | BSD 3-Clause | https://scipy.org |
| scikit-learn | BSD 3-Clause | https://scikit-learn.org |
| Matplotlib | Matplotlib License (PSF-based) | https://matplotlib.org |
| SQLite (Python's sqlite3 module) | Public domain | https://www.sqlite.org |

Other Pyodide packages are loaded only if a lesson or your own code imports them; each comes with its upstream license.

## Referenced, not included

Lessons link to papers, documentation and websites, and code samples marked "run locally" use libraries such as PyTorch, Hugging Face Transformers and PEFT, sentence-transformers and the Anthropic Python SDK. These are not part of ML Atlas; they are installed separately by the learner under their own licenses and terms.
