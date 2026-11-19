/* ML Atlas – the concept map: mind map / prerequisite tree / outline, with pan, zoom, search and path highlighting */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};
  const NW = 188, NH = 62, HW = 176, HH = 46;

  function wrap(text, max) {
    const words = String(text).split(/\s+/); const lines = []; let cur = '';
    for (const w of words) { if (!cur) cur = w; else if ((cur + ' ' + w).length <= max) cur += ' ' + w; else { lines.push(cur); cur = w; } }
    if (cur) lines.push(cur);
    if (lines.length > 2) { lines.length = 2; lines[1] = lines[1].slice(0, max - 1) + '…'; }
    return lines;
  }

  MLA.components['view-map'] = {
    props: { route: Object },
    data() {
      const s = MLA.store.state;
      return { state: s, layout: s.profile.mapLayout || 'mindmap', tx: 0, ty: 0, k: 0.6, hover: null, selected: (this.route && this.route.query.focus) || null, q: '', showCross: false, pathFor: null, branchOff: {}, dragging: false, size: { w: 1000, h: 700 }, outlineFilter: 'all' };
    },
    computed: {
      P() { return MLA.progress; },
      g() { return MLA.content.graph; },
      branches() { return MLA.data.branches; },
      L() { return this.layout === 'tree' ? MLA.graph.layoutLanes(this.g) : MLA.graph.layoutMindmap(this.g); },
      goal() { return this.state.profile.goal ? this.P.track(this.state.profile.goal) : null; },
      goalSet() { const ids = this.goal ? this.P.trackConcepts(this.goal) : []; return ids.length ? new Set(ids) : null; },
      recId() { const r = this.P.recommendNext(); return r && r.concept ? r.concept.id : null; },
      nodes() {
        const out = [];
        for (const id of this.g.order) {
          const c = this.g.nodes[id]; const p = this.L.pos[id]; if (!p) continue;
          const pr = this.P.conceptProgress(id); const b = this.P.branch(c.branch) || {};
          const solved = pr.solved, np = pr.problems;
          out.push({ id, x: p.x, y: p.y, title: c.title, lines: wrap(c.short || c.title, 23), color: b.color || '#888', branch: c.branch, status: this.P.status(id), pct: pr.pct, meta: (np ? solved + '/' + np + ' problems · ' : '') + c.minutes + ' min' });
        }
        return out;
      },
      hubs() {
        if (this.layout !== 'mindmap') return [];
        return this.L.hubs.map(h => { const b = this.P.branch(h.slice(4)); const p = this.L.pos[h]; const st = this.P.branchStats().find(x => x.branch.id === b.id); return { id: h, x: p.x, y: p.y, label: b.short || b.name, color: b.color, pct: st ? st.pct : 0, branch: b.id }; });
      },
      posOf() { const m = {}; for (const n of this.nodes) m[n.id] = n; for (const h of this.hubs) m[h.id] = h; return m; },
      focusId() { return this.hover || this.selected; },
      anc() { const id = this.pathFor || this.focusId; return id && this.g.nodes[id] ? MLA.graph.ancestors(this.g, id) : new Set(); },
      desc() { const id = this.focusId; return id && this.g.nodes[id] && !this.pathFor ? MLA.graph.descendants(this.g, id) : new Set(); },
      edges() {
        const out = []; const P = this.posOf;
        const focus = this.pathFor || this.focusId;
        const chain = new Set(this.anc); if (focus) chain.add(focus);
        const add = (a, b, kind) => { const A = P[a], B = P[b]; if (!A || !B) return; const isHubA = a.startsWith('hub:') || a === this.L.center; const color = B.color; let cls = kind; if (focus) { if (chain.has(a) && chain.has(b)) cls += ' hl-anc'; else if ((b !== focus && this.desc.has(b) && (a === focus || this.desc.has(a)))) cls += ' hl-desc'; else cls += ' faded'; } out.push({ key: a + '>' + b + kind, d: this.path(A, B, isHubA, a, b), color, cls }); };
        if (this.layout === 'mindmap') {
          for (const [a, b] of this.L.treeEdges) add(a, b, 'tree');
          const treeSet = new Set(this.L.treeEdges.map(e => e[0] + '>' + e[1]));
          for (const id of this.g.order) for (const p of this.g.parents[id]) {
            if (treeSet.has(p + '>' + id)) continue;
            const inChain = focus && chain.has(id) && chain.has(p);
            const toDesc = focus && !this.pathFor && (p === focus || this.desc.has(p)) && this.desc.has(id);
            if (this.showCross || inChain || toDesc) add(p, id, 'cross');
          }
        } else {
          for (const id of this.g.order) for (const p of this.g.parents[id]) add(p, id, 'tree');
        }
        return out;
      },
      lanes() { return this.layout === 'tree' ? this.L.lanes.map(l => Object.assign({}, l, { b: this.P.branch(l.branch) })) : []; },
      bounds() {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const n of this.nodes.concat(this.hubs)) { minX = Math.min(minX, n.x - NW / 2); maxX = Math.max(maxX, n.x + NW / 2); minY = Math.min(minY, n.y - NH / 2); maxY = Math.max(maxY, n.y + NH / 2); }
        if (this.layout === 'tree') minX = Math.min(minX, 0);
        return { minX, maxX, minY, maxY };
      },
      transform() { return 'translate(' + this.tx.toFixed(1) + ',' + this.ty.toFixed(1) + ') scale(' + this.k.toFixed(4) + ')'; },
      sel() { return this.selected && this.g.nodes[this.selected] ? this.g.nodes[this.selected] : null; },
      selHub() { return this.selected && this.selected.startsWith('hub:') ? this.P.branch(this.selected.slice(4)) : null; },
      selInfo() {
        const c = this.sel; if (!c) return null;
        const pr = this.P.conceptProgress(c.id);
        return { c, pr, status: this.P.status(c.id), prereqs: c.prereqs.map(this.P.concept), next: this.g.children[c.id].map(this.P.concept), path: MLA.graph.pathTo(this.g, [c.id]).filter(id => id !== c.id).map(this.P.concept), branch: this.P.branch(c.branch) };
      },
      hubInfo() { const b = this.selHub; if (!b) return null; const list = MLA.graph.topoOrder(this.g).filter(id => this.g.nodes[id].branch === b.id).map(this.P.concept); return { b, list, stats: this.P.branchStats().find(x => x.branch.id === b.id) }; },
      matches() { const q = this.q.trim().toLowerCase(); if (!q) return []; return MLA.data.concepts.filter(c => c.title.toLowerCase().includes(q) || (c.summary || '').toLowerCase().includes(q) || (c.tags || []).some(t => t.includes(q))).slice(0, 8); },
      outline() {
        const order = MLA.graph.topoOrder(this.g); const dep = MLA.graph.depths(this.g);
        const q = this.q.trim().toLowerCase();
        return this.branches.map(b => {
          let list = order.filter(id => this.g.nodes[id].branch === b.id).map(id => this.g.nodes[id]);
          if (q) list = list.filter(c => c.title.toLowerCase().includes(q) || (c.summary || '').toLowerCase().includes(q));
          if (this.outlineFilter !== 'all') list = list.filter(c => { const s = this.P.status(c.id); return this.outlineFilter === 'todo' ? !['completed', 'mastered'].includes(s) : ['completed', 'mastered'].includes(s); });
          const min = Math.min(...list.map(c => dep[c.id]));
          return { b, items: list.map(c => ({ c, indent: Math.min(4, dep[c.id] - min), status: this.P.status(c.id), pr: this.P.conceptProgress(c.id) })), stats: this.P.branchStats().find(x => x.branch.id === b.id) };
        }).filter(x => x.items.length);
      }
    },
    watch: {
      layout(v) { this.state.profile.mapLayout = v; this.$nextTick(() => this.fit(true)); },
      'route.query.focus'(v) { if (v) this.focusOn(v); }
    },
    mounted() {
      this.measure();
      this.$nextTick(() => { if (this.selected) this.focusOn(this.selected, true); else this.fit(true); });
      this._onResize = () => this.measure(); global.addEventListener('resize', this._onResize);
      this._onKey = e => this.onKey(e); document.addEventListener('keydown', this._onKey);
    },
    unmounted() { global.removeEventListener('resize', this._onResize); document.removeEventListener('keydown', this._onKey); },
    methods: {
      measure() { const el = this.$refs.svg; if (!el) return; const r = el.getBoundingClientRect(); this.size = { w: r.width || 1000, h: r.height || 700 }; },
      path(A, B, fromHub, a, b) {
        if (this.layout === 'tree') {
          const x1 = A.x + NW / 2, y1 = A.y, x2 = B.x - NW / 2, y2 = B.y; const dx = Math.max(36, (x2 - x1) * 0.5);
          return 'M' + x1 + ' ' + y1 + ' C' + (x1 + dx) + ' ' + y1 + ',' + (x2 - dx) + ' ' + y2 + ',' + x2 + ' ' + y2;
        }
        const pa = this.L.pos[a] || { side: 0 }, pb = this.L.pos[b] || { side: 0 };
        const side = pb.side || pa.side || 1;
        const wa = a.startsWith('hub:') ? HW : NW, wb = b.startsWith('hub:') ? HW : NW;
        if (pa.side === pb.side || pa.side === 0) {
          const x1 = A.x + side * (pa.side === 0 ? 70 : wa / 2), x2 = B.x - side * wb / 2; const dx = (x2 - x1) * 0.5;
          return 'M' + x1 + ' ' + A.y + ' C' + (x1 + dx) + ' ' + A.y + ',' + (x2 - dx) + ' ' + B.y + ',' + x2 + ' ' + B.y;
        }
        const x1 = A.x, x2 = B.x, my = Math.min(A.y, B.y) - 120;
        return 'M' + x1 + ' ' + (A.y - NH / 2) + ' C' + x1 + ' ' + my + ',' + x2 + ' ' + my + ',' + x2 + ' ' + (B.y - NH / 2);
      },
      fit(initial) {
        this.measure(); const b = this.bounds; if (!isFinite(b.minX)) return;
        const pad = 40; const w = b.maxX - b.minX + pad * 2, h = b.maxY - b.minY + pad * 2;
        let k = Math.min(this.size.w / w, this.size.h / h);
        if (initial) k = Math.max(k, this.size.w < 700 ? 0.42 : 0.5);
        k = Math.max(0.15, Math.min(1.1, k));
        this.k = k;
        if (initial && this.layout === 'tree') { this.tx = pad * k - b.minX * k; this.ty = this.size.h / 2 - ((b.minY + b.maxY) / 2) * k; if ((b.maxY - b.minY) * k > this.size.h) this.ty = pad - b.minY * k; return; }
        this.tx = this.size.w / 2 - ((b.minX + b.maxX) / 2) * k; this.ty = this.size.h / 2 - ((b.minY + b.maxY) / 2) * k;
      },
      fitAll() { this.fit(false); },
      center(id, k) { const n = this.posOf[id]; if (!n) return; this.measure(); if (k) this.k = k; const drawerOffset = (global.innerWidth || 1200) > 900 ? 170 : 0; this.tx = this.size.w / 2 - drawerOffset - n.x * this.k; this.ty = this.size.h / 2 - n.y * this.k; },
      focusOn(id, instant) { if (!this.posOf[id]) return; this.selected = id; this.center(id, Math.max(this.k, 0.8)); },
      zoomAt(px, py, nk) { nk = Math.max(0.15, Math.min(2.5, nk)); const wx = (px - this.tx) / this.k, wy = (py - this.ty) / this.k; this.k = nk; this.tx = px - wx * nk; this.ty = py - wy * nk; },
      zoomBy(f) { this.measure(); this.zoomAt(this.size.w / 2, this.size.h / 2, this.k * f); },
      onWheel(e) { e.preventDefault(); const r = this.$refs.svg.getBoundingClientRect(); const f = Math.exp(-(e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY) * 0.0015); this.zoomAt(e.clientX - r.left, e.clientY - r.top, this.k * f); },
      onDown(e) {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        this._pts = this._pts || new Map(); this._pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
        this._moved = false;
        if (this._pts.size === 1) this._start = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty, id: e.pointerId };
        if (this._pts.size === 2) { const [a, b] = Array.from(this._pts.values()); this._pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), k: this.k }; }
      },
      onMove(e) {
        if (!this._pts || !this._pts.has(e.pointerId)) return;
        this._pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (this._pts.size === 1 && this._start) {
          const dx = e.clientX - this._start.x, dy = e.clientY - this._start.y;
          if (!this._moved && Math.hypot(dx, dy) > 4) { this._moved = true; this.dragging = true; try { this.$refs.svg.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } }
          if (this._moved) { this.tx = this._start.tx + dx; this.ty = this._start.ty + dy; }
        } else if (this._pts.size === 2 && this._pinch) {
          const [a, b] = Array.from(this._pts.values()); const r = this.$refs.svg.getBoundingClientRect();
          this._moved = true; this.zoomAt((a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top, this._pinch.k * Math.hypot(a.x - b.x, a.y - b.y) / this._pinch.d);
        }
      },
      onUp(e) { if (this._pts) this._pts.delete(e.pointerId); if (!this._pts || this._pts.size === 0) { this.dragging = false; this._pinch = null; setTimeout(() => { this._moved = false; }, 0); } else if (this._pts.size === 1) { const [p] = Array.from(this._pts.values()); this._start = { x: p.x, y: p.y, tx: this.tx, ty: this.ty }; this._pinch = null; } },
      onBgClick() { if (this._moved) return; this.selected = null; this.pathFor = null; },
      select(id) { if (this._moved) return; this.selected = this.selected === id ? null : id; this.pathFor = null; if (this.selected) this.$nextTick(() => this.ensureVisible(id)); },
      // keep the selected node out from under the detail drawer (right side on desktop, bottom sheet on mobile)
      ensureVisible(id) {
        const n = this.posOf[id]; if (!n) return; this.measure();
        const sx = n.x * this.k + this.tx, sy = n.y * this.k + this.ty, half = 100 * this.k;
        if ((global.innerWidth || 1200) > 900) { const limit = this.size.w - 370; if (sx + half > limit) this.tx -= (sx + half) - limit + 20; if (sx - half < 10) this.tx += 10 - (sx - half); }
        else { const limit = this.size.h * 0.42; if (sy > limit) this.ty -= sy - limit; }
      },
      showPath(id) { this.pathFor = this.pathFor === id ? null : id; },
      pick(c) { this.q = ''; if (this.layout === 'outline') { this.go('/concept/' + c.id); return; } this.focusOn(c.id); },
      searchEnter() { if (this.matches[0]) this.pick(this.matches[0]); },
      go(p) { MLA.router.navigate(p); },
      setGoalFor(id) {
        const t = (MLA.data.tracks || []).find(x => !x.all && !x.custom && x.goals.length === 1 && x.goals[0] === id);
        if (t) this.state.profile.goal = t.id;
        else { this.state.profile.customGoal = id; this.state.profile.goal = 'custom'; }
        MLA.ui.toast('Goal set: ' + this.P.concept(id).title + '. Recommendations now follow this path.', 'success');
      },
      clearGoal() { this.state.profile.goal = null; },
      nodeClass(n) {
        const f = this.pathFor || this.focusId; const cls = [n.status];
        if (this.selected === n.id) cls.push('sel');
        if (f) { if (n.id === f) cls.push('focus'); else if (this.anc.has(n.id)) cls.push('anc'); else if (this.desc.has(n.id)) cls.push('desc'); else cls.push('faded'); }
        else if (this.goalSet && !this.goalSet.has(n.id)) cls.push('off-goal');
        if (this.branchOff[n.branch]) cls.push('faded');
        if (n.id === this.recId) cls.push('rec');
        if (this.layout === 'mindmap' && n.id === this.L.center) cls.push('center');
        return cls;
      },
      toggleBranch(id) { this.branchOff[id] = !this.branchOff[id]; },
      onKey(e) {
        if (/INPUT|TEXTAREA|SELECT/.test((e.target && e.target.tagName) || '')) return;
        if (e.key === '+' || e.key === '=') this.zoomBy(1.2); else if (e.key === '-') this.zoomBy(1 / 1.2); else if (e.key === '0' || e.key === 'f') this.fitAll(); else if (e.key === 'Escape') { this.selected = null; this.pathFor = null; }
      },
      statusLabel(s) { return { mastered: 'Mastered', completed: 'Completed', started: 'In progress', ready: 'Ready — prerequisites done', locked: 'Prerequisites not finished yet' }[s]; },
      cstatus(id) { return this.P.status(id); }
    },
    template: `
    <div class="map-view">
      <div class="map-toolbar">
        <div class="pills">
          <button :class="{active:layout==='mindmap'}" @click="layout='mindmap'"><mla-icon name="radial" :size="14"/> Mind map</button>
          <button :class="{active:layout==='tree'}" @click="layout='tree'"><mla-icon name="tree" :size="14"/> Prerequisite tree</button>
          <button :class="{active:layout==='outline'}" @click="layout='outline'"><mla-icon name="list" :size="14"/> Outline</button>
        </div>
        <div class="map-search">
          <mla-icon name="search" :size="15"/>
          <input type="search" v-model="q" placeholder="Find a concept… (e.g. attention)" @keydown.enter="searchEnter" aria-label="Search concepts" />
          <div v-if="matches.length && layout!=='outline'" class="search-pop">
            <a v-for="c in matches" :key="c.id" @click="pick(c)"><status-dot :status="cstatus(c.id)" :size="16"/> <span>{{ c.title }}</span> <branch-chip :id="c.branch" small /></a>
          </div>
        </div>
        <div v-if="layout!=='outline'" class="row map-zoom">
          <label v-if="layout==='mindmap'" class="checkbox small" title="Show every cross-branch prerequisite link"><input type="checkbox" v-model="showCross"> all links</label>
          <button class="icon-btn" title="Zoom out (-)" @click="zoomBy(1/1.25)"><mla-icon name="minus"/></button>
          <span class="tiny dim mono" style="width:38px;text-align:center">{{ Math.round(k*100) }}%</span>
          <button class="icon-btn" title="Zoom in (+)" @click="zoomBy(1.25)"><mla-icon name="plus"/></button>
          <button class="icon-btn" title="Fit everything (0)" @click="fitAll"><mla-icon name="fit"/></button>
        </div>
      </div>

      <div v-if="goal && layout!=='outline'" class="goal-banner"><mla-icon name="flag" :size="14"/> Goal: <b>{{ goal.title }}</b> — {{ P.trackProgress(goal).done }}/{{ P.trackProgress(goal).total }} concepts done. Concepts outside this path are dimmed. <button class="btn xs ghost" @click="clearGoal">Clear goal</button></div>

      <div v-if="layout==='outline'" class="outline-wrap">
        <div class="pills mb"><button :class="{active:outlineFilter==='all'}" @click="outlineFilter='all'">All</button><button :class="{active:outlineFilter==='todo'}" @click="outlineFilter='todo'">Still to learn</button><button :class="{active:outlineFilter==='done'}" @click="outlineFilter='done'">Completed</button></div>
        <div v-for="grp in outline" :key="grp.b.id" class="card outline-branch" :style="{'--bc': grp.b.color}">
          <div class="card-title"><h3><span class="branch-dot"></span>{{ grp.b.name }}</h3><span class="small muted">{{ grp.stats.done }}/{{ grp.stats.concepts }} concepts · {{ grp.stats.solved }}/{{ grp.stats.problems }} problems</span></div>
          <p class="small muted">{{ grp.b.desc }}</p>
          <div class="outline-list">
            <a v-for="it in grp.items" :key="it.c.id" class="outline-item" :style="{paddingLeft: (10 + it.indent*18)+'px'}" @click="go('/concept/'+it.c.id)">
              <status-dot :status="it.status" /><span class="oi-title">{{ it.c.title }}</span>
              <span class="oi-meta tiny dim">{{ it.c.minutes }} min<span v-if="it.c.problems.length"> · {{ it.pr.solved }}/{{ it.c.problems.length }} problems</span></span>
              <mla-progress v-if="it.pr.started && !it.pr.completed" :value="it.pr.pct" thin style="width:60px" />
            </a>
          </div>
        </div>
      </div>

      <div v-else class="map-stage">
        <svg ref="svg" class="map-svg" :class="{dragging, zoomedout: k < 0.5}" @wheel="onWheel" @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp" @pointercancel="onUp" @click="onBgClick">
          <g :transform="transform">
            <g v-for="l in lanes" :key="l.branch" class="lane">
              <rect :x="-20" :y="l.y - 18" :width="L.width + 40" :height="l.h + 36" rx="16" :fill="l.b.color" class="lane-bg"/>
              <text :x="0" :y="l.y - 2" class="lane-label" :fill="l.b.color">{{ l.b.name }}</text>
            </g>
            <path v-for="e in edges" :key="e.key" :d="e.d" class="edge" :class="e.cls" :stroke="e.color" />
            <g v-if="layout==='mindmap'">
              <g v-for="h in hubs" :key="h.id" class="hub" :class="{sel: selected===h.id, faded: branchOff[h.branch]}" :transform="'translate('+h.x+','+h.y+')'" @click.stop="select(h.id)">
                <rect :x="-88" :y="-23" width="176" height="46" rx="23" :fill="h.color" class="hub-bg"/>
                <text x="0" y="-2" text-anchor="middle" class="hub-label">{{ h.label }}</text>
                <text x="0" y="14" text-anchor="middle" class="hub-meta">{{ h.pct }}% complete</text>
              </g>
            </g>
            <g v-for="n in nodes" :key="n.id" class="mnode" :class="nodeClass(n)" :transform="'translate('+n.x+','+n.y+')'" @click.stop="select(n.id)" @dblclick.stop="go('/concept/'+n.id)" @pointerenter="hover=n.id" @pointerleave="hover=null" tabindex="0" role="button" :aria-label="n.title" @keydown.enter="select(n.id)">
              <rect class="mnode-bg" x="-94" y="-31" width="188" height="62" rx="12" :style="{stroke: n.color}"/>
              <rect class="mnode-strip" x="-94" y="-31" width="6" height="62" rx="3" :fill="n.color"/>
              <text class="mnode-title" x="-80" :y="n.lines.length>1 ? -11 : -4"><tspan v-for="(ln,i) in n.lines" :key="i" x="-80" :dy="i ? 15 : 0">{{ ln }}</tspan></text>
              <text class="mnode-meta" x="-80" :y="n.lines.length>1 ? 21 : 15">{{ n.meta }}</text>
              <rect class="mnode-track" x="-80" y="25" width="160" height="2.5" rx="1.2"/>
              <rect class="mnode-bar" x="-80" y="25" :width="1.6*n.pct" height="2.5" rx="1.2" :fill="n.color"/>
              <g class="mnode-status" transform="translate(78,-17)">
                <circle r="9" class="ms-circle"/>
                <path v-if="n.status==='completed'" d="M-4 0.5 l2.6 2.6 l5-5.4" class="ms-glyph"/>
                <path v-else-if="n.status==='mastered'" d="M0 -5.5 L1.6 -1.7 L5.6 -1.4 L2.5 1.2 L3.4 5.2 L0 3.1 L-3.4 5.2 L-2.5 1.2 L-5.6 -1.4 L-1.6 -1.7 Z" class="ms-star"/>
                <path v-else-if="n.status==='locked'" d="M-3.5 -0.5 h7 v5 h-7 z M-2 -0.5 v-1.8 a2 2 0 0 1 4 0 v1.8" class="ms-glyph"/>
                <circle v-else-if="n.status==='started'" r="3.5" class="ms-dot"/>
                <path v-else d="M-3 0 h6 M0 -3 v6" class="ms-glyph"/>
              </g>
            </g>
          </g>
        </svg>

        <div class="map-legend">
          <div class="legend-row"><span class="lg completed"></span>completed <span class="lg mastered"></span>mastered <span class="lg started"></span>in progress <span class="lg ready"></span>ready <span class="lg locked"></span>needs prerequisites</div>
          <div class="legend-row"><button v-for="b in branches.slice(1)" :key="b.id" class="lg-branch" :class="{off: branchOff[b.id]}" :style="{'--bc': b.color}" @click="toggleBranch(b.id)" :title="'Dim / undim ' + b.name"><span class="branch-dot"></span>{{ b.short }}</button></div>
          <div class="tiny dim">Drag to pan · scroll or pinch to zoom · click a concept for details · double-click to open it</div>
        </div>

        <aside v-if="selInfo" class="map-drawer">
          <button class="icon-btn drawer-close" aria-label="Close" @click="selected=null; pathFor=null"><mla-icon name="x"/></button>
          <div class="row" style="gap:6px"><branch-chip :id="selInfo.c.branch" small /><level-badge :level="selInfo.c.level"/><span class="chip tiny"><mla-icon name="clock" :size="12"/> {{ selInfo.c.minutes }} min</span></div>
          <h3 class="mt" style="margin-bottom:4px">{{ selInfo.c.title }}</h3>
          <p class="small muted">{{ selInfo.c.summary }}</p>
          <div class="row small"><status-dot :status="selInfo.status"/> {{ statusLabel(selInfo.status) }}</div>
          <mla-progress :value="selInfo.pr.pct" thin :color="selInfo.branch.color" style="margin:8px 0 4px" />
          <div class="tiny dim">{{ selInfo.pr.sectionsRead }}/{{ selInfo.pr.sections }} sections · quiz {{ selInfo.pr.quizPassed ? 'passed ✓' : (selInfo.pr.quizAttempts ? 'best ' + selInfo.pr.quizBest + '%' : 'not taken') }} · {{ selInfo.pr.solved }}/{{ selInfo.pr.problems }} problems</div>
          <div class="btn-row mt">
            <button class="btn primary sm" @click="go('/concept/'+selInfo.c.id)"><mla-icon name="book" :size="14"/> {{ selInfo.pr.started ? 'Continue' : 'Start learning' }}</button>
            <button v-if="selInfo.c.problems.length" class="btn sm" @click="go('/concept/'+selInfo.c.id+'?tab=practice')"><mla-icon name="code" :size="14"/> Practice</button>
          </div>
          <div class="drawer-sec" v-if="selInfo.prereqs.length">
            <div class="tiny dim upper">Prerequisites</div>
            <div class="chips"><a v-for="p in selInfo.prereqs" :key="p.id" class="chip clickable" @click="focusOn(p.id)"><status-dot :status="cstatus(p.id)" :size="15"/> {{ p.title }}</a></div>
          </div>
          <div class="drawer-sec" v-if="selInfo.next.length">
            <div class="tiny dim upper">Unlocks</div>
            <div class="chips"><a v-for="p in selInfo.next" :key="p.id" class="chip clickable" @click="focusOn(p.id)"><status-dot :status="cstatus(p.id)" :size="15"/> {{ p.title }}</a></div>
          </div>
          <div class="drawer-sec">
            <div class="row between"><div class="tiny dim upper">Full learning path ({{ selInfo.path.length }} concepts before this)</div><button class="btn xs" :class="{primary: pathFor===selInfo.c.id}" @click="showPath(selInfo.c.id)">{{ pathFor===selInfo.c.id ? 'Hide' : 'Highlight' }}</button></div>
            <ol class="path-list small" v-if="pathFor===selInfo.c.id">
              <li v-for="p in selInfo.path" :key="p.id" :class="cstatus(p.id)"><a @click="go('/concept/'+p.id)">{{ p.title }}</a> <span class="tiny dim">{{ ['completed','mastered'].includes(cstatus(p.id)) ? '✓' : '' }}</span></li>
            </ol>
            <button class="btn xs ghost mt" @click="setGoalFor(selInfo.c.id)"><mla-icon name="flag" :size="12"/> Make this my goal</button>
          </div>
        </aside>
        <aside v-else-if="hubInfo" class="map-drawer">
          <button class="icon-btn drawer-close" aria-label="Close" @click="selected=null"><mla-icon name="x"/></button>
          <h3 :style="{color: hubInfo.b.color}">{{ hubInfo.b.name }}</h3>
          <p class="small muted">{{ hubInfo.b.desc }}</p>
          <mla-progress :value="hubInfo.stats.pct" thin :color="hubInfo.b.color" :label="hubInfo.stats.done + '/' + hubInfo.stats.concepts + ' concepts · ' + hubInfo.stats.solved + '/' + hubInfo.stats.problems + ' problems'" />
          <ol class="path-list small mt"><li v-for="c in hubInfo.list" :key="c.id" :class="cstatus(c.id)"><a @click="focusOn(c.id)">{{ c.title }}</a></li></ol>
        </aside>
      </div>
    </div>`
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
