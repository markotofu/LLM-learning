/* ML Atlas – concept graph engine (framework independent, unit tested in Node).
 * Concepts form a DAG through their `prereqs` (required). `softPrereqs` (recommended: helpful, never required) are kept
 * apart in `softParents` / `softChildren`, so they never change locking, depth, learning order or tracks.
 * This module computes depth, learning order, prerequisite chains, sub-graphs (one branch and what it builds on, or
 * the map without some branches) and two layouts: a layered "prerequisite tree" and a two-sided "mind map". */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  function build(concepts, branches) {
    const nodes = {}, children = {}, parents = {}, softChildren = {}, softParents = {}, branchOrder = {};
    (branches || []).forEach((b, i) => { branchOrder[b.id] = i; });
    concepts.forEach((c, i) => { nodes[c.id] = c; children[c.id] = []; softChildren[c.id] = []; parents[c.id] = (c.prereqs || []).slice(); softParents[c.id] = (c.softPrereqs || []).slice(); if (c._order == null) c._order = i; });
    const missing = [];
    for (const c of concepts) {
      for (const p of c.prereqs || []) { if (!nodes[p]) missing.push([c.id, p]); else children[p].push(c.id); }
      for (const p of c.softPrereqs || []) { if (!nodes[p]) missing.push([c.id, p]); else softChildren[p].push(c.id); }
    }
    return { nodes, children, parents, softChildren, softParents, order: concepts.map(c => c.id), branchOrder, branches: branches || [], missing };
  }

  // The graph limited to some concepts (a Set or a list of ids), with the same shape as build(); edges to concepts
  // outside the set are dropped, so depths and layouts are computed for what is shown.
  function subgraph(g, keep) {
    const set = keep instanceof Set ? keep : new Set(keep);
    const sg = build(g.order.filter(id => set.has(id)).map(id => g.nodes[id]), g.branches);
    for (const id of sg.order) { sg.parents[id] = sg.parents[id].filter(p => sg.nodes[p]); sg.softParents[id] = sg.softParents[id].filter(p => sg.nodes[p]); }
    sg.missing = [];
    return sg;
  }

  // "Zoom to a branch": the branch's own concepts (inside) plus the concepts from other branches that they build on
  // directly, required or recommended (outside).
  function branchFocus(g, branch) {
    const inside = new Set(g.order.filter(id => g.nodes[id].branch === branch)), outside = new Set();
    for (const id of inside) for (const p of g.parents[id].concat(g.softParents[id] || [])) if (g.nodes[p] && !inside.has(p)) outside.add(p);
    return { inside, outside, ids: new Set(g.order.filter(id => inside.has(id) || outside.has(id))) };
  }

  // A cycle through the prerequisites (with { soft: true }, the recommended ones count too), or null.
  function findCycle(g, opts) {
    const soft = !!(opts && opts.soft);
    const color = {}; const stack = []; let cycle = null;
    function dfs(id) {
      color[id] = 1; stack.push(id);
      for (const p of soft ? (g.parents[id] || []).concat(g.softParents[id] || []) : (g.parents[id] || [])) {
        if (!g.nodes[p]) continue;
        if (color[p] === 1) { cycle = stack.slice(stack.indexOf(p)).concat(p); return true; }
        if (!color[p] && dfs(p)) return true;
      }
      stack.pop(); color[id] = 2; return false;
    }
    for (const id of g.order) if (!color[id] && dfs(id)) break;
    return cycle;
  }

  // Longest path from a root (a concept with no prerequisites) – "how many steps in" a concept is.
  function depths(g) {
    if (g._depths) return g._depths;
    const memo = {}, visiting = {};
    function d(id) {
      if (memo[id] != null) return memo[id];
      if (visiting[id]) return 0; // cycle guard (tests forbid cycles)
      visiting[id] = true;
      let v = 0;
      for (const p of g.parents[id]) if (g.nodes[p]) v = Math.max(v, d(p) + 1);
      visiting[id] = false; memo[id] = v; return v;
    }
    g.order.forEach(d);
    g._depths = memo;
    return memo;
  }

  // Kahn topological sort; ties broken by depth, then branch order, then authoring order.
  function topoOrder(g) {
    if (g._topo) return g._topo;
    const dep = depths(g);
    const indeg = {};
    for (const id of g.order) indeg[id] = g.parents[id].filter(p => g.nodes[p]).length;
    const ready = g.order.filter(id => indeg[id] === 0);
    const key = id => [dep[id], g.branchOrder[g.nodes[id].branch] || 0, g.nodes[id]._order];
    const cmp = (a, b) => { const x = key(a), y = key(b); return x[0] - y[0] || x[1] - y[1] || x[2] - y[2]; };
    const out = [];
    while (ready.length) {
      ready.sort(cmp);
      const id = ready.shift(); out.push(id);
      for (const c of g.children[id]) { indeg[c]--; if (indeg[c] === 0) ready.push(c); }
    }
    g._topo = out;
    return out;
  }

  function ancestors(g, id) {
    const out = new Set(); const st = [id];
    while (st.length) { const x = st.pop(); for (const p of g.parents[x] || []) if (g.nodes[p] && !out.has(p)) { out.add(p); st.push(p); } }
    return out;
  }
  function descendants(g, id) {
    const out = new Set(); const st = [id];
    while (st.length) { const x = st.pop(); for (const c of g.children[x] || []) if (!out.has(c)) { out.add(c); st.push(c); } }
    return out;
  }
  // Every concept needed to reach the targets (inclusive), in a valid learning order.
  function pathTo(g, targets) {
    const need = new Set();
    for (const t of targets || []) { if (!g.nodes[t]) continue; need.add(t); ancestors(g, t).forEach(a => need.add(a)); }
    return topoOrder(g).filter(id => need.has(id));
  }
  // Edges among a set of node ids (for highlighting a chain)
  function edgesWithin(g, set) {
    const out = [];
    for (const id of set) for (const p of g.parents[id] || []) if (set.has(p)) out.push([p, id]);
    return out;
  }

  // ---------- Layout 1: layered prerequisite tree with one horizontal lane per branch ----------
  function layoutLanes(g, opts) {
    const o = Object.assign({ colW: 236, rowH: 86, laneGap: 46, padX: 200, padY: 50 }, opts || {});
    const dep = depths(g);
    const laneIds = g.branches.map(b => b.id);
    const laneOf = id => (laneIds.includes(g.nodes[id].branch) ? g.nodes[id].branch : laneIds[laneIds.length - 1]);
    const cells = {};
    for (const id of g.order) { const l = laneOf(id), d = dep[id]; cells[l] = cells[l] || {}; (cells[l][d] = cells[l][d] || []).push(id); }
    let maxDepth = 0; for (const id of g.order) maxDepth = Math.max(maxDepth, dep[id]);
    const lanes = []; let y = o.padY;
    for (const l of laneIds) {
      const c = cells[l] || {}; const rows = Math.max(0, ...Object.values(c).map(a => a.length));
      if (!rows) continue;
      const h = Math.max(rows, 1) * o.rowH;
      lanes.push({ branch: l, y, h, rows });
      y += h + o.laneGap;
    }
    const pos = {};
    for (let d = 0; d <= maxDepth; d++) {
      for (const L of lanes) {
        const ids = ((cells[L.branch] || {})[d] || []).slice();
        if (!ids.length) continue;
        const bc = id => { const ps = g.parents[id].filter(p => pos[p]); return ps.length ? ps.reduce((s, p) => s + pos[p].y, 0) / ps.length : L.y + L.h / 2; };
        ids.sort((a, b) => bc(a) - bc(b) || g.nodes[a]._order - g.nodes[b]._order);
        const total = ids.length * o.rowH;
        const mean = ids.reduce((s, id) => s + bc(id), 0) / ids.length;
        let start = Math.max(L.y, Math.min(L.y + L.h - total, mean - total / 2));
        start = L.y + Math.round((start - L.y) / (o.rowH / 2)) * (o.rowH / 2); // snap to half rows
        start = Math.max(L.y, Math.min(L.y + L.h - total, start));
        ids.forEach((id, i) => { pos[id] = { x: o.padX + d * o.colW, y: start + i * o.rowH + o.rowH / 2, depth: d }; });
      }
    }
    return { kind: 'lanes', pos, lanes, width: o.padX + (maxDepth + 1) * o.colW, height: y - o.laneGap + o.padY, maxDepth, colW: o.colW };
  }

  // ---------- Layout 2: two-sided mind map: root in the centre, one hub per branch ----------
  // Each concept hangs under its deepest prerequisite *in the same branch* (or `parent` if given);
  // concepts with no same-branch prerequisite hang off their branch hub.
  // Options: `center`, the concept to put in the middle (default: the first one with no prerequisites, so a sub-graph
  // can keep the usual root); `focus`, a branch id for "zoom to a branch": that branch's hub sits in the middle with its
  // concepts on the right, and the concepts it builds on from other branches hang under their own hubs on the left.
  function layoutMindmap(g, opts) {
    const o = Object.assign({ colW: 222, rowH: 78, hubGap: 30 }, opts || {});
    const dep = depths(g);
    const ups = id => g.parents[id].filter(p => g.nodes[p]);
    const focus = o.focus || null;
    const center = focus ? 'hub:' + focus : (o.center && g.nodes[o.center] ? o.center : g.order.find(id => !ups(id).length));
    const prim = {};
    for (const id of g.order) {
      if (id === center) { prim[id] = null; continue; }
      const n = g.nodes[id];
      if (focus && n.branch !== focus) { prim[id] = 'hub:' + n.branch; continue; }   // context: one column under its branch hub
      const ps = ups(id);
      if (n.parent && ps.includes(n.parent) && g.nodes[n.parent].branch === n.branch) { prim[id] = n.parent; continue; }
      const same = ps.filter(p => g.nodes[p].branch === n.branch);
      if (same.length) { same.sort((a, b) => dep[b] - dep[a] || g.nodes[a]._order - g.nodes[b]._order); prim[id] = same[0]; }
      else prim[id] = focus ? center : 'hub:' + n.branch;
    }
    const kids = {};
    const hubs = g.branches.filter(b => b.id !== focus && g.order.some(id => id !== center && g.nodes[id].branch === b.id)).map(b => 'hub:' + b.id);
    hubs.forEach(h => { kids[h] = []; });
    kids[center] = [];
    g.order.forEach(id => { kids[id] = []; });
    g.order.forEach(id => { const p = prim[id]; if (p) (kids[p] = kids[p] || []).push(id); });
    Object.keys(kids).forEach(k => kids[k].sort((a, b) => dep[a] - dep[b] || g.nodes[a]._order - g.nodes[b]._order));
    const leaves = {};
    const cnt = id => { if (leaves[id] != null) return leaves[id]; const k = kids[id] || []; leaves[id] = k.length ? k.reduce((s, c) => s + cnt(c), 0) : 1; return leaves[id]; };
    hubs.forEach(cnt); kids[center].forEach(cnt);
    let right = [], left = [];
    if (focus) { right = kids[center].slice(); left = hubs.slice(); }
    else {
      // split hubs into right / left keeping branch order clockwise
      const total = hubs.reduce((s, h) => s + leaves[h], 0);
      let acc = 0;
      for (const h of hubs) { if (acc + leaves[h] / 2 <= total / 2 || !right.length) { right.push(h); acc += leaves[h]; } else left.push(h); }
      left.reverse();
    }
    const pos = {};
    pos[center] = { x: 0, y: 0, depth: 0, side: 0, hub: !!focus };
    function place(id, depth, side, yTop) {
      const k = kids[id] || [];
      let y;
      if (!k.length) y = yTop + o.rowH / 2;
      else { let cur = yTop; const ys = []; for (const c of k) { ys.push(place(c, depth + 1, side, cur)); cur += leaves[c] * o.rowH; } y = (ys[0] + ys[ys.length - 1]) / 2; }
      pos[id] = { x: side * (depth * o.colW + (depth > 0 ? 40 : 0)), y, depth, side, hub: id.startsWith('hub:') };
      return y;
    }
    function placeSide(list, side) {
      const tot = list.reduce((s, h) => s + leaves[h] * o.rowH, 0) + (list.length - 1) * o.hubGap;
      let cur = -tot / 2;
      for (const h of list) { place(h, 1, side, cur); cur += leaves[h] * o.rowH + o.hubGap; }
    }
    placeSide(right, 1); placeSide(left, -1);
    // anything not placed yet (should not happen) goes below the centre
    g.order.filter(id => !pos[id]).forEach((id, i) => { pos[id] = { x: 0, y: 140 + i * o.rowH, depth: 0, side: 0 }; });
    let minX = 0, maxX = 0, minY = 0, maxY = 0;
    for (const p of Object.values(pos)) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); }
    const treeEdges = [];
    for (const id of Object.keys(prim)) if (prim[id]) treeEdges.push([prim[id], id]);
    hubs.forEach(h => treeEdges.push([center, h]));
    return { kind: 'mindmap', pos, prim, kids, hubs, center, focus, treeEdges, bounds: { minX, maxX, minY, maxY } };
  }

  MLA.graph = { build, subgraph, branchFocus, findCycle, depths, topoOrder, ancestors, descendants, pathTo, edgesWithin, layoutLanes, layoutMindmap };
})(typeof globalThis !== 'undefined' ? globalThis : window);
