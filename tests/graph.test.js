/* Unit tests for the concept-graph engine. Run: node tests/graph.test.js */
'use strict';
global.window = global;
require('../js/core/util.js');
require('../js/engines/graph.js');
const { suite } = require('./_load.js');
const T = suite('graph');
const G = global.MLA.graph;

const B = [{ id: 'start' }, { id: 'math' }, { id: 'ml' }];
const C = (id, branch, prereqs) => ({ id, branch, prereqs });
const concepts = [
  C('root', 'start', []), C('vec', 'math', ['root']), C('deriv', 'math', ['root']), C('gd', 'ml', ['deriv']),
  C('linreg', 'ml', ['gd', 'vec']), C('logreg', 'ml', ['linreg']), C('stats', 'math', ['root']), C('eval', 'ml', ['logreg', 'stats'])
];
const g = G.build(concepts, B);
T.ok('no missing prereqs', g.missing.length === 0);
T.ok('no cycle', G.findCycle(g) === null);
const d = G.depths(g);
T.ok('depths', d.root === 0 && d.vec === 1 && d.gd === 2 && d.linreg === 3 && d.logreg === 4 && d.eval === 5, JSON.stringify(d));
const topo = G.topoOrder(g);
const pos = Object.fromEntries(topo.map((id, i) => [id, i]));
T.ok('topo contains all', topo.length === concepts.length);
T.ok('topo respects prereqs', concepts.every(c => c.prereqs.every(p => pos[p] < pos[c.id])));
T.ok('ancestors', [...G.ancestors(g, 'logreg')].sort().join() === ['deriv', 'gd', 'linreg', 'root', 'vec'].join());
T.ok('descendants', [...G.descendants(g, 'gd')].sort().join() === ['eval', 'linreg', 'logreg'].join());
const path = G.pathTo(g, ['eval']);
T.ok('pathTo includes everything needed, in order', path.length === 8 && path[path.length - 1] === 'eval' && path.every((id, i) => g.parents[id].every(p => path.indexOf(p) < i)));
T.ok('pathTo excludes unrelated', G.pathTo(g, ['gd']).join() === 'root,deriv,gd');
T.ok('edgesWithin', G.edgesWithin(g, new Set(['gd', 'linreg', 'vec'])).length === 2);

// cycle detection
const gc = G.build([C('a', 'start', ['c']), C('b', 'math', ['a']), C('c', 'math', ['b'])], B);
T.ok('detects cycle', Array.isArray(G.findCycle(gc)));
T.ok('reports missing prereq', G.build([C('a', 'start', ['zzz'])], B).missing.length === 1);

// layouts: every node placed, no two nodes overlap
function noOverlap(posMap, w, h) {
  const ids = Object.keys(posMap);
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
    const a = posMap[ids[i]], b = posMap[ids[j]];
    if (Math.abs(a.x - b.x) < w && Math.abs(a.y - b.y) < h) return ids[i] + ' / ' + ids[j];
  }
  return null;
}
const lanes = G.layoutLanes(g);
T.ok('lanes: all placed', concepts.every(c => lanes.pos[c.id]));
T.ok('lanes: no overlap', noOverlap(lanes.pos, 188, 62) === null, noOverlap(lanes.pos, 188, 62));
T.ok('lanes: x grows with depth', concepts.every(c => c.prereqs.every(p => lanes.pos[p].x < lanes.pos[c.id].x)));
const mm = G.layoutMindmap(g);
T.ok('mindmap: centre is root', mm.center === 'root' && mm.pos.root.x === 0);
T.ok('mindmap: a hub per used branch', mm.hubs.length === 2);
T.ok('mindmap: all placed', concepts.every(c => mm.pos[c.id]));
T.ok('mindmap: no overlap', noOverlap(mm.pos, 188, 62) === null, noOverlap(mm.pos, 188, 62));
T.ok('mindmap: same-branch parent chosen', mm.prim.logreg === 'linreg' && mm.prim.gd === 'hub:ml' && mm.prim.vec === 'hub:math');
T.ok('mindmap: tree edges cover every node', mm.treeEdges.length === concepts.length - 1 + mm.hubs.length);

// recommended ("soft") prerequisites: kept apart from the required ones
const S = (id, branch, prereqs, softPrereqs) => ({ id, branch, prereqs, softPrereqs });
const soft = concepts.map(c => c.id === 'eval' ? S('eval', 'ml', ['logreg', 'stats'], ['vec']) : c.id === 'logreg' ? S('logreg', 'ml', ['linreg'], ['stats']) : c);
const gs = G.build(soft, B);
T.ok('soft: parents and children', gs.softParents.eval.join() === 'vec' && gs.softChildren.stats.join() === 'logreg' && gs.softChildren.vec.join() === 'eval');
T.ok('soft: required structure unchanged', gs.parents.logreg.join() === 'linreg' && JSON.stringify(G.depths(gs)) === JSON.stringify(d) && G.topoOrder(gs).join() === topo.join());
T.ok('soft: reports a missing recommended prerequisite', G.build([S('a', 'start', [], ['zzz'])], B).missing.length === 1);
const gsc = G.build([C('a', 'start', []), S('b', 'math', ['a'], ['c']), C('c', 'math', ['b'])], B);
T.ok('soft: a cycle through a recommended edge is found only when asked', G.findCycle(gsc) === null && Array.isArray(G.findCycle(gsc, { soft: true })));

// sub-graphs and zoom to a branch
const sub = G.subgraph(gs, new Set(['gd', 'linreg', 'logreg', 'eval']));
T.ok('subgraph: keeps the chosen nodes in order', sub.order.join() === 'gd,linreg,logreg,eval');
T.ok('subgraph: drops edges to hidden nodes', sub.parents.linreg.join() === 'gd' && sub.parents.eval.join() === 'logreg' && sub.softParents.logreg.length === 0 && sub.softParents.eval.length === 0);
T.ok('subgraph: depths are recomputed', G.depths(sub).gd === 0 && G.depths(sub).eval === 3);
const bf = G.branchFocus(gs, 'ml');
T.ok('branchFocus: the branch inside', [...bf.inside].sort().join() === 'eval,gd,linreg,logreg');
T.ok('branchFocus: direct prerequisites from other branches outside, recommended ones included', [...bf.outside].sort().join() === 'deriv,stats,vec' && bf.ids.size === 7);
const fg = G.subgraph(gs, bf.ids);
const fm = G.layoutMindmap(fg, { focus: 'ml' });
T.ok('focus mindmap: the branch hub is the centre', fm.center === 'hub:ml' && fm.pos['hub:ml'].x === 0 && fm.pos['hub:ml'].hub);
T.ok('focus mindmap: every node placed, none overlap', fg.order.every(id => fm.pos[id]) && noOverlap(fm.pos, 188, 62) === null, noOverlap(fm.pos, 188, 62));
T.ok('focus mindmap: the branch on the right, what it builds on on the left', [...bf.inside].every(id => fm.pos[id].x > 0) && [...bf.outside].every(id => fm.pos[id].x < 0));
T.ok('focus mindmap: one hub per other branch, on the left', fm.hubs.join() === 'hub:math' && fm.pos['hub:math'].x < 0);
T.ok('focus mindmap: tree edges cover every node', fm.treeEdges.length === fg.order.length + fm.hubs.length);
// a map without some branches keeps the usual root in the centre even when other nodes lose all their prerequisites
const part = G.subgraph(g, new Set(['root', 'gd', 'linreg', 'logreg', 'eval']));
const pm = G.layoutMindmap(part, { center: 'root' });
T.ok('mindmap with a centre: the centre stays, every node placed, none overlap', pm.center === 'root' && pm.pos.root.x === 0 && part.order.every(id => pm.pos[id]) && noOverlap(pm.pos, 188, 62) === null, noOverlap(pm.pos, 188, 62));
T.ok('mindmap with a centre: a node without prerequisites hangs off its hub, not the centre', pm.prim.gd === 'hub:ml' && pm.pos.gd.x !== 0);
const pl = G.layoutLanes(part);
T.ok('lanes of a sub-graph: every node placed, none overlap', part.order.every(id => pl.pos[id]) && noOverlap(pl.pos, 188, 62) === null);
T.done();
