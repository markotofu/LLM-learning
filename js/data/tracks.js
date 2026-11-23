/* Learning tracks: a goal set of lessons; the app adds every prerequisite automatically and orders them.
 * The onboarding dialog offers every track that is not hidden. Keep 'everything' and 'custom' as they are. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};

  MLA.data.tracks = [
    { id: 'basics', icon: 'flag', title: 'Game Development Basics', short: 'The game loop, vectors and collisions', level: 'Beginner', goals: ['collisions'],
      desc: 'How a game runs frame by frame, the vector maths behind movement, and how objects detect that they touch: the groundwork for every gameplay script.' },
    { id: 'everything', icon: 'map', title: 'The Complete Course', short: 'Every lesson, in learning order', level: 'All levels', all: true, goals: [],
      desc: 'Every lesson on the map, ordered so that each one comes after everything it builds on.' },
    { id: 'custom', custom: true, hidden: true, icon: 'flag', short: 'A goal you picked on the map',
      get title() { const id = MLA.store && MLA.store.state.profile.customGoal; const c = id && MLA.content && MLA.content.byId ? MLA.content.byId[id] : null; return 'Goal: ' + (c ? c.title : 'custom'); },
      get goals() { const id = MLA.store && MLA.store.state.profile.customGoal; return id ? [id] : []; },
      desc: 'A personal goal chosen from the concept map: that concept plus everything it depends on.' }
  ];
})(typeof globalThis !== 'undefined' ? globalThis : window);
