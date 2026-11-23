/* Course settings: the one file to edit when this template becomes a new course (docs/MAKING_A_COURSE.md walks through
 * the rest). It loads first, before every other script, so it must not depend on anything. Two static files repeat the
 * name and description because they are read before any script runs: index.html (title, description, loading screen)
 * and manifest.webmanifest (the installed app's name). */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.course = {
    // A short id, unique among your courses (lowercase letters, digits and dashes). It names this course's saved
    // progress, uploaded files and progress files, so several courses published on one site (you.github.io/one,
    // you.github.io/two) never mix up their data, and a progress file from one course cannot be imported into another.
    // Changing it later starts everyone's progress afresh.
    id: 'course-atlas',
    name: 'Course Atlas',                        // shown in the header, page titles, legal pages and exported files
    tagline: 'From first ideas to real projects', // shown under the name in the sidebar
    subject: 'game development',                 // completes sentences such as "an interactive map of game development"
    owner: 'Your Name',                          // copyright holder on the legal pages and in static lesson pages
    contact: 'https://github.com/your-name',     // where people can reach you about the course (legal pages)
    year: '2026',                                // copyright year
    updated: '23 November 2026',                 // "last updated" date of the terms of use and privacy policy
    siteUrl: 'https://your-name.github.io/your-course' // the published address, used by the static lesson pages and sitemap
  };
  MLA.course.storageKey = MLA.course.id + '.state.v1';        // localStorage key of the learner's progress
  MLA.course.datasetsDb = MLA.course.id + '-datasets';        // IndexedDB database of uploaded files
})(typeof globalThis !== 'undefined' ? globalThis : window);
