/* Screenshot comparison of every demo and diagram in the dark and light themes.
 * Run it through tools/visual-diff.js (npm run test:visual): that renders a git ref as the baseline first,
 * then compares the working tree against it, so both sets of pictures come from the same machine. */
'use strict';
const { test, expect } = require('@playwright/test');
const { boot, go, lessons } = require('./helpers');

for (const theme of ['dark', 'light']) {
  test('demos and diagrams, ' + theme + ' theme', async ({ page }) => {
    await boot(page, '/home', { profile: { theme } });
    const list = (await lessons(page)).filter(c => c.demos.length);
    for (const c of list) {
      await go(page, '/concept/' + c.id, c.title);
      await page.waitForTimeout(250);
      const demos = page.locator('.demo');
      const n = await demos.count();
      for (let i = 0; i < n; i++) {
        await demos.nth(i).scrollIntoViewIfNeeded();
        await expect.soft(demos.nth(i)).toHaveScreenshot(theme + '-' + c.id + '-' + (c.demos[i] || 'demo' + i) + '.png');
      }
    }
  });
}
