// Browser tests for ML Atlas (Playwright + Chromium).
//   npm run test:e2e      functional and layout tests (also run in CI before every deploy)
//   npm run test:visual   screenshot comparison of every demo and diagram against a git ref (tools/visual-diff.js)
// One-time setup: npm install && npx playwright install chromium
const { defineConfig, devices } = require('@playwright/test');

const external = process.env.MLA_URL; // set by tools/visual-diff.js, which starts its own servers

module.exports = defineConfig({
  testDir: 'tests/e2e',
  timeout: 180 * 1000,
  expect: { timeout: 20 * 1000, toHaveScreenshot: { maxDiffPixelRatio: 0.002, animations: 'disabled', caret: 'hide' } },
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list'], ['html', { open: 'never' }]],
  snapshotPathTemplate: '{testDir}/.baseline/{arg}{ext}',
  updateSnapshots: process.env.MLA_UPDATE_SNAPSHOTS || 'none',
  use: {
    baseURL: external || 'http://localhost:4173',
    serviceWorkers: 'block',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: external ? undefined : { command: 'node serve.js 4173', url: 'http://localhost:4173', reuseExistingServer: !process.env.CI, timeout: 30 * 1000 },
  projects: [
    { name: 'e2e', testMatch: /app\.spec\.js/, use: { ...devices['Desktop Chrome'] } },
    { name: 'visual', testMatch: /visual\.spec\.js/, use: { ...devices['Desktop Chrome'], viewport: { width: 1200, height: 900 } } },
  ],
});
