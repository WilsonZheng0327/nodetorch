// Screenshot the Backprop panel end-to-end: load the mlp-mnist preset, train it,
// then open the backward step-through and capture a per-layer backward cell.
import puppeteer from 'puppeteer-core';

const OUT = '/tmp/backprop-panel.png';

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1000'],
  defaultViewport: { width: 1600, height: 1000 },
});

const page = await browser.newPage();
page.on('console', (m) => {
  const t = m.text();
  if (t.includes('Backprop') || t.toLowerCase().includes('error')) console.log('[page]', t);
});

const log = (...a) => console.log(...a);

await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
await page.waitForSelector('.toolbar', { timeout: 20000 });
log('app loaded');

// --- Load the mlp-mnist preset ---
await page.click('button[title="Load a model preset"]');
await page.waitForSelector('.toolbar-presets-item', { timeout: 5000 });
const clicked = await page.evaluate(() => {
  const items = [...document.querySelectorAll('.toolbar-presets-item')];
  const t = items.find((b) => /mlp/i.test(b.textContent || ''));
  if (t) {
    t.click();
    return t.textContent;
  }
  return null;
});
log('preset clicked:', clicked);
await new Promise((r) => setTimeout(r, 1500));

// --- Train ---
await page.click('button.toolbar-btn-train');
log('training started, waiting for completion…');

// Done when the Test button (disabled until modelTrained) becomes enabled.
await page.waitForFunction(
  () => {
    const b = document.querySelector('button.toolbar-btn-test');
    return b && !b.disabled;
  },
  { timeout: 300000, polling: 1000 },
);
log('training complete');
await new Promise((r) => setTimeout(r, 800));

// --- Open Backprop panel ---
await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.bp-panel', { timeout: 30000 });
log('backprop panel open');

// Step a couple layers inward so we land on a weighted layer (Linear), not the loss.
await page.keyboard.press('ArrowLeft');
await new Promise((r) => setTimeout(r, 400));
await page.keyboard.press('ArrowLeft');
await new Promise((r) => setTimeout(r, 600));

await page.screenshot({ path: OUT });
log('screenshot saved:', OUT);

await browser.close();
