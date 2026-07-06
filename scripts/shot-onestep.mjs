// End-to-end shot of Phase 3: train mlp-mnist, open Backprop, take one gradient
// step, land on a Linear layer → capture the before/after banner + weight change.
import puppeteer from 'puppeteer-core';

const OUT = '/tmp/backprop-onestep.png';
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1000'],
  defaultViewport: { width: 1600, height: 1000 },
});
const page = await browser.newPage();
page.on('console', (m) => {
  const t = m.text();
  if (/error/i.test(t)) console.log('[page]', t);
});
const log = (...a) => console.log(...a);

await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
await page.waitForSelector('.toolbar', { timeout: 20000 });
log('app loaded');

await page.click('button[title="Load a model preset"]');
await page.waitForSelector('.toolbar-presets-item', { timeout: 5000 });
await page.evaluate(() => {
  const t = [...document.querySelectorAll('.toolbar-presets-item')].find((b) =>
    /mlp/i.test(b.textContent || ''),
  );
  t && t.click();
});
await new Promise((r) => setTimeout(r, 1500));

await page.click('button.toolbar-btn-train');
log('training…');
await page.waitForFunction(
  () => {
    const b = document.querySelector('button.toolbar-btn-test');
    return b && !b.disabled;
  },
  { timeout: 300000, polling: 1000 },
);
log('trained');
await new Promise((r) => setTimeout(r, 800));

await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.bp-panel', { timeout: 30000 });
log('panel open');
await new Promise((r) => setTimeout(r, 600));

// Take one gradient step.
await page.click('.backprop-step-btn');
await page.waitForSelector('.bp-step-banner', { timeout: 30000 });
log('step taken');
await new Promise((r) => setTimeout(r, 600));

// Step left to a Linear layer so the weight before/after shows in the cell.
for (let i = 0; i < 8; i++) {
  const name = await page.$eval('.stage-detail-name', (el) => el.textContent || '');
  if (/Linear/i.test(name)) break;
  await page.keyboard.press('ArrowLeft');
  await new Promise((r) => setTimeout(r, 300));
}
await new Promise((r) => setTimeout(r, 500));

await page.screenshot({ path: OUT });
log('saved', OUT);
await browser.close();
