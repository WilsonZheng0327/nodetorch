// End-to-end shot of the backprop frame-fix: sample preview, loss-seed cell,
// 3-panel layer cell, and the After-step view. Trains mlp-mnist first.
import puppeteer from 'puppeteer-core';

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1000'],
  defaultViewport: { width: 1600, height: 1000 },
});
const page = await browser.newPage();
page.on('console', (m) => {
  if (/error/i.test(m.text())) console.log('[page]', m.text());
});
const log = (...a) => console.log(...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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
await sleep(1500);

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
await sleep(800);

await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.bp-cell, .bp-seed-cols', { timeout: 30000 });
await sleep(600);

// View 1: land on a Linear layer (3-panel cell, before-state).
async function detailName() {
  return page.$eval('.stage-detail-name', (el) => el.textContent || '').catch(() => '');
}
for (let i = 0; i < 8; i++) {
  if (/Linear/i.test(await detailName())) break;
  await page.keyboard.press('ArrowLeft');
  await sleep(250);
}
await sleep(400);
await page.screenshot({ path: '/tmp/ff-1-layer.png' });
log('shot 1: layer cell');

// View 2: the loss node (rightmost stage) → loss seed.
for (let i = 0; i < 10; i++) {
  await page.keyboard.press('ArrowRight');
  await sleep(120);
}
await sleep(400);
await page.screenshot({ path: '/tmp/ff-2-lossseed.png' });
log('shot 2: loss seed');

// View 3: take a step, then land on a Linear layer showing the After state.
await page.click('.backprop-step-btn');
await page.waitForSelector('.bp-step-banner', { timeout: 30000 });
await sleep(500);
for (let i = 0; i < 8; i++) {
  if (/Linear/i.test(await detailName())) break;
  await page.keyboard.press('ArrowLeft');
  await sleep(250);
}
await sleep(500);
await page.screenshot({ path: '/tmp/ff-3-after.png' });
log('shot 3: after step');

await browser.close();
log('done');
