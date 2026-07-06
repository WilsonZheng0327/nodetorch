import puppeteer from 'puppeteer-core';

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1000'],
  defaultViewport: { width: 1600, height: 1000 },
});
const page = await browser.newPage();
const log = (...a) => console.log(...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
await page.waitForSelector('.toolbar', { timeout: 20000 });
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
await page.waitForFunction(() => {
  const b = document.querySelector('button.toolbar-btn-test');
  return b && !b.disabled;
}, { timeout: 300000, polling: 1000 });
log('trained');
await sleep(800);

// Open backprop — defaults to the "Over training" playback page.
await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.bp-play-samples', { timeout: 30000 });
await sleep(600);
await page.screenshot({ path: '/tmp/pb-late.png' });
log('saved late-epoch');

// Jump to the first epoch (⏮ = first control button).
await page.evaluate(() => document.querySelector('.bp-play-controls button')?.click());
await sleep(600);
await page.screenshot({ path: '/tmp/pb-early.png' });
log('saved early-epoch');

// Switch to the "Per layer" tab.
await page.evaluate(() => {
  const t = [...document.querySelectorAll('.backprop-page-tab')].find((b) =>
    /per layer/i.test(b.textContent || ''),
  );
  t && t.click();
});
await page.waitForSelector('.stage-timeline', { timeout: 30000 });
await sleep(800);
await page.screenshot({ path: '/tmp/pb-perlayer.png' });
log('saved per-layer');

await browser.close();
log('done');
