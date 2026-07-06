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
await sleep(1000);

// Fix 1: dashboard — should have NO "Samples" tab (classification).
const dashOpen = await page.$('.dashboard-tabs');
if (!dashOpen) await page.keyboard.press('2');
await page.waitForSelector('.dashboard-tabs', { timeout: 5000 });
await sleep(500);
const tabs = await page.$$eval('.dashboard-tabs .dashboard-tab', (bs) =>
  bs.map((b) => b.textContent?.trim()),
);
log('dashboard tabs:', JSON.stringify(tabs));
await page.screenshot({ path: '/tmp/fx-dashboard.png' });
await page.keyboard.press('Escape');
await sleep(400);

// Fix 4 + 2: backprop "over training" — 4-per-row bigger cards.
await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.bp-play-samples', { timeout: 30000 });
await sleep(700);
await page.screenshot({ path: '/tmp/fx-playback.png' });
log('saved playback');

// Fix 3: per-layer timeline horizontal scroll — narrow the window so it overflows.
await page.evaluate(() => {
  const t = [...document.querySelectorAll('.backprop-page-tab')].find((b) =>
    /per layer/i.test(b.textContent || ''),
  );
  t && t.click();
});
await page.waitForSelector('.stage-timeline', { timeout: 30000 });
await sleep(800);
await page.setViewport({ width: 760, height: 900 });
await sleep(500);
// scroll the timeline to the left to prove the input end is reachable
const scrollInfo = await page.$eval('.stage-timeline', (el) => {
  const before = { scrollLeft: el.scrollLeft, scrollWidth: el.scrollWidth, clientWidth: el.clientWidth };
  el.scrollLeft = 0;
  return { ...before, overflows: el.scrollWidth > el.clientWidth + 2 };
});
log('timeline overflow:', JSON.stringify(scrollInfo));
await sleep(400);
await page.screenshot({ path: '/tmp/fx-timeline-scroll.png' });
log('saved timeline-scroll');

await browser.close();
log('done');
