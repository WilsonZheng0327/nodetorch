import puppeteer from 'puppeteer-core';

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1050'],
  defaultViewport: { width: 1600, height: 1050 },
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

await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.backprop-page-tabs', { timeout: 30000 });
await sleep(500);
// centered toggle — capture the "over training" header
await page.screenshot({ path: '/tmp/lo-header.png', clip: { x: 24, y: 24, width: 1552, height: 90 } });

// per layer → land on a Linear (has mechanism + wiggle side by side)
await page.evaluate(() => {
  const t = [...document.querySelectorAll('.backprop-page-tab')].find((b) =>
    /per layer/i.test(b.textContent || ''),
  );
  t && t.click();
});
await page.waitForSelector('.stage-timeline', { timeout: 30000 });
await sleep(600);
for (let i = 0; i < 8; i++) {
  const n = await page.$eval('.stage-detail-name', (e) => e.textContent || '').catch(() => '');
  if (/Linear/i.test(n)) break;
  await page.keyboard.press('ArrowLeft');
  await sleep(250);
}
// wait for the auto-expanded wiggle chart
await page.waitForSelector('.bp-wiggle-svg', { timeout: 30000 });
await sleep(700);
await page.evaluate(() => document.querySelector('.bp-lower')?.scrollIntoView({ block: 'center' }));
await sleep(400);
await page.screenshot({ path: '/tmp/lo-lower.png' });
log('saved');
await browser.close();
log('done');
