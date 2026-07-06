import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1050'],
  defaultViewport: { width: 1600, height: 1050 },
});
const page = await browser.newPage();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
await page.waitForSelector('.toolbar', { timeout: 20000 });
await page.click('button[title="Load a model preset"]');
await page.waitForSelector('.toolbar-presets-item', { timeout: 5000 });
await page.evaluate(() => {
  const t = [...document.querySelectorAll('.toolbar-presets-item')].find((b) => /mlp/i.test(b.textContent || ''));
  t && t.click();
});
await sleep(1500);
// backend already has a trained model — go straight to per-layer
await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.backprop-page-tabs', { timeout: 30000 });
await page.evaluate(() => {
  const t = [...document.querySelectorAll('.backprop-page-tab')].find((b) => /per layer/i.test(b.textContent || ''));
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
await page.waitForSelector('.bp-wiggle-svg', { timeout: 30000 });
await sleep(800);
const box = await page.$eval('.bp-lower', (el) => {
  const r = el.getBoundingClientRect();
  return { x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8), width: Math.min(1590, r.width + 16), height: Math.min(1040, r.height + 16) };
});
await page.screenshot({ path: '/tmp/lo-lower2.png', clip: box });
console.log('saved', JSON.stringify(box));
await browser.close();
