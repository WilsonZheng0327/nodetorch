// Shot the wiggle-a-weight hero: train mlp, open backprop, land on a Linear,
// click "Wiggle a weight", capture the loss-vs-weight curve.
import puppeteer from 'puppeteer-core';

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1050'],
  defaultViewport: { width: 1600, height: 1050 },
});
const page = await browser.newPage();
page.on('console', (m) => {
  if (/error/i.test(m.text())) console.log('[page]', m.text());
});
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
await page.waitForSelector('.stage-detail', { timeout: 30000 });
await sleep(600);

// land on a Linear
for (let i = 0; i < 8; i++) {
  const n = await page.$eval('.stage-detail-name', (e) => e.textContent || '').catch(() => '');
  if (/Linear/i.test(n)) break;
  await page.keyboard.press('ArrowLeft');
  await sleep(250);
}
await sleep(300);

// click "Wiggle a weight ↕" (the prompt button inside .bp-wiggle)
const clicked = await page.evaluate(() => {
  const btn = [...document.querySelectorAll('.bp-wiggle button')].find((b) =>
    /wiggle a weight/i.test(b.textContent || ''),
  );
  if (btn) {
    btn.scrollIntoView({ block: 'center' });
    btn.click();
    return true;
  }
  return false;
});
log('wiggle clicked:', clicked);
await page.waitForSelector('.bp-wiggle-svg', { timeout: 30000 });
await sleep(600);

// scroll the wiggle plot into view within the scrolling detail
await page.evaluate(() => {
  document.querySelector('.bp-wiggle-svg')?.scrollIntoView({ block: 'center' });
});
await sleep(400);
await page.screenshot({ path: '/tmp/ff-wiggle.png' });
log('saved /tmp/ff-wiggle.png');

await browser.close();
