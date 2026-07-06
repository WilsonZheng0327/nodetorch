// Second shot: land on a Linear layer to confirm the weight-gradient panel +
// health badge. Assumes a model is already trained in backend memory.
import puppeteer from 'puppeteer-core';

const OUT = '/tmp/backprop-linear.png';
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1000'],
  defaultViewport: { width: 1600, height: 1000 },
});
const page = await browser.newPage();
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
await new Promise((r) => setTimeout(r, 1500));

await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.bp-panel', { timeout: 30000 });
await new Promise((r) => setTimeout(r, 600));

// Step left until the detail header shows "Linear".
for (let i = 0; i < 8; i++) {
  const name = await page.$eval('.stage-detail-name', (el) => el.textContent || '');
  if (/Linear/i.test(name)) break;
  await page.keyboard.press('ArrowLeft');
  await new Promise((r) => setTimeout(r, 300));
}
await new Promise((r) => setTimeout(r, 400));
await page.screenshot({ path: OUT });
console.log('saved', OUT);
await browser.close();
