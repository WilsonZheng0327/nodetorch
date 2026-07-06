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

// Backprop — clean (slider above timeline, icon on data card)
await page.click('button[title="Step through backprop"]');
await page.waitForSelector('.stage-timeline', { timeout: 30000 });
await sleep(700);
await page.screenshot({ path: '/tmp/icon-backprop.png' });
log('saved backprop clean');
// open the menu
await page.evaluate(() => document.querySelector('.sample-menu-btn')?.click());
await page.waitForSelector('.sample-menu-pop', { timeout: 5000 });
await sleep(400);
await page.screenshot({ path: '/tmp/icon-backprop-menu.png' });
log('saved backprop menu');
await page.keyboard.press('Escape');
await sleep(300);
await page.keyboard.press('Escape');
await sleep(500);

// Forward — clean (icon on data card)
await page.click('button[title="Step through forward pass"]');
await page.waitForSelector('.stage-timeline', { timeout: 30000 });
await sleep(700);
await page.screenshot({ path: '/tmp/icon-forward.png' });
log('saved forward clean');

await browser.close();
log('done');
