// Verify the moved sample-change UX: no top bar; a "change" button on the data
// node card opens a popup (new sample + by-class). Forward + backprop.
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

async function openMenuAndShoot(openTitle, outPath, label) {
  await page.click(`button[title="${openTitle}"]`);
  await page.waitForSelector('.stage-timeline', { timeout: 30000 });
  await sleep(700);
  // click the "change" button on the data node card
  const clicked = await page.evaluate(() => {
    const b = document.querySelector('.sample-menu-btn');
    if (b) {
      b.click();
      return true;
    }
    return false;
  });
  log(`${label}: change clicked = ${clicked}`);
  await page.waitForSelector('.sample-menu-pop', { timeout: 5000 });
  await sleep(400);
  await page.screenshot({ path: outPath });
  log(`${label}: saved ${outPath}`);
  // close the panel
  await page.keyboard.press('Escape');
  await sleep(300);
  await page.keyboard.press('Escape');
  await sleep(500);
}

await openMenuAndShoot('Step through forward pass', '/tmp/sm-forward.png', 'forward');
await openMenuAndShoot('Step through backprop', '/tmp/sm-backprop.png', 'backprop');

await browser.close();
log('done');
