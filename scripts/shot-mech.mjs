import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({ executablePath:'/usr/bin/chromium', headless:'new', args:['--no-sandbox','--window-size=1600,1000'], defaultViewport:{width:1600,height:1000} });
const p = await b.newPage();
const sleep=(m)=>new Promise(r=>setTimeout(r,m));
await p.goto('http://localhost:5173',{waitUntil:'networkidle2'});
await p.waitForSelector('.toolbar',{timeout:20000});
await p.click('button[title="Load a model preset"]');
await p.waitForSelector('.toolbar-presets-item',{timeout:5000});
await p.evaluate(()=>{const t=[...document.querySelectorAll('.toolbar-presets-item')].find(x=>/mlp/i.test(x.textContent||''));t&&t.click();});
await sleep(1500);
await p.click('button[title="Step through backprop"]');
await p.waitForSelector('.stage-detail',{timeout:30000});
await sleep(600);
// go to a Linear with a mechanism
for(let i=0;i<8;i++){const n=await p.$eval('.stage-detail-name',e=>e.textContent||'').catch(()=>'');if(/Linear/i.test(n))break;await p.keyboard.press('ArrowLeft');await sleep(250);}
await sleep(400);
// scroll the detail to reveal the trace
await p.$eval('.stage-detail', el=>{el.scrollTop = el.scrollHeight;});
await sleep(500);
await p.screenshot({path:'/tmp/ff-mech-trace.png'});
console.log('saved');
await b.close();
