// Drive the running app over CDP with Puppeteer (connect works where Playwright's connectOverCDP does not).
//   QT_QPA_PLATFORM=offscreen QTWEBENGINE_REMOTE_DEBUGGING=9339 python app.py &
//   npm i puppeteer-core && node puppeteer_connect.mjs     (puppeteer-core downloads no browser)
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.connect({ browserURL: 'http://127.0.0.1:9339', defaultViewport: null });
try {
  const pages = await browser.pages();
  console.log('pages', pages.map(p => p.url()));
  const page = pages.find(p => p.url().startsWith('app://'));
  await page.waitForFunction('window.__ready === true');
  const box = await (await page.$('#k')).boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.evaluate(() => document.getElementById('k').drag(-50));
  console.log('text', await page.$eval('#k', e => e.textContent));
  await page.screenshot({ path: 'pshot.png' });
  console.log('OK');
} finally {
  browser.disconnect();
}
