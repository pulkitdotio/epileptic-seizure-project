// Isolated, localhost-only QA. No browser dependencies or existing Chrome tabs.
import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const origin = 'http://127.0.0.1:9223';
const app = 'http://127.0.0.1:8503';
const here = new URL('./', import.meta.url);
const [action = 'inspect', arg = '', width = '1366', height = '768'] = process.argv.slice(2);
let target;
if (action === 'init') {
  target = await (await fetch(`${origin}/json/new?${encodeURIComponent(app)}`, {method: 'PUT'})).json();
  await fs.writeFile(new URL('qa_target.json', here), JSON.stringify(target));
} else {
  target = JSON.parse(await fs.readFile(new URL('qa_target.json', here), 'utf8'));
}
if (new URL(target.url).origin !== app || !target.webSocketDebuggerUrl.startsWith('ws://127.0.0.1:9223/')) throw Error('Wrong QA target');
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
let id = 0;
const pending = new Map();
const downloadEvents = [];
socket.onmessage = ({data}) => {
  const msg = JSON.parse(data);
  if(msg.method?.startsWith('Browser.download')) downloadEvents.push(msg);
  if (pending.has(msg.id)) {const p = pending.get(msg.id); pending.delete(msg.id); msg.error ? p.reject(msg.error) : p.resolve(msg.result);}
};
const send = (method, params = {}) => new Promise((resolve, reject) => {pending.set(++id, {resolve, reject}); socket.send(JSON.stringify({id, method, params}));});
const evaluate = async expression => {
  const r = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
};
const pause = ms => new Promise(r => setTimeout(r, ms));
await send('Runtime.enable');
await send('Page.enable');
if (action === 'close') {await send('Browser.close'); process.exit(0);}
await send('Emulation.setDeviceMetricsOverride', {width: +width, height: +height, deviceScaleFactor: 1, mobile: false});
for (let i=0; i<60; i++) {if (await evaluate(`!!document.querySelector('.eeg-header')`)) break; await pause(500);}
if (action === 'suite') {
  const checks = [];
  await fs.mkdir(new URL('screenshots/', here), {recursive: true});
  for (const [w,h] of [[1366,768],[1920,1080]]) {
    await send('Emulation.setDeviceMetricsOverride', {width:w,height:h,deviceScaleFactor:1,mobile:false});
    for (const [slug,name] of [['overview','Overview'],['raw','Raw EEG'],['preprocessing','Preprocessing'],['annotations','Seizure Annotations'],['segmentation','Segmentation'],['ai','AI Detection'],['evaluation','Training & Evaluation'],['early','Early Prediction']]) {
      await evaluate(`(() => {const e=[...document.querySelectorAll('[data-testid="stSidebar"] label')].find(e=>e.innerText.trim()===${JSON.stringify(name)}); if(!e)throw Error('Missing navigation'); e.click();})()`);
      await pause(2000);
      await evaluate(`document.querySelector('[data-testid="stMain"]').scrollTop=0`);
      await pause(250);
      const state = await evaluate(`(() => ({heading:document.querySelector('.eeg-header h1')?.innerText,errors:[...document.querySelectorAll('[data-testid="stException"], [data-testid="stAlertContentError"]')].map(e=>e.innerText),overflow:document.documentElement.scrollWidth>innerWidth,headerTop:document.querySelector('.eeg-header')?.getBoundingClientRect().top,images:[...document.images].map(i=>({loaded:i.complete&&i.naturalWidth>0,width:i.width})),contentHeight:document.querySelector('[data-testid="stMain"]').scrollHeight}))()`);
      if(state.heading!==name||state.errors.length||state.overflow||state.headerTop<56||state.images.some(i=>!i.loaded)) throw Error(JSON.stringify(state));
      const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
      await fs.writeFile(new URL(`screenshots/${slug}_${w}x${h}.png`,here),Buffer.from(shot.data,'base64'));
      if(['raw','preprocessing','annotations','segmentation','ai','evaluation'].includes(slug)) {
        await evaluate(`document.querySelector('[data-testid="stMain"]').scrollTop=Math.max(0,document.querySelector('[data-testid="stMain"]').scrollHeight-innerHeight)`);
        await pause(250);
        const bottom=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
        await fs.writeFile(new URL(`screenshots/${slug}_lower_${w}x${h}.png`,here),Buffer.from(bottom.data,'base64'));
      }
      checks.push({width:w,height:h,section:name,...state});
    }
  }
  await fs.writeFile(new URL('browser_checks.json',here),JSON.stringify(checks,null,2));
  console.log(JSON.stringify(checks,null,2));
}
if (action === 'click') {
  console.log(await evaluate(`(() => {const text = ${JSON.stringify(arg)}; const candidates = [...document.querySelectorAll('button,label,summary')]; const e = candidates.find(e => e.innerText.trim() === text || (e.tagName==='SUMMARY'&&e.innerText.trim().endsWith(text))); if (!e) throw Error('Control not found: '+text); e.click(); return {clicked:text};})()`));
  await pause(1600);
}
if (action === 'probe') {
  console.log(await evaluate(`JSON.stringify([...document.querySelectorAll('button')].filter(e=>e.innerText==='View raw EEG').map(e=>({html:e.outerHTML,fg:getComputedStyle(e).color,bg:getComputedStyle(e).backgroundColor})))`));
  console.log(await evaluate(`JSON.stringify([...document.querySelectorAll('[data-baseweb="tag"]')].slice(0,1).map(e=>({html:e.outerHTML,color:getComputedStyle(e).color})))`));
  console.log(await evaluate(`document.querySelector('[data-testid="stMultiSelect"]')?.outerHTML.slice(0,9000)`));
  console.log(JSON.stringify(await evaluate(`(() => ({captions:[...document.querySelectorAll('[data-testid*=Caption]')].slice(0,4).map(e=>({html:e.outerHTML.slice(0,700),color:getComputedStyle(e).color})),paragraphs:[...document.querySelectorAll('p')].filter(e=>e.innerText.includes('Prerecorded')||e.innerText.includes('Saved threshold')).map(e=>({html:e.outerHTML,color:getComputedStyle(e).color,font:getComputedStyle(e).fontSize})),inputs:[...document.querySelectorAll('input')].map(e=>({type:e.type,value:e.value,label:e.getAttribute('aria-label')}))}))()`),null,2));
}
if (action === 'upload') {
  const accepted = arg === 'edf' ? '.edf' : '.txt';
  const files = arg === 'edf' ? ['C:/Main/Code/eeg-seizure-detection/data/raw/chbmit/chb01/chb01_03.edf'] : ['C:/Main/Code/eeg-seizure-detection/data/raw/chbmit/chb01/chb01-summary.txt'];
  const found = await evaluate(`!!document.querySelector('input[type="file"][accept*="${accepted}"]')`);
  if (!found) throw Error('Expected local upload control missing');
  const {root} = await send('DOM.getDocument');
  const {nodeId} = await send('DOM.querySelector', {nodeId:root.nodeId,selector:`input[type="file"][accept*="${accepted}"]`});
  await send('DOM.setFileInputFiles', {nodeId,files});
  await pause(3000);
}
if (action === 'download') {
  if(arg !== 'Download window labels (CSV)') throw Error('Only the QA label export is permitted');
  const destination = new URL('qa_exports/',here);
  await fs.mkdir(destination,{recursive:true});
  await send('Browser.setDownloadBehavior',{behavior:'allow',eventsEnabled:true,downloadPath:fileURLToPath(destination)});
  const point = await evaluate(`(() => {const e=[...document.querySelectorAll('button')].find(e=>e.innerText.trim()===${JSON.stringify(arg)}); if(!e)throw Error('Missing CSV control'); e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});
  await pause(2000);
  console.log(await fs.readdir(destination));
  console.log(JSON.stringify(downloadEvents));
}
if (action === 'scroll') {console.log(await evaluate(`(() => {const e = document.querySelector('[data-testid="stMain"]'); e.scrollTop = ${Number(arg)}; return e.scrollTop;})()`)); await pause(250);}
if (action === 'capture') {
  await pause(600);
  await fs.mkdir(new URL('screenshots/', here), {recursive: true});
  const r = await send('Page.captureScreenshot', {format:'png', captureBeyondViewport:false});
  await fs.writeFile(new URL(`screenshots/${arg}_${width}x${height}.png`, here), Buffer.from(r.data, 'base64'));
}
console.log(JSON.stringify(await evaluate(`(() => ({title:document.title, heading:document.querySelector('.eeg-header h1')?.innerText, text:document.querySelector('[data-testid="stMain"]')?.innerText, controls:[...document.querySelectorAll('button,label,summary')].map(e=>e.innerText.trim()).filter(Boolean), errors:[...document.querySelectorAll('[data-testid="stException"], [data-testid="stAlertContentError"]')].map(e=>e.innerText), overflow:document.documentElement.scrollWidth>innerWidth, images:document.images.length}))()`), null, 2));
socket.close();
