import { Playback } from './playback.mjs';
const $ = id => document.getElementById(id);
let state, selected = null, playing = false, busy = false;
const colors = ['#7de2bd', '#ff9a77', '#83b9ff', '#ce9ef5', '#f4cf7a'];
let errorTimer;
function showError(error) { $('error').textContent = error.message; $('error').hidden = false; clearTimeout(errorTimer); errorTimer = setTimeout(() => $('error').hidden = true, 12000); }
async function api(path) {
  const response = await fetch(path);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
const player = new Playback(api);
function pause() { playing = false; $('play').textContent = 'Play recording'; }
async function operation(fn) {
  if (busy) return;
  busy = true;
  for (const id of ['step','back','load','refresh','recorded-run','timeline']) $(id).disabled = true;
  try { await fn(); } catch (e) { pause(); showError(e); }
  finally { busy = false; for (const id of ['step','back','load','refresh','recorded-run','timeline']) $(id).disabled = false; }
}
function speciesColor(name) { return colors[Object.keys(state.species).indexOf(name) % colors.length]; }
function fitWorld() {
  const canvas=$('world'),holder=canvas.parentElement;
  if(!holder.clientWidth||!holder.clientHeight)return;
  const ratio=canvas.width/canvas.height;
  const width=Math.min(holder.clientWidth,holder.clientHeight*ratio);
  // Match the element's actual bounds to the drawn board so click-to-inspect
  // stays accurate when a wide window leaves space beside the smaller board.
  const cssWidth=`${width}px`,cssHeight=`${width/ratio}px`;
  if(canvas.style.width!==cssWidth)canvas.style.width=cssWidth;
  if(canvas.style.height!==cssHeight)canvas.style.height=cssHeight;
}
new ResizeObserver(fitWorld).observe($('world').parentElement);
function drawWorld() {
  const canvas = $('world'), ctx = canvas.getContext('2d');
  const height=Math.round(canvas.width * state.height / state.width);
  if(canvas.height!==height){canvas.height=height;fitWorld();}
  else if(!canvas.style.width)fitWorld();
  const cw = canvas.width / state.width, ch = canvas.height / state.height;
  ctx.fillStyle = '#101a22'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#172630'; ctx.lineWidth = .6;
  for (let x=0; x<=state.width; x++) { ctx.beginPath();ctx.moveTo(x*cw,0);ctx.lineTo(x*cw,canvas.height);ctx.stroke(); }
  for (let y=0; y<=state.height; y++) { ctx.beginPath();ctx.moveTo(0,y*ch);ctx.lineTo(canvas.width,y*ch);ctx.stroke(); }
  for (const [x,y] of state.obstacles) { ctx.fillStyle='#3a4b5c';ctx.fillRect(x*cw+1,y*ch+1,cw-2,ch-2); }
  const capacity = state.resource_capacity;
  for (const r of state.resources) { ctx.fillStyle=`rgba(99,171,111,${.12+.66*r.quantity/capacity})`;const size=.2+.6*Math.sqrt(r.quantity/capacity);ctx.fillRect((r.x+.5-size/2)*cw,(r.y+.5-size/2)*ch,cw*size,ch*size); }
  for (const a of state.animals) {
    const x=(a.x+.5)*cw,y=(a.y+.5)*ch,r=cw*.31;
    ctx.save();ctx.translate(x,y);ctx.rotate(a.heading*Math.PI/2);ctx.fillStyle=speciesColor(a.species);
    ctx.beginPath();
    if(state.species[a.species].diet==='prey') { ctx.moveTo(0,-r*1.2);ctx.lineTo(r,r);ctx.lineTo(-r,r);ctx.closePath(); }
    else ctx.arc(0,0,r,0,Math.PI*2);
    ctx.fill();ctx.fillStyle='#13212a';ctx.beginPath();ctx.arc(0,-r*.55,1.8,0,Math.PI*2);ctx.fill();ctx.restore();
    if(a.id===selected) { ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,r+5,0,Math.PI*2);ctx.stroke(); }
  }
}
function drawDeathHighlights() {
  const canvas=$('world'),ctx=canvas.getContext('2d');
  const cw=canvas.width/state.width,ch=canvas.height/state.height;
  const markers=[...state.natural_death_highlights.map(e=>({...e,natural:true})),...state.kill_highlights];
  for(const kill of markers) {
    if(!kill.position)continue;
    const x=(kill.position[0]+.5)*cw,y=(kill.position[1]+.5)*ch;
    const radius=cw*(.65+kill.age*.035);
    ctx.save();ctx.globalAlpha=Math.max(.25,1-kill.age/14);
    const preyDeath=kill.natural&&state.species[kill.species]?.diet==='resource';
    const color=kill.natural?(preyDeath?'#7de2bd':'#f1c36b'):'#ff5268';
    ctx.fillStyle=kill.natural?(preyDeath?'#7de2bd25':'#f1c36b25'):'#ff4f6425';ctx.strokeStyle=color;ctx.lineWidth=3;
    ctx.shadowColor=color;ctx.shadowBlur=12;
    ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.shadowBlur=0;ctx.strokeStyle=kill.natural?(preyDeath?'#d4fff0':'#fff0cb'):'#ffe3df';ctx.lineWidth=3;
    const cross=cw*.22;ctx.beginPath();ctx.moveTo(x-cross,y-cross);ctx.lineTo(x+cross,y+cross);ctx.moveTo(x+cross,y-cross);ctx.lineTo(x-cross,y+cross);ctx.stroke();
    ctx.font='bold 15px system-ui';ctx.textAlign='center';
    const labelX=Math.max(30,Math.min(canvas.width-30,x)),labelY=y<50?y+radius+19:y-radius-8;
    ctx.fillStyle='#1c1019';ctx.fillRect(labelX-25,labelY-16,50,21);
    ctx.fillStyle=kill.natural?(preyDeath?'#a4f2d6':'#f7d99e'):'#ffadad';ctx.fillText(kill.natural?'DIED':'KILL',labelX,labelY);
    const hunter=kill.natural?null:state.animals.find(a=>a.id===kill.actor);
    if(hunter){ctx.strokeStyle='#ffd18a';ctx.lineWidth=3;ctx.beginPath();ctx.arc((hunter.x+.5)*cw,(hunter.y+.5)*ch,cw*.48,0,Math.PI*2);ctx.stroke();}
    ctx.restore();
  }
}
function drawChart() {
  const c=$('chart'),rect=c.getBoundingClientRect();
  if(!state||!rect.width||!rect.height)return;
  const width=rect.width,height=rect.height,dpr=window.devicePixelRatio||1;
  const pixelWidth=Math.round(width*dpr),pixelHeight=Math.round(height*dpr);
  if(c.width!==pixelWidth)c.width=pixelWidth;
  if(c.height!==pixelHeight)c.height=pixelHeight;
  const ctx=c.getContext('2d'),style=getComputedStyle(c),history=state.history;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,width,height);
  // Draw in CSS pixels so labels match surrounding text, without stretching
  // the old fixed-size bitmap into the compact chart's aspect ratio.
  ctx.font=`${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();
  ctx.textBaseline='middle';
  const popMax=Math.max(1,...history.flatMap(m=>Object.values(m.populations)));
  const foodMax=Math.max(1,...history.map(m=>m.food)),timeMax=Math.max(1,state.tick);
  const popLabel=popMax.toLocaleString(),foodLabel=Math.round(foodMax).toLocaleString();
  const left=Math.max(30,ctx.measureText(popLabel).width+10);
  const right=Math.max(36,ctx.measureText(foodLabel).width+10),top=12,bottom=23;
  const w=Math.max(1,width-left-right),h=Math.max(1,height-top-bottom);
  ctx.textAlign='right';ctx.fillText(popLabel,left-8,top);ctx.fillText('0',left-8,top+h);
  ctx.textAlign='left';ctx.fillText(foodLabel,left+w+8,top);ctx.fillText('0',left,height-7);
  ctx.textAlign='right';ctx.fillText(state.tick.toLocaleString(),left+w,height-7);
  for(let n=0;n<=3;n++) {const y=top+h*n/3;ctx.strokeStyle='#263341';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+w,y);ctx.stroke();}
  const series=[...Object.keys(state.species).map(s=>({color:speciesColor(s),max:popMax,get:m=>m.populations[s]})),{color:'#548c63',max:foodMax,get:m=>m.food}];
  for(const seriesItem of series) {ctx.strokeStyle=seriesItem.color;ctx.lineWidth=1.5;ctx.beginPath();history.forEach((m,i)=>{const x=left+w*m.tick/timeMax,y=top+h*(1-seriesItem.get(m)/seriesItem.max);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();}
}
new ResizeObserver(()=>{if(state)drawChart();}).observe($('chart'));
function text(tag, value, cls) { const el=document.createElement(tag);el.textContent=value;if(cls)el.className=cls;return el; }
function drawInspector() {
  const box=$('inspector');box.replaceChildren();const a=state.animals.find(a=>a.id===selected),b=state.selected;
  $('agent-label').textContent=a?`${a.species} #${a.id}`:'No selection';$('inspect-hint').hidden=Boolean(a);
  if(!a)return;
  if(!b){box.append(text('p',`Energy ${a.energy.toFixed(1)} · Health ${a.health.toFixed(1)} · Age ${a.age}`,'muted'));box.append(text('p','Brain diagnostics were not recorded for this frame.','muted'));return;}
  box.append(text('p',`Brain diagnostics recorded at step ${b.tick}; body state is current.`,'muted'));
  const details=text('div','','details');
  for(const [label,value] of [['Energy',a.energy.toFixed(1)],['Health',a.health.toFixed(1)],['Age',a.age],['Last reward',a.last_reward.toFixed(3)],['Exploration',b.epsilon.toFixed(3)],['Training updates',b.updates],['Own experiences',b.replay_size],['Last action',a.last_action]]) {const cell=text('div','');cell.append(text('span',label),text('strong',String(value)));details.append(cell);}box.append(details);
  box.append(text('p','Predicted future reward per action','muted'));
  const values=Object.values(b.action_values),best=Math.max(...values);
  for(const [action,value] of Object.entries(b.action_values)) {const row=text('div','','action'+(value===best?' best':''));row.append(text('span',action.replaceAll('_',' ')),text('span',value.toFixed(3)));box.append(row);}
  const features=text('details','','feature-details');features.append(text('summary','Observation features (20)'));
  features.append(text('pre',JSON.stringify(b.features.map(v=>Number(v.toFixed(3))),null,2)));box.append(features);
}
function render(snapshot = null) {
  state=snapshot??player.snapshot(selected);
  $('tick').textContent=state.tick.toLocaleString();$('run').textContent=`${state.run} / ${state.total_runs}`;
  let prey=0,predators=0;for(const [name,s] of Object.entries(state.species)) {if(s.diet==='resource')prey+=state.metrics.populations[name];else predators+=state.metrics.populations[name];}
  $('prey').textContent=prey;$('predator').textContent=predators;$('food').textContent=Math.round(state.metrics.food).toLocaleString();$('updates').textContent=state.total_updates.toLocaleString();
  const killCount=Object.values(state.metrics.kills).reduce((a,b)=>a+b,0);
  $('kills').textContent=killCount.toLocaleString();
  $('kills').parentElement.classList.toggle('kill-now',state.kill_highlights.some(k=>k.frame_index===state.frame_index));
  const latestEvent=[...state.natural_death_highlights.map(e=>({...e,natural:true})),...state.kill_highlights].sort((a,b)=>a.frame_index-b.frame_index).at(-1);
  $('kill-banner').hidden=!latestEvent;
  $('kill-banner').classList.toggle('natural-death',Boolean(latestEvent?.natural));
  $('kill-banner').classList.toggle('prey-natural-death',Boolean(latestEvent?.natural&&state.species[latestEvent.species]?.diet==='resource'));
  const species=latestEvent?.species?`${latestEvent.species[0].toUpperCase()}${latestEvent.species.slice(1)}`:'Animal';
  const banner=!latestEvent?'':latestEvent.natural
    ?`NATURAL DEATH · ${species} #${latestEvent.actor} · ${latestEvent.cause} · Step ${latestEvent.tick}`
    :`KILL · Predator #${latestEvent.actor} → prey #${latestEvent.target} · Step ${latestEvent.tick}`;
  if($('kill-banner').textContent!==banner)$('kill-banner').textContent=banner;
  $('connection').textContent=`Recorded playback · ${state.mode}`;
  $('empty').hidden=!state.done;$('empty').textContent='End of recording. Seek backward or load another saved experiment.';
  $('outcomes').textContent=`Starvation: ${state.metrics.deaths.starvation||0} · Predation: ${state.metrics.deaths.predation||0} · Hunts: ${Object.values(state.metrics.kills).reduce((a,b)=>a+b,0)}`;
  $('recorded-run').value=state.run;
  $('timeline').max=state.frame_count-1;$('timeline').value=state.frame_index;
  $('frame-label').textContent=`Frame ${state.frame_index+1} / ${state.frame_count}`;
  $('summaries').textContent=`${player.frames.length} frames loaded · one chunk in memory`;
  drawWorld();drawDeathHighlights();drawChart();drawInspector();if(state.done)pause();
}
async function openRecording() {
  const id=$('recording').value;if(!id)return;
  selected=null;await player.open(id);
  $('recorded-run').replaceChildren(...player.runs.map(run=>{const option=text('option',`Run ${run.run} · Seed ${run.seed} · ${run.frame_count} frames`);option.value=run.run;return option;}));
  $('config').value=JSON.stringify(player.metadata.config,null,2);
  $('play').disabled=false;render();
}
async function refreshCatalog() {
  const records=await api('/api/recordings'),previous=$('recording').value;
  $('recording').replaceChildren(...records.map(record=>{const option=text('option',`${record.name} · ${record.id} · ${record.status}`);option.value=record.id;return option;}));
  if(records.some(r=>r.id===previous))$('recording').value=previous;
  if(!records.length){$('connection').textContent='No recordings yet';$('empty').hidden=false;$('empty').textContent='Run Python from the terminal to create a recording, then refresh this list.';return;}
  await openRecording();
}
async function loop() {
  if(!playing)return;
  await operation(async()=>{render(await player.advance(selected));});
  const hold=$('hold-kills').checked && state && [...state.kill_highlights,...state.natural_death_highlights].some(e=>e.frame_index===state.frame_index);
  if(playing)setTimeout(loop,Math.max(1000/Number($('speed').value),hold?900:0));
}
$('play').onclick=()=>{if(playing){pause();return;}if(!state||state.done)return;playing=true;$('play').textContent='Pause';loop();};
$('step').onclick=()=>{pause();if(state)operation(async()=>{render(await player.advance(selected));});};
$('back').onclick=()=>{pause();if(state)operation(async()=>{render(await player.seek(Math.max(0,player.frameIndex-1),selected));});};
$('load').onclick=()=>{pause();operation(openRecording);};
$('refresh').onclick=()=>{pause();operation(refreshCatalog);};
$('recorded-run').onchange=()=>{pause();if(state)operation(async()=>{selected=null;render(await player.changeRun($('recorded-run').value,selected));});};
$('timeline').onchange=()=>{pause();if(state)operation(async()=>{render(await player.seek(Number($('timeline').value),selected));});};
$('speed').oninput=()=>{$('speed-value').textContent=$('speed').value;};
$('world').onclick=event=>{if(busy||!state)return;const rect=$('world').getBoundingClientRect(),x=Math.floor((event.clientX-rect.left)/rect.width*state.width),y=Math.floor((event.clientY-rect.top)/rect.height*state.height);selected=state.animals.find(a=>a.x===x&&a.y===y)?.id??null;render();};
operation(refreshCatalog);
