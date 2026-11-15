import { Playback } from './playback.mjs';
import { behaviorRows } from './behavior-stats.mjs';
const $ = id => document.getElementById(id);
let state, selected = null, playing = false, busy = false;
const colors = ['#7de2bd', '#ff9a77', '#83b9ff', '#ce9ef5', '#f4cf7a'];
let errorTimer, analysisKey="", analysisRequest=0;
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
function drawBirthHighlights() {
  const canvas=$('world'),ctx=canvas.getContext('2d');
  const cw=canvas.width/state.width,ch=canvas.height/state.height;
  for(const birth of state.birth_highlights) {
    if(!birth.position)continue;
    const child=state.animals.find(a=>a.id===birth.target);
    const x=((child?.x??birth.position[0])+.5)*cw,y=((child?.y??birth.position[1])+.5)*ch;
    const radius=cw*(.52+birth.age*.035);
    ctx.save();ctx.globalAlpha=Math.max(.25,1-birth.age/14);
    ctx.strokeStyle='#6edbff';ctx.fillStyle='#6edbff20';ctx.lineWidth=3;
    ctx.shadowColor='#6edbff';ctx.shadowBlur=14;
    ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.shadowBlur=0;
    // A plus sign and a word distinguish births from crossed death markers.
    const plusX=x+cw*.38,plusY=y-cw*.38,size=cw*.13;
    ctx.strokeStyle='#d9f6ff';ctx.beginPath();ctx.moveTo(plusX-size,plusY);ctx.lineTo(plusX+size,plusY);ctx.moveTo(plusX,plusY-size);ctx.lineTo(plusX,plusY+size);ctx.stroke();
    const labelX=Math.max(38,Math.min(canvas.width-38,x));
    const labelY=y<50?y+radius+19:y-radius-8;
    ctx.font='bold 15px system-ui';ctx.textAlign='center';
    ctx.fillStyle='#102632';ctx.fillRect(labelX-35,labelY-16,70,21);
    ctx.fillStyle='#b6eeff';ctx.fillText('BORN',labelX,labelY);
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
  box.append(text('p',`Generation ${a.generation??0} · ${a.parent_id>=0?`Parent #${a.parent_id}`:'Founding animal'}`,'muted'));
  if(!b){box.append(text('p',`Energy ${a.energy.toFixed(1)} · Health ${a.health.toFixed(1)} · Age ${a.age}`,'muted'));box.append(text('p','Brain diagnostics were not recorded for this frame.','muted'));return;}
  box.append(text('p',`Brain diagnostics recorded at step ${b.tick}; body state is current.`,'muted'));
  const policy=Boolean(b.action_probabilities);
  const algorithm={ppo:'PPO',a2c:'A2C',reinforce:'REINFORCE',grpo:'GRPO (experimental)'}[b.algorithm]||'Q-learning';
  const details=text('div','','details');
  const fields=[['Brain',algorithm],['Energy',a.energy.toFixed(1)],['Health',a.health.toFixed(1)],['Age',a.age],['Last reward',a.last_reward.toFixed(3)],['Training updates',b.updates],['Last action',a.last_action]];
  if(policy){
    fields.push(['Policy entropy',b.entropy.toFixed(3)]);
    if(Number.isFinite(b.state_value))fields.push(['State value',b.state_value.toFixed(3)]);
    if(Number.isFinite(b.return_baseline))fields.push(['Return baseline',b.return_baseline.toFixed(3)]);
    if(b.buffer_kind==='group'){fields.push(['Trial groups',b.rollout_updates],['Trials per group',b.group_size],['Trial horizon',b.rollout_horizon]);if(Number.isFinite(b.training_metrics?.return_std))fields.push(['Group reward spread',b.training_metrics.return_std.toFixed(3)]);}
    else fields.push([b.buffer_kind==='episode'?'Episode buffer':'Rollout buffer',b.buffer_size??b.rollout_size]);
  }else fields.push(['Exploration',b.epsilon.toFixed(3)],['Own experiences',b.replay_size]);
  for(const [label,value] of fields){const cell=text('div','');cell.append(text('span',label),text('strong',String(value)));details.append(cell);}box.append(details);
  box.append(text('p',policy?'Action probabilities':'Predicted future reward per action','muted'));
  const actionScores=policy?b.action_probabilities:b.action_values;
  const values=Object.values(actionScores),best=Math.max(...values);
  for(const [action,value] of Object.entries(actionScores)){const row=text('div','','action'+(value===best?' best':''));row.append(text('span',action.replaceAll('_',' ')),text('span',policy?`${(value*100).toFixed(1)}%`:value.toFixed(3)));box.append(row);}
  const features=text('details','','feature-details');features.append(text('summary','Observation features (20)'));
  features.append(text('pre',JSON.stringify(b.features.map(v=>Number(v.toFixed(3))),null,2)));box.append(features);
}
async function loadBehavior() {
  const key=`${player.id}:${state.run}`;
  if(key===analysisKey)return;
  analysisKey=key;
  const request=++analysisRequest,box=$('behavior');
  box.replaceChildren(text('p','Analyzing saved frames…'));
  try{
    const report=await api(`/api/recordings/${encodeURIComponent(player.id)}/runs/${state.run}/analysis`);
    if(request!==analysisRequest)return;
    const run=report.runs[0];box.replaceChildren();
    const coverage=run.coverage.step_fraction;
    box.append(text('p',`Steps ${run.first_tick}–${run.last_tick} · ${run.complete?'Complete':'Partial'} run · Consecutive-step coverage ${coverage===null?'—':`${(coverage*100).toFixed(1)}%`}`));
    for(const species of Object.keys(run.species)){
      const section=text('div','','behavior-species');
      section.append(text('strong',`${species} · ${run.species[species].learning.algorithm}`));
      const table=document.createElement('table');table.className='behavior-table';
      const heading=document.createElement('tr');
      for(const label of ['Metric','Early','Late'])heading.append(text('th',label));
      const head=document.createElement('thead');head.append(heading);table.append(head);
      const body=document.createElement('tbody');
      for(const row of behaviorRows(run,species)){
        const tr=document.createElement('tr');
        for(const value of row)tr.append(text('td',value));
        body.append(tr);
      }
      table.append(body);section.append(table);
      const stats=run.species[species],learning=stats.learning;
      section.append(text('p',`Observed optimizer updates: ≥${learning.observed_optimizer_updates} · Mean age at death: ${stats.mean_completed_lifespan===null?'—':stats.mean_completed_lifespan.toFixed(0)} steps · ${stats.survivors} survivors (excluded from lifespan mean).`));
      const budget=stats.energy_budget;
      section.append(text('p',`Food supplied ${budget.food_input_share===null?'—':`${(budget.food_input_share*100).toFixed(1)}%`} of recorded energy inputs. Birth grants added ${budget.net_reproduction_energy.toFixed(0)} net energy after parent costs.`));
      if(learning.algorithm==='grpo'){
        section.append(text('p',`New trial groups observed: ${learning.observed_grpo_groups} · Equal-return groups: ${learning.grpo_zero_variance_fraction===null?'—':`${(learning.grpo_zero_variance_fraction*100).toFixed(1)}%`} of ${learning.classified_grpo_groups} classified. Equal returns produce no GRPO update.`));
      }
      box.append(section);
    }
    box.append(text('p',`Early: ${run.windows[0].start_tick}–${run.windows[0].end_tick}; late: ${run.windows.at(-1).start_tick}–${run.windows.at(-1).end_tick}. Policy entropy of 100% means uniform probabilities. Updates exclude inherited counter baselines. Successful actions are reconstructed where possible on death steps. Compare frozen policies on held-out seeds to establish learning.`));
  }catch(error){if(request===analysisRequest)box.replaceChildren(text('p',`Analysis unavailable: ${error.message}`));}
}
function render(snapshot = null) {
  state=snapshot??player.snapshot(selected);
  $('tick').textContent=state.tick.toLocaleString();$('run').textContent=`${state.run} / ${state.total_runs}`;
  let prey=0,predators=0;for(const [name,s] of Object.entries(state.species)) {if(s.diet==='resource')prey+=state.metrics.populations[name];else predators+=state.metrics.populations[name];}
  $('prey').textContent=prey;$('predator').textContent=predators;$('food').textContent=Math.round(state.metrics.food).toLocaleString();$('updates').textContent=state.total_updates.toLocaleString();
  const killCount=Object.values(state.metrics.kills).reduce((a,b)=>a+b,0);
  $('kills').textContent=killCount.toLocaleString();
  $('births').textContent=Object.values(state.metrics.births??{}).reduce((a,b)=>a+b,0).toLocaleString();
  $('births').parentElement.classList.toggle('birth-now',state.birth_highlights.some(b=>b.frame_index===state.frame_index));
  const latestBirth=state.birth_highlights.at(-1);
  $('birth-banner').hidden=!latestBirth;
  if(latestBirth){
    const count=state.birth_highlights.filter(b=>b.frame_index===latestBirth.frame_index).length;
    const species=latestBirth.species??'animal';
    const message=`BORN · ${species} #${latestBirth.target} · Parent #${latestBirth.actor} · Step ${latestBirth.tick}${count>1?` · ${count} births shown`:''}`;
    if($('birth-banner').textContent!==message)$('birth-banner').textContent=message;
  }
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
  $('outcomes').textContent=`Births: ${Object.values(state.metrics.births??{}).reduce((a,b)=>a+b,0)} · Starvation: ${state.metrics.deaths.starvation||0} · Predation: ${state.metrics.deaths.predation||0} · Hunts: ${Object.values(state.metrics.kills).reduce((a,b)=>a+b,0)}`;
  $('recorded-run').value=state.run;
  $('timeline').max=state.frame_count-1;$('timeline').value=state.frame_index;
  $('frame-label').textContent=`Frame ${state.frame_index+1} / ${state.frame_count}`;
  $('summaries').textContent=`${player.frames.length} frames loaded · one chunk in memory`;
  loadBehavior();
  drawWorld();drawBirthHighlights();drawDeathHighlights();drawChart();drawInspector();if(state.done)pause();
}
async function openRecording() {
  const id=$('recording').value;if(!id)return;
  selected=null;analysisKey="";await player.open(id);
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
  if(playing)setTimeout(loop,1000/Number($('speed').value));
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
