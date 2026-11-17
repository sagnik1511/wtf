import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {terrainView,drawTerrain} from '../viewer/terrain.mjs';
import {Playback} from '../viewer/playback.mjs';
import {series} from '../viewer/dashboard-stats.mjs';

const map={map_version:1,name:'river',width:3,height:3,rows:['1..','...','.~.'],
  legend:{'.':{},'1':{elevation:2},'~':{water_depth:1.5,flow:[.25,-.5]}}};

test('Static terrain metadata decodes sparsely with exact heights, depths and flow',()=>{
  const original=JSON.stringify(map),t=terrainView(map);
  assert.equal(t.tiles.length,2);assert.equal(t.maxElevation,2);
  assert.deepEqual(t.cells.get(7),{x:1,y:2,elevation:0,depth:1.5,flow:[.25,-.5]});
  assert.equal(t.hasWater,true);assert.equal(t.hasElevation,true);
  assert.equal(JSON.stringify(map),original);assert.equal(terrainView(undefined),null);
});

test('Terrain renderer draws map cells and flow arrows; absent maps draw nothing',()=>{
  const calls=[];const ctx={fillRect:(...x)=>calls.push(['rect',...x]),beginPath:()=>{},
    moveTo:(...x)=>calls.push(['move',...x]),lineTo:()=>{},stroke:()=>calls.push(['stroke']),fillText:()=>{}};
  drawTerrain(ctx,null,10,10);assert.equal(calls.length,0);
  drawTerrain(ctx,terrainView(map),10,10);
  assert.deepEqual(calls.filter(x=>x[0]==='rect'),[['rect',0,0,10,10],['rect',10,20,10,10]]);
  assert.equal(calls.filter(x=>x[0]==='stroke').length,1);
});

test('Playback retains map metadata across chunks and run switches without adding it to frames',async()=>{
  const metadata={recording_version:1,name:'test',mode:'train',recording:{frames_per_chunk:2},
    config:{world:{width:3,height:3},resources:{capacity_per_patch:20},terrain:map,species:{deer:{diet:'resource'}},experiment:{runs:2}},
    runs:[{run:1,frame_count:4,obstacles:[]},{run:2,frame_count:4,obstacles:[]}]};
  const frame=tick=>({tick,animals:[],events:[],resources:[],metrics:{food:0,populations:{deer:0},births:{}}});
  const p=new Playback(async path=>{
    if(path==='/api/recordings/test')return metadata;
    if(path.endsWith('/metrics'))return [];
    const chunk=Number(path.split('/').at(-1));return [frame(chunk*2),frame(chunk*2+1)];
  });
  let state=await p.open('test');assert.equal(state.terrain,map);
  state=await p.seek(3);assert.equal(state.terrain,map);assert(p.frames.every(f=>!('terrain' in f)));
  state=await p.changeRun(2);assert.equal(state.terrain,map);
});

test('Terrain dashboard metrics remain missing when older telemetry has no terrain fields',()=>{
  const points=[{tick:1,species:{deer:{}}},{tick:2,species:{deer:{swim_attempts:2,drift_moves:3,terrain_energy:4,drowning_deaths:1}}}];
  for(const [key,value] of [['swims',2],['drift',3],['terrainEnergy',4],['drowning',1]]){
    assert.deepEqual(series(points,['deer'],key)[0].points.map(x=>x.value),[null,value]);
  }
});

test('All optional map files decode independently for the viewer',async()=>{
  for(const name of ['river_crossing','highlands','enclosed_valleys']){
    const data=JSON.parse(await readFile(new URL(`../maps/${name}.json`,import.meta.url),'utf8'));
    const t=terrainView(data);assert.equal(t.name,name);assert(t.tiles.length>0);
    assert.equal(t.hasWater,name!=='highlands');
  }
});


test('Death and kill markers use recorded final coordinates after river drift',async()=>{
  const {recentNaturalDeaths}=await import('../viewer/death-highlights.mjs');
  const {recentKills}=await import('../viewer/kill-highlights.mjs');
  const first={tick:0,animals:[{id:1,species:'deer',x:1,y:2,heading:1}],events:[]};
  const last={tick:1,animals:[],events:[{kind:'drifted',actor:1},{kind:'died',actor:1,cause:'drowning',position:[4,3]}]};
  assert.deepEqual(recentNaturalDeaths([first,last],0,1)[0].position,[4,3]);
  const kill={tick:2,animals:[],events:[{kind:'kill',actor:8,target:9,position:[3,4]}]};
  assert.deepEqual(recentKills([kill],0,0)[0].position,[3,4]);
});


test('Decision metrics preserve missing history, zero attempts and changing values',()=>{
  const points=[{tick:0,species:{deer:{}}},
    {tick:1,species:{deer:{failed_actions:0,reproduction_attempts:0,reproduction_success:null}}},
    {tick:2,species:{deer:{failed_actions:1,reproduction_attempts:4,reproduction_success:.5}}}];
  for(const [key,expected] of [['failedActions',[null,0,1]],
    ['reproductionAttempts',[null,0,4]],['reproductionSuccess',[null,null,.5]]]){
    assert.deepEqual(series(points,['deer'],key)[0].points.map(p=>p.value),expected);
  }
});
