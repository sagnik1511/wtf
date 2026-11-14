import test from 'node:test';
import assert from 'node:assert/strict';
import { Playback } from '../viewer/playback.mjs';

function setup() {
  const manifest = {recording_version:1,name:'test',mode:'train',recording:{frames_per_chunk:3},config:{world:{width:8,height:8},resources:{capacity_per_patch:20},species:{prey:{diet:'resource'}}},runs:[{run:1,obstacles:[],frame_count:8},{run:2,obstacles:[[1,1]],frame_count:2}]};
  const calls=[];
  const p=new Playback(async path=>{
    calls.push(path);
    assert.ok(!path.includes('/api/step'));
    if(path==='/api/recordings/test')return manifest;
    if(path.endsWith('/metrics')){const run=Number(path.match(/runs\/(\d+)/)[1]);return Array.from({length:manifest.runs[run-1].frame_count},(_,tick)=>({tick,populations:{prey:4},food:10*run}));}
    const match=path.match(/runs\/(\d+)\/chunks\/(\d+)$/);
    const run=Number(match[1]),chunk=Number(match[2]),count=manifest.runs[run-1].frame_count;
    return Array.from({length:Math.min(3,count-chunk*3)},(_,i)=>({tick:chunk*3+i,animals:[],resources:[],metrics:{tick:chunk*3+i,populations:{prey:4},food:10*run},...(i===0?{diagnostics:{'7':{tick:chunk*3,action_values:{rest:1}}}}:{})}));
  });
  return {p,calls};
}

test('Playback reads one bounded chunk and reuses it for local frames',async()=>{
  const {p,calls}=setup();await p.open('test');assert.equal(calls.length,3);
  await p.advance();await p.advance();assert.equal(calls.length,3);
  assert.equal(p.snapshot(7).selected.tick,0);
  await p.advance();assert.equal(calls.length,4);assert.equal(p.frames.length,3);
  assert.equal(p.snapshot().tick,3);assert.equal(p.snapshot(7).selected.tick,3);
  await p.seek(7);assert.equal(p.frames.length,2);assert.equal(p.snapshot().tick,7);
  await p.seek(0);assert.equal(p.frames.length,3);assert.equal(p.snapshot().tick,0);
});
test('Playback handles run boundaries, end, and arbitrary seeking',async()=>{
  const {p}=setup();await p.open('test');await p.seek(7);assert.equal(p.atEnd,false);
  await p.advance();assert.equal(p.run.run,2);assert.equal(p.frameIndex,0);
  await p.advance();assert.equal(p.atEnd,true);
  await p.advance();assert.equal(p.frameIndex,1);
  await p.changeRun(1);assert.equal(p.snapshot().tick,0);
  await assert.rejects(p.seek(-1));await assert.rejects(p.seek(8));await assert.rejects(p.changeRun(99));
});

test('Kill markers identify removed prey from the attacking predator direction', async()=>{
  const { recentKills }=await import('../viewer/kill-highlights.mjs');
  const frame={tick:10,animals:[{id:4,x:3,y:2,heading:1}],events:[{kind:'kill',actor:4,target:8},{kind:'died',actor:9,cause:'starvation'}]};
  const markers=recentKills([frame],0,0);
  assert.equal(markers.length,1);
  assert.deepEqual(markers[0].position,[4,2]);
  assert.equal(markers[0].actor,4);assert.equal(markers[0].target,8);
  // Do not invent a position if the hunter also disappeared this step.
  assert.equal(recentKills([{...frame,animals:[]}],0,0)[0].position,null);
});
test('Kill markers expire, do not show future events, and stay bounded', async()=>{
  const { recentKills, KILL_HOLD_FRAMES }=await import('../viewer/kill-highlights.mjs');
  const frames=Array.from({length:20},(_,tick)=>({tick,animals:[{id:4,x:2,y:2,heading:0}],events:tick===2?[{kind:'kill',actor:4,target:8}]:[]}));
  assert.equal(recentKills(frames,0,1).length,0);
  assert.equal(recentKills(frames,0,2).length,1);
  assert.equal(recentKills(frames,0,2+KILL_HOLD_FRAMES).length,0);
  const crowded={...frames[2],events:Array.from({length:100},(_,i)=>({kind:'kill',actor:4,target:i}))};
  assert.equal(recentKills([crowded],0,0).length,32);
});
test('Sequential chunk transitions retain kill markers without retaining frames',async()=>{
  const metadata={recording_version:1,name:'kills',mode:'train',recording:{frames_per_chunk:3},config:{world:{width:8,height:8},resources:{capacity_per_patch:20},species:{prey:{diet:'resource'}}},runs:[{run:1,obstacles:[],frame_count:6},{run:2,obstacles:[],frame_count:3}]};
  const p=new Playback(async path=>{
    if(path==='/api/recordings/kills')return metadata;
    if(path.endsWith('/metrics')){const run=Number(path.match(/runs\/(\d+)/)[1]);return Array.from({length:metadata.runs[run-1].frame_count},(_,tick)=>({tick,populations:{prey:4},food:10*run}));}
    const match=path.match(/runs\/(\d+)\/chunks\/(\d+)$/),run=Number(match[1]),chunk=Number(match[2]);
    return Array.from({length:3},(_,i)=>({tick:chunk*3+i,animals:[{id:4,x:3,y:2,heading:1}],metrics:{tick:chunk*3+i,populations:{prey:4},food:10*run},events:run===1&&chunk===0&&i===2?[{kind:'kill',actor:4,target:8}]:[]}));
  });
  await p.open('kills');await p.seek(2);
  assert.equal(p.snapshot().kill_highlights.length,1);
  await p.advance();assert.equal(p.snapshot().kill_highlights[0].age,1);
  assert.equal(p.frames.length,3);
  await p.seek(1);assert.equal(p.snapshot().kill_highlights.length,0);
  await p.seek(2);await p.changeRun(2);
  assert.equal(p.snapshot().kill_highlights.length,0);
});


test('Chart history survives chunk boundaries and random seeks without future data',async()=>{
  const {p,calls}=setup();await p.open('test');
  assert.deepEqual(p.snapshot().history.map(m=>m.tick),[0]);
  await p.seek(2);await p.advance();
  assert.deepEqual(p.snapshot().history.map(m=>m.tick),[0,1,2,3]);
  assert.equal(p.frames.length,3);
  await p.seek(7);
  assert.deepEqual(p.snapshot().history.map(m=>m.tick),[0,1,2,3,4,5,6,7]);
  assert.equal(p.frames.length,2);
  await p.seek(1);
  assert.deepEqual(p.snapshot().history.map(m=>m.tick),[0,1]);
  assert.equal(calls.filter(path=>path.endsWith('/metrics')).length,1);
  await p.changeRun(2);
  assert.deepEqual(p.snapshot().history,[{tick:0,populations:{prey:4},food:20}]);
});

test('Chart appends the exact current frame between sampled metrics',async()=>{
  const {p}=setup();await p.open('test');
  p.metrics=[{tick:0,populations:{prey:4},food:10},{tick:4,populations:{prey:4},food:10}];
  await p.seek(3);
  assert.deepEqual(p.snapshot().history.map(m=>m.tick),[0,3]);
  await p.seek(4);
  assert.deepEqual(p.snapshot().history.map(m=>m.tick),[0,4]);
  assert.equal(p.metrics.length,2);
});

test('Natural death markers recover final positions for prey and predators',async()=>{
  const {recentNaturalDeaths}=await import('../viewer/death-highlights.mjs');
  const prey={id:7,species:'prey',x:1,y:2,heading:1},predator={id:8,species:'predator',x:5,y:6,heading:3};
  const frames=[
    {tick:0,animals:[prey,predator],events:[]},
    {tick:1,animals:[predator],events:[{kind:'moved',actor:7},{kind:'died',actor:7,cause:'starvation'}]},
    {tick:2,animals:[],events:[{kind:'movement_blocked',actor:8},{kind:'died',actor:8,cause:'starvation'}]}
  ];
  const markers=recentNaturalDeaths(frames,0,2);
  assert.deepEqual(markers.map(m=>m.position),[[2,2],[5,6]]);
  assert.deepEqual(markers.map(m=>m.species),['prey','predator']);
  assert.deepEqual(markers.map(m=>m.age),[1,0]);
});
test('Natural deaths exclude predation, stay bounded, and never show future events',async()=>{
  const {recentNaturalDeaths}=await import('../viewer/death-highlights.mjs');
  const animal={id:7,species:'prey',x:1,y:2,heading:1};
  const frames=Array.from({length:20},(_,tick)=>({tick,animals:tick===0?[animal]:[],events:tick===1?[{kind:'died',actor:7,cause:'starvation'},{kind:'died',actor:8,cause:'predation'},{kind:'kill',actor:9,target:10},{kind:'died',actor:10,cause:''}]:[]}));
  assert.equal(recentNaturalDeaths(frames,0,0).length,0);
  assert.equal(recentNaturalDeaths(frames,0,1).length,1);
  assert.equal(recentNaturalDeaths(frames,0,15).length,0);
  const crowded={tick:0,animals:[],events:Array.from({length:100},(_,actor)=>({kind:'died',actor,cause:'starvation'}))};
  assert.equal(recentNaturalDeaths([crowded],0,0).length,32);
});
test('Sampled recordings do not invent natural death positions across missing steps',async()=>{
  const {recentNaturalDeaths}=await import('../viewer/death-highlights.mjs');
  const animal={id:7,species:'prey',x:1,y:2,heading:1};
  const frames=[{tick:0,animals:[animal],events:[]},{tick:10,animals:[],events:[{kind:'died',actor:7,cause:'starvation'}]}];
  assert.equal(recentNaturalDeaths(frames,0,1)[0].position,null);
  assert.equal(recentNaturalDeaths(frames,0,1)[0].species,'prey');
  frames[1].events[0].position=[3,4];
  assert.deepEqual(recentNaturalDeaths(frames,0,1)[0].position,[3,4]);
});
test('Natural death effects locate first-frame chunk deaths and survive seeking',async()=>{
  const metadata={recording_version:1,name:'deaths',mode:'train',recording:{frames_per_chunk:3},config:{world:{width:8,height:8},resources:{capacity_per_patch:20},species:{prey:{diet:'resource'}}},runs:[{run:1,obstacles:[],frame_count:6},{run:2,obstacles:[],frame_count:3}]};
  const p=new Playback(async path=>{
    if(path==='/api/recordings/deaths')return metadata;
    const run=Number(path.match(/runs\/(\d+)/)[1]);
    if(path.endsWith('/metrics'))return Array.from({length:metadata.runs[run-1].frame_count},(_,tick)=>({tick,populations:{prey:1},food:10}));
    const chunk=Number(path.match(/chunks\/(\d+)$/)[1]);
    return Array.from({length:3},(_,i)=>{
      const tick=chunk*3+i;
      return {tick,animals:run===1&&tick>=3?[]:[{id:7,species:'prey',x:1,y:2,heading:1}],metrics:{tick,populations:{prey:1},food:10},events:run===1&&tick===3?[{kind:'moved',actor:7},{kind:'died',actor:7,cause:'starvation'}]:[]};
    });
  });
  await p.open('deaths');await p.seek(2);await p.advance();
  assert.deepEqual(p.snapshot().natural_death_highlights[0].position,[2,2]);
  assert.equal(p.snapshot().kill_highlights.length,0);
  await p.advance();assert.equal(p.snapshot().natural_death_highlights[0].age,1);
  await p.seek(0);assert.equal(p.snapshot().natural_death_highlights.length,0);
  // Directly seek to the first frame of a new chunk: read the preceding
  // positions rather than keeping the old world chunk in memory.
  await p.seek(3);
  assert.deepEqual(p.snapshot().natural_death_highlights[0].position,[2,2]);
  assert.equal(p.frames.length,3);assert.equal(p.previousAgents.length,1);
  await p.changeRun(2);assert.equal(p.snapshot().natural_death_highlights.length,0);
});
