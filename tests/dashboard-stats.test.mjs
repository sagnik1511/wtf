import test from 'node:test';
import assert from 'node:assert/strict';
import { TelemetryCache, series } from '../viewer/dashboard-stats.mjs';

function setup(count=400,maxPages=12){
  const calls=[];
  const load=async path=>{
    calls.push(path);const page=Number(path.split('/').at(-1));
    return {page,page_size:128,start_frame:page*128,points:Array.from({length:Math.min(128,count-page*128)},(_,i)=>({tick:page*128+i,consecutive:true,species:{prey:{population:page*128+i,predictions:i%10===0?{rest:.3}:{},q_values:{}}}}))};
  };
  return {cache:new TelemetryCache(load,maxPages),calls,setCount:n=>count=n};
}
test('History follows each step, hides future points and restores past values on seek',async()=>{
  const {cache,calls}=setup();
  let points=await cache.history('a',1,125,400,128);
  assert.equal(points.at(-1).tick,125);assert.equal(points.length,126);
  points=await cache.history('a',1,126,400,128);assert.equal(points.at(-1).tick,126);assert.equal(calls.length,1);
  points=await cache.history('a',1,131,400,128);assert.equal(points[0].tick,4);assert.equal(points.at(-1).tick,131);
  points=await cache.history('a',1,5,400,128);assert.deepEqual(points.map(p=>p.tick),[0,1,2,3,4,5]);
});
test('Run keys isolate populations and page count stays bounded',async()=>{
  const {cache,calls}=setup(1500,3);
  for(let index=127;index<1500;index+=128)await cache.history('a',1,index,1500,128);
  assert(cache.pages.size<=3);
  await cache.history('a',2,5,1500,128);assert(calls.at(-1).includes('/runs/2/'));
});
test('Writing recordings extend cached partial pages and concurrent requests deduplicate',async()=>{
  const {cache,calls,setCount}=setup(5);
  await Promise.all([cache.history('a',1,4,5),cache.history('a',1,3,5)]);assert.equal(calls.length,1);
  setCount(8);const points=await cache.history('a',1,7,8);assert.equal(points.length,8);assert.equal(calls.length,2);
});
test('Predictions remain missing between samples and Q values are separate from probabilities',()=>{
  const points=[{tick:0,consecutive:false,species:{prey:{predictions:{rest:.3},q_values:{}}}},{tick:1,consecutive:true,species:{prey:{predictions:{},q_values:{rest:-3}}}}];
  assert.deepEqual(series(points,['prey'],'probability').at(0).points.map(p=>p.value),[.3,null]);
  assert.deepEqual(series(points,['prey'],'qValue').at(0).points.map(p=>p.value),[null,-3]);
});

test('An older running server produces a clear restart instruction',async()=>{
  const cache=new TelemetryCache(async()=>{throw new Error('Recording endpoint not found');});
  await assert.rejects(cache.history('a',1,0,1),/Restart \.\/run\.sh/);
  assert.equal(cache.pending.size,0);
  assert.equal(cache.pages.size,0);
});
