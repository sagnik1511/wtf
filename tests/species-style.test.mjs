import test from 'node:test';
import assert from 'node:assert/strict';
import {speciesStyles} from '../viewer/species-style.mjs';

test('Renamed and reordered species keep primary prey/predator styles by diet',()=>{
  const styles=speciesStyles({wolves:{diet:'prey'},deer:{diet:'resource'}});
  assert.equal(styles.get('wolves').female,'#ff9a77');assert.equal(styles.get('wolves').male,'#f6d365');
  assert.equal(styles.get('deer').female,'#7de2bd');assert.equal(styles.get('deer').male,'#60a5fa');
  assert.equal(styles.get('wolves').symbol,'▲');assert.equal(styles.get('deer').symbol,'●');
});

test('Extra species get deterministic colors without cycling the first palette',()=>{
  const species=Object.fromEntries(Array.from({length:15},(_,i)=>['animal_'+i,{diet:i%2?'prey':'resource'}]));
  const styles=speciesStyles(species);
  assert.equal(styles.size,15);
  assert.deepEqual(styles,speciesStyles(species));
  assert.equal(new Set([...styles.values()].map(s=>s.color)).size,15);
  assert([...styles.values()].every(s=>s.female!==s.male));
});

test('Optional JSON appearance drives both body and chart base colors without mutating config',()=>{
  const species={deer:{diet:'resource',display:{color:'#b4e580',male_color:'#ca94ff'}}};
  const before=JSON.stringify(species),style=speciesStyles(species).get('deer');
  assert.equal(style.female,'#b4e580');assert.equal(style.color,style.female);assert.equal(style.male,'#ca94ff');
  assert.equal(JSON.stringify(species),before);
});


test('Named predator identity stays with kill markers after the attacker leaves the current frame',async()=>{
  const {recentKills}=await import('../viewer/kill-highlights.mjs');
  const first={tick:1,animals:[{id:4,species:'wolves',x:2,y:2,heading:1}],events:[{kind:'kill',actor:4,target:8}]};
  const markers=recentKills([first],0,0);
  assert.equal(markers[0].attacker_species,'wolves');
  const carried=recentKills([{tick:2,animals:[],events:[]}],1,0,markers);
  assert.equal(carried[0].attacker_species,'wolves');
});
