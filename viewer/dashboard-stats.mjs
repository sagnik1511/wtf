// Bounded scalar telemetry cache; never retains world frames or future-visible data.
export const PAGE_SIZE=128;
export const metrics={
  population:{label:'Population',get:s=>s.population},
  energy:{label:'Mean energy',percent:true,get:s=>s.energy},
  health:{label:'Mean health',percent:true,get:s=>s.health},
  failedActions:{label:'Failed action attempts / step',get:s=>s.failed_actions},
  reproductionAttempts:{label:'Reproduction attempts / step',get:s=>s.reproduction_attempts},
  reproductionSuccess:{label:'Successful reproducing parents / attempt',percent:true,get:s=>s.reproduction_success},
  births:{label:'Births / step',get:s=>s.births},
  deaths:{label:'Deaths / step',get:s=>s.deaths},
  kills:{label:'Kills / step',get:s=>s.kills},
  predation:{label:'Deaths from predation / step',get:s=>s.predation_deaths},
  drowning:{label:'Deaths from drowning / step',get:s=>s.drowning_deaths},
  swims:{label:'Swim attempts / step',get:s=>s.swim_attempts},
  drift:{label:'Current drift moves / step',get:s=>s.drift_moves},
  terrainEnergy:{label:'Terrain energy demand / step',get:s=>s.terrain_energy},
  starvation:{label:'Deaths from starvation / step',get:s=>s.starvation_deaths},
  starving:{label:'Starving animals',get:s=>s.starving},
  attempts:{label:'Attack / eat attempts per step',get:s=>s.attempts},
  success:{label:'Kills / attack · feeds / eat',percent:true,get:s=>s.success},
  actionCoverage:{label:'Recovered action coverage',percent:true,get:s=>s.action_coverage},
  reward:{label:'Mean reward · surviving actors',get:s=>s.reward},
  age:{label:'Mean age (steps)',get:s=>s.mean_age},
  generation:{label:'Mean generation',get:s=>s.mean_generation},
  clusters:{label:'Spatial groups (2+ animals)',get:s=>s.groups.clusters},
  solitary:{label:'Solitary animals',get:s=>s.groups.solitary},
  largest:{label:'Largest spatial group',get:s=>s.groups.largest},
  probability:{label:'Mean action probability',percent:true,sampled:true,action:true,get:(s,action)=>s.predictions[action]},
  qValue:{label:'Mean predicted Q value',sampled:true,action:true,get:(s,action)=>s.q_values[action]},
  stateValue:{label:'Mean predicted state value',sampled:true,get:s=>s.state_value},
  entropy:{label:'Policy entropy',sampled:true,percent:true,get:s=>s.entropy},
  exploration:{label:'Q exploration rate',sampled:true,percent:true,get:s=>s.epsilon},
};

export function series(points,names,metric,action='rest') {
  const definition=metrics[metric];if(!definition)throw new Error('Unknown telemetry metric');
  return names.map(name=>({name,points:points.map(point=>({tick:point.tick,consecutive:point.consecutive,
    value:point.species[name]?definition.get(point.species[name],action)??null:null}))}));
}

export class TelemetryCache {
  constructor(loadJSON,maxPages=12){this.loadJSON=loadJSON;this.maxPages=maxPages;this.pages=new Map();this.pending=new Map();}
  async page(id,run,page,count){
    const key=`${id}:${run}:${page}`,existing=this.pages.get(key),expected=Math.min(PAGE_SIZE,count-page*PAGE_SIZE);
    if(existing&&existing.points.length>=expected){this.pages.delete(key);this.pages.set(key,existing);return existing;}
    if(this.pending.has(key))return this.pending.get(key);
    const promise=this.loadJSON(`/api/recordings/${encodeURIComponent(id)}/runs/${run}/telemetry/${page}`).then(result=>{
      if(result.page_size!==PAGE_SIZE||result.start_frame!==page*PAGE_SIZE||!Array.isArray(result.points)||!result.points.length||result.points.length>PAGE_SIZE)throw new Error('Invalid per-step telemetry page');
      this.pages.set(key,result);
      while(this.pages.size>this.maxPages)this.pages.delete(this.pages.keys().next().value);
      return result;
    }).catch(error=>{
      if(error.message==='Recording endpoint not found'){
        throw new Error('The playback server needs an update. Restart ./run.sh, then refresh this page.');
      }
      throw error;
    }).finally(()=>this.pending.delete(key));
    this.pending.set(key,promise);return promise;
  }
  async history(id,run,index,count,length=512){
    if(!Number.isInteger(length)||length<1||length>1024)throw new Error('Invalid telemetry window');
    const start=Math.max(0,index-length+1),first=Math.floor(start/PAGE_SIZE),last=Math.floor(index/PAGE_SIZE),pages=[];
    for(let page=first;page<=last;page++)pages.push(page);
    const loaded=await Promise.all(pages.map(page=>this.page(id,run,page,count)));
    return loaded.flatMap(page=>page.points.filter((_,i)=>page.start_frame+i>=start&&page.start_frame+i<=index));
  }
}
