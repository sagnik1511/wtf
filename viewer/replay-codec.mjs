// A format decoder only: reconstruction never advances simulation rules.
const CODEC='wtf-delta-v1';
const clone=value=>structuredClone(value);
const put=(target,key,value)=>Object.defineProperty(target,key,{value,enumerable:true,writable:true,configurable:true});
const entityKey=(record,kind)=>kind==='animals'?record.id:`${record.x},${record.y}`;
const patchKey=(key,kind)=>kind==='animals'?key:`${key[0]},${key[1]}`;

function applyMapping(target,patch){
  for(const key of patch.drop??[])delete target[key];
  for(const [key,value] of Object.entries(patch.set??{}))put(target,key,clone(value));
}
function applyEntities(records,patch,fields,kind){
  const entities=new Map(records.map(record=>[entityKey(record,kind),record]));
  for(const key of patch.drop??[]){
    if(!entities.delete(patchKey(key,kind)))throw new Error('Missing replay entity');
  }
  for(const type of ['replace','add']){
    for(const record of patch[type]??[])entities.set(entityKey(record,kind),clone(record));
  }
  for(const [key,mask,values] of patch.update??[]){
    if(!Number.isSafeInteger(mask)||mask<0||mask>=2**fields.length||!Array.isArray(values))throw new Error('Invalid replay field mask');
    const record=entities.get(patchKey(key,kind));
    if(!record)throw new Error('Missing replay entity');
    let index=0;
    fields.forEach((field,i)=>{
      if(Math.floor(mask/2**i)%2){
        if(index>=values.length)throw new Error('Missing replay field value');
        put(record,field,clone(values[index++]));
      }
    });
    if(index!==values.length)throw new Error('Extra replay field values');
  }
  if(patch.order){
    const order=patch.order.map(key=>patchKey(key,kind));
    if(order.length!==entities.size||new Set(order).size!==entities.size||order.some(key=>!entities.has(key)))throw new Error('Invalid replay entity order');
    return order.map(key=>entities.get(key));
  }
  return [...entities.values()];
}
export function decodeChunk(payload){
  if(Array.isArray(payload))return payload;
  if(!payload||payload.codec!==CODEC||!Array.isArray(payload.frames))throw new Error('Unsupported replay chunk encoding');
  const tables={animals:payload.animal_fields,resources:payload.resource_fields};
  for(const fields of Object.values(tables)){
    if(!Array.isArray(fields)||fields.length>52||fields.some(field=>typeof field!=='string')||new Set(fields).size!==fields.length)throw new Error('Invalid replay field table');
  }
  const frames=[];
  let previous;
  for(const packet of payload.frames){
    if(!packet||typeof packet!=='object'||Array.isArray(packet))throw new Error('Invalid replay frame');
    let current;
    if(previous===undefined)current=clone(packet);
    else{
      const overwritten=new Set([...(packet.drop??[]),...Object.keys(packet.set??{})]);
      current={};
      for(const [key,value] of Object.entries(previous))if(!overwritten.has(key))put(current,key,clone(value));
      applyMapping(current,packet);
      if(packet.metrics){current.metrics??={};applyMapping(current.metrics,packet.metrics);}
      for(const kind of Object.keys(tables))if(packet[kind])current[kind]=applyEntities(current[kind]??[],packet[kind],tables[kind],kind);
    }
    frames.push(current);previous=current;
  }
  return frames;
}
