// WTFB v1: typed numeric arrays, symbol references and structured record columns.
const MAX_RECORD=256*1024*1024;
const utf8=new TextDecoder('utf-8',{fatal:true});
export function decodeBinaryReplay(buffer){
  const bytes=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer);
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  let position=0;
  const symbols=[];
  const take=n=>{if(!Number.isSafeInteger(n)||n<0||position+n>bytes.length)throw new Error('Truncated binary value');const start=position;position+=n;return start;};
  const uint=()=>{let value=0,factor=1;for(let i=0;i<8;i++){const b=bytes[take(1)];if(i===7&&(b&127)>=16){const exact=BigInt(value)+BigInt(b&127)*BigInt(factor);if((b&128)||exact>2n*BigInt(Number.MAX_SAFE_INTEGER)+1n)throw new Error('Invalid binary integer');return exact;}value+=(b&127)*factor;if(!(b&128)){if(value>2*Number.MAX_SAFE_INTEGER+1)throw new Error('Invalid binary integer');return value;}factor*=128;}throw new Error('Invalid binary varint');};
  const length=()=>{const n=uint();if(!Number.isSafeInteger(n)||n>MAX_RECORD)throw new Error('Binary collection too large');return n;};
  const put=(o,k,v)=>Object.defineProperty(o,k,{value:v,enumerable:true,writable:true,configurable:true});
  const integer=()=>{const n=uint();return typeof n==='bigint'?Number(n%2n?-(n/2n)-1n:n/2n):(n%2?-(Math.floor(n/2))-1:n/2);};
  let limit=bytes.length;
  const get=(depth=0)=>{
    if(depth>128)throw new Error('Binary nesting too deep');
    const tag=bytes[take(1)];
    if(tag===0)return null;if(tag===1)return false;if(tag===2)return true;
    if(tag===3)return integer();
    if(tag===4){const n=view.getFloat64(take(8),true);if(!Number.isFinite(n))throw new Error('Non-finite binary number');return n;}
    if(tag===5){const n=length(),value=utf8.decode(bytes.subarray(take(n),position));symbols.push(value);return value;}
    if(tag===6){const i=uint();if(i>=symbols.length)throw new Error('Invalid binary symbol');return symbols[i];}
    if(tag===7){const n=length(),a=[];for(let i=0;i<n;i++)a.push(get(depth+1));return a;}
    if(tag===8){const n=length(),o={};for(let i=0;i<n;i++){const k=get(depth+1);if(typeof k!=='string'||Object.hasOwn(o,k))throw new Error('Invalid binary map key');put(o,k,get(depth+1));}return o;}
    if(tag===9){const n=length(),a=[];for(let i=0;i<n;i++)a.push(integer());return a;}
    if(tag===10||tag===15){const width=tag===10?8:4,n=length(),a=[],start=take(n*width);for(let i=0;i<n;i++){const v=tag===10?view.getFloat64(start+i*width,true):view.getFloat32(start+i*width,true);if(!Number.isFinite(v))throw new Error('Non-finite binary array');a.push(v);}return a;}
    if(tag===14){const n=length();if(!n||n>128)throw new Error('Binary integer too large');const start=take(n);let value=0n;for(let i=n-1;i>=0;i--)value=value*256n+BigInt(bytes[start+i]);if(bytes[start+n-1]&128)value-=1n<<BigInt(n*8);return value;}
    if(tag===12||tag===16||tag===17){const width=tag===16?1:tag===17?2:4,n=length(),a=[],start=take(n*width);for(let i=0;i<n;i++)a.push(width===1?view.getUint8(start+i):width===2?view.getUint16(start+i*2,true):view.getUint32(start+i*4,true));return a;}
    if(tag===11){const n=length(),fields=get(depth+1);if(!Array.isArray(fields)||!fields.length||fields.some(k=>typeof k!=='string')||new Set(fields).size!==fields.length||n>limit)throw new Error('Invalid binary column fields');const columns=fields.map(()=>get(depth+1));if(columns.some(c=>!Array.isArray(c)||c.length!==n))throw new Error('Invalid binary column length');return Array.from({length:n},(_,i)=>{const o={};fields.forEach((k,j)=>put(o,k,columns[j][i]));return o;});}
    throw new Error('Unknown binary tag');
  };
  for(const b of [87,84,70,66,1])if(bytes[take(1)]!==b)throw new Error('Unsupported binary file version');
  const values=[];
  while(position<bytes.length){const n=view.getUint32(take(4),true);if(!n||n>MAX_RECORD||position+n>bytes.length)throw new Error('Invalid binary record size');limit=position+n;const value=get();if(position!==limit)throw new Error('Trailing binary record data');values.push(value);}
  const header=values.shift();
  if(header?.kind!=='replay'||!header.delta)throw new Error('Invalid binary replay header');
  const footer=values.pop();
  if(footer?.kind!=='replay_end'||footer.frames!==values.length)throw new Error('Missing or invalid binary replay footer');
  return {...header.delta,frames:values};
}
