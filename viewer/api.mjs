import { decodeBinaryReplay } from './binary-codec.mjs';

const RESTART_MESSAGE='The playback server needs restarting to read these recordings. Stop the old ./run.sh process, start ./run.sh again, then refresh this page.';

export async function api(path, fetchImpl=globalThis.fetch){
  const chunk=path.includes('/chunks/');
  const response=await fetchImpl(path,chunk?{headers:{'X-WTF-Replay':'binary-v1'}}:undefined);
  const protocol=response.headers.get('X-WTF-Replay-Protocol');
  const value=response.headers.get('Content-Type')?.includes('application/vnd.wtf.replay-binary')
    ?decodeBinaryReplay(await response.arrayBuffer()):await response.json();
  if(!response.ok){
    if(chunk&&response.status===404&&protocol!=='binary-v1')throw new Error(RESTART_MESSAGE);
    throw new Error(value.error||`HTTP ${response.status}`);
  }
  const requiresBinary=value?.recording_version===1&&(
    value.storage_format==='binary'||value.recording?.encoding==='binary');
  if(requiresBinary&&protocol!=='binary-v1')throw new Error(RESTART_MESSAGE);
  return value;
}
