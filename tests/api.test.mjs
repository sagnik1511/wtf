import test from 'node:test';
import assert from 'node:assert/strict';
import { api } from '../viewer/api.mjs';

const metadata = {
  recording_version: 1,
  recording: { encoding: 'binary' },
  storage_format: 'binary',
  runs: [],
};
const response = (value, headers = {}, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });

test('Binary recordings on an old server report a clear restart instruction before requesting chunks', async () => {
  await assert.rejects(
    api('/api/recordings/run', async () => response(metadata)),
    /server needs restarting.*run.sh/,
  );
});

test('Current servers advertise binary capability and retain metadata', async () => {
  assert.deepEqual(
    await api('/api/recordings/run', async () =>
      response(metadata, { 'X-WTF-Replay-Protocol': 'binary-v1' }),
    ),
    metadata,
  );
});

test('Legacy JSON recordings still load from older servers', async () => {
  const legacy = {
    recording_version: 1,
    recording: { encoding: 'delta' },
    storage_format: 'delta',
  };
  assert.deepEqual(
    await api('/api/recordings/run', async () => response(legacy)),
    legacy,
  );
});

test('Legacy-to-binary converted storage is recognized independently of original configuration', async () => {
  await assert.rejects(
    api('/api/recordings/run', async () =>
      response({ ...metadata, recording: { encoding: 'delta' } }),
    ),
    /server needs restarting/,
  );
});

test('Missing binary chunks on an old server explain the version mismatch', async () => {
  await assert.rejects(
    api('/api/recordings/run/runs/1/chunks/0', async () =>
      response({ error: 'Recording or chunk not found' }, {}, 404),
    ),
    /server needs restarting/,
  );
});

test('Actual missing chunks on a current server preserve the real error', async () => {
  await assert.rejects(
    api('/api/recordings/run/runs/1/chunks/0', async () =>
      response(
        { error: 'Recording or chunk not found' },
        { 'X-WTF-Replay-Protocol': 'binary-v1' },
        404,
      ),
    ),
    /^Error: Recording or chunk not found$/,
  );
});

test('Only chunk requests negotiate the binary transport', async () => {
  let options;
  const fetch = async (path, init) => {
    options = init;
    return response([]);
  };
  await api('/api/recordings/run/runs/1/chunks/0', fetch);
  assert.equal(options.headers['X-WTF-Replay'], 'binary-v1');
  await api('/api/recordings', fetch);
  assert.equal(options, undefined);
});
