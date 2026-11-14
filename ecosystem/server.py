"""Read-only playback server. Never imports or instantiates simulation/brains."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from .recording import RecordingStore

ROOT = Path(__file__).resolve().parent.parent


class State:
    def __init__(self, results=None):
        self.recordings = RecordingStore(results if results is not None else ROOT / 'results')


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, status, data, content_type='application/json'):
            body = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            parts = path.strip('/').split('/')
            try:
                if path == '/api/recordings':
                    return self.send(200, state.recordings.catalog())
                if len(parts) >= 3 and parts[:2] == ['api', 'recordings']:
                    if len(parts) == 3:
                        return self.send(200, state.recordings.metadata(parts[2]))
                    if len(parts) == 6 and parts[3] == 'runs' and parts[5] == 'metrics':
                        return self.send(200, state.recordings.metrics(parts[2], int(parts[4])))
                    if len(parts) == 7 and parts[3] == 'runs' and parts[5] == 'chunks':
                        chunk = state.recordings.chunk(parts[2], int(parts[4]), int(parts[6]))
                        return self.send(200, chunk.read_bytes())
                    return self.send(404, {'error': 'Recording endpoint not found'})
                target = (ROOT / 'viewer' / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
                if (ROOT / 'viewer').resolve() not in target.parents or not target.is_file():
                    return self.send(404, {'error': 'Not found'})
                return self.send(200, target.read_bytes(), mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
            except FileNotFoundError:
                return self.send(404, {'error': 'Recording or chunk not found'})
            except (ValueError, OSError, KeyError) as error:
                return self.send(400, {'error': str(error)})

        def do_POST(self):
            return self.send(405, {'error': 'Playback is read-only. Run simulations with python -m ecosystem.cli.'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Serve saved ecosystem recordings without loading simulation brains')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(State(args.results)))
    print(f'Ecosystem playback: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
