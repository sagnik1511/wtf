"""Read-only playback server. Never imports or instantiates simulation/brains."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
import gzip
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from wtf.storage.recording import RecordingStore
from wtf.inspection.explanations import ExplanationStore
from wtf.storage.replay_io import read_chunk_bytes, load_chunk
from wtf.resources import asset


class State:
    """Share recording and explanation services between HTTP handlers."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-few-public-methods

    def __init__(self, results=None):
        self.recordings = RecordingStore(
            results if results is not None else Path.cwd() / "results"
        )
        self.explanations = ExplanationStore(self.recordings)


def make_handler(state):
    """Create an HTTP handler bound to the supplied recording services."""

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-statements

    class Handler(BaseHTTPRequestHandler):
        """Serve read-only HTTP responses using the configured local services."""

        def log_message(self, format, *args):
            """Retain the stdlib logging signature while keeping playback quiet."""

        def send(
            self,
            status,
            data,
            content_type="application/json",
            content_encoding=None,
            vary_encoding=False,
        ):
            """Send a response with the requested status, content type and body."""
            body = (
                data
                if isinstance(data, bytes)
                else json.dumps(data, allow_nan=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("X-WTF-Replay-Protocol", "binary-v1")
            self.send_header("Content-Length", str(len(body)))
            if content_encoding:
                self.send_header("Content-Encoding", content_encoding)
            if vary_encoding:
                self.send_header("Vary", "Accept-Encoding, X-WTF-Replay")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """Serve the supported read-only HTTP routes."""

            # These callback/keyword names are fixed by the stdlib or injected
            # constructor interface.
            # pylint: disable=invalid-name
            # Keep this ordered phase and its shared state together; splitting it only
            # to meet a count obscures the contract.
            # pylint: disable=too-many-branches
            # pylint: disable=too-many-locals
            # pylint: disable=too-many-nested-blocks
            # pylint: disable=too-many-return-statements

            path = unquote(urlparse(self.path).path)
            parts = path.strip("/").split("/")
            try:
                if path == "/api/recordings":
                    return self.send(200, state.recordings.catalog())
                if len(parts) >= 3 and parts[:2] == ["api", "recordings"]:
                    if len(parts) == 3:
                        return self.send(200, state.recordings.metadata(parts[2]))
                    if (
                        len(parts) == 6
                        and parts[3] == "runs"
                        and parts[5] == "analysis"
                    ):
                        return self.send(
                            200, state.recordings.analysis(parts[2], int(parts[4]))
                        )
                    if (
                        len(parts) == 7
                        and parts[3] == "runs"
                        and parts[5] == "movement"
                    ):
                        query = parse_qs(
                            urlparse(self.path).query, keep_blank_values=True
                        )
                        if set(query) - {"species", "by_species"} or any(
                            len(v) != 1 for v in query.values()
                        ):
                            raise ValueError("Invalid movement query")
                        if query.get("by_species", ["0"])[0] not in {"0", "1"}:
                            raise ValueError("Invalid movement grouping")
                        return self.send(
                            200,
                            state.recordings.movement(
                                parts[2],
                                int(parts[4]),
                                int(parts[6]),
                                species=query.get("species", [None])[0],
                                by_species=query.get("by_species", ["0"])[0] == "1",
                            ),
                        )
                    if len(parts) == 6 and parts[3] == "runs" and parts[5] == "metrics":
                        return self.send(
                            200, state.recordings.metrics(parts[2], int(parts[4]))
                        )
                    if (
                        len(parts) == 7
                        and parts[3] == "runs"
                        and parts[5] == "telemetry"
                    ):
                        return self.send(
                            200,
                            state.recordings.telemetry(
                                parts[2], int(parts[4]), int(parts[6])
                            ),
                        )
                    if len(parts) == 8 and parts[3] == "runs" and parts[5] == "explain":
                        query = parse_qs(
                            urlparse(self.path).query, keep_blank_values=True
                        )
                        if set(query) - {
                            "criterion",
                            "action",
                            "feature",
                            "value",
                        } or any(len(v) != 1 for v in query.values()):
                            raise ValueError("Invalid explanation query")
                        options = {k: v[0] for k, v in query.items()}
                        if "feature" in options:
                            options["feature"] = int(options["feature"])
                        if "value" in options:
                            options["value"] = float(options["value"])
                        return self.send(
                            200,
                            state.explanations.explain(
                                parts[2],
                                int(parts[4]),
                                int(parts[7]),
                                int(parts[6]),
                                **options,
                            ),
                        )
                    if len(parts) == 7 and parts[3] == "runs" and parts[5] == "chunks":
                        chunk = state.recordings.chunk(
                            parts[2], int(parts[4]), int(parts[6])
                        )
                        accepted = self.headers.get("Accept-Encoding", "")
                        gzip_ok = False
                        for encoding in accepted.lower().split(","):
                            parts = [part.strip() for part in encoding.split(";")]
                            if parts[0] == "gzip":
                                quality = next(
                                    (
                                        part[2:]
                                        for part in parts[1:]
                                        if part.startswith("q=")
                                    ),
                                    "1",
                                )
                                try:
                                    gzip_ok = float(quality) > 0
                                except ValueError:
                                    gzip_ok = False
                        binary = ".bin" in chunk.suffixes
                        requested = self.headers.get("X-WTF-Replay")
                        if (
                            requested == "binary-v1"
                            or requested == "delta-v1"
                            and not binary
                        ):
                            content_type = (
                                "application/vnd.wtf.replay-binary"
                                if binary
                                else "application/json"
                            )
                            if chunk.suffix == ".gz" and gzip_ok:
                                return self.send(
                                    200,
                                    chunk.read_bytes(),
                                    content_type,
                                    content_encoding="gzip",
                                    vary_encoding=True,
                                )
                            return self.send(
                                200,
                                read_chunk_bytes(chunk),
                                content_type,
                                vary_encoding=True,
                            )
                        # Older clients keep the full-frame JSON API. Modern
                        # viewers receive compact stored chunks without expansion.
                        data = json.dumps(
                            load_chunk(chunk), separators=(",", ":"), allow_nan=False
                        ).encode()
                        if gzip_ok:
                            return self.send(
                                200,
                                gzip.compress(data, compresslevel=1),
                                content_encoding="gzip",
                                vary_encoding=True,
                            )
                        return self.send(200, data, vary_encoding=True)
                    return self.send(404, {"error": "Recording endpoint not found"})
                try:
                    target = asset(
                        "viewer/" + ("index.html" if path == "/" else path.lstrip("/"))
                    )
                except ValueError:
                    return self.send(404, {"error": "Not found"})
                if not target.is_file():
                    return self.send(404, {"error": "Not found"})
                return self.send(
                    200,
                    target.read_bytes(),
                    mimetypes.guess_type(str(target))[0] or "application/octet-stream",
                )
            except FileNotFoundError:
                return self.send(404, {"error": "Recording or chunk not found"})
            except (ValueError, OSError, KeyError, EOFError) as error:
                return self.send(400, {"error": str(error)})

        def do_POST(self):
            """Reject request-body methods for the read-only server."""

            # These callback/keyword names are fixed by the stdlib or injected
            # constructor interface.
            # pylint: disable=invalid-name

            return self.send(
                405,
                {
                    "error": "Playback is read-only. Run simulations with python -m wtf.cli."
                },
            )

    return Handler


def argument_parser():
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="WTF — Wildlife Trial Framework: serve saved RL ecosystem recordings"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; use 0.0.0.0 inside a container",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--results", type=Path, default=Path.cwd() / "results")
    return parser


def main():
    """Parse command-line options and run the requested operation."""
    args = argument_parser().parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(State(args.results))
    )
    print(f"WTF playback: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
