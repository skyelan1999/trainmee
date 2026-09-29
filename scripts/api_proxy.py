#!/usr/bin/env python3
"""Expose a stable OpenAI-compatible model ID over an MLX-LM server."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length"}


def make_handler(upstream: str, model_id: str, upstream_model: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print("[trainmee-api] " + fmt % args, flush=True)

        def do_GET(self):
            self.forward("GET")

        def do_POST(self):
            self.forward("POST")

        def forward(self, method: str):
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else None
            if method == "POST" and self.path.split("?", 1)[0] == "/v1/chat/completions" and body:
                try:
                    payload = json.loads(body)
                    payload["model"] = upstream_model
                    body = json.dumps(payload).encode("utf-8")
                except (json.JSONDecodeError, TypeError):
                    self.send_error(400, "Request body must be valid JSON")
                    return

            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
            if body is not None:
                headers["Content-Length"] = str(len(body))
            request = Request(upstream + self.path, data=body, headers=headers, method=method)
            try:
                response = urlopen(request, timeout=3600)
                self.write_response(response.status, response.headers, response, method)
            except HTTPError as exc:
                self.write_response(exc.code, exc.headers, exc, method)
            except (URLError, TimeoutError, ConnectionError) as exc:
                payload = json.dumps({"error": {"message": f"MLX-LM upstream unavailable: {exc}"}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def write_response(self, status, headers, response, method):
            content_type = headers.get("Content-Type", "application/octet-stream")
            is_stream = "text/event-stream" in content_type.lower()
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() not in HOP_BY_HOP:
                    self.send_header(key, value)
            if is_stream:
                self.send_header("Connection", "close")
                self.close_connection = True
            else:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            if method == "HEAD":
                return
            source = upstream_model.encode()
            target = model_id.encode()
            if is_stream:
                pending = b""
                keep = max(0, len(source) - 1)
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    pending += chunk
                    if len(pending) > keep:
                        boundary = len(pending) - keep
                        while True:
                            start = pending.rfind(source, 0, boundary)
                            if start >= 0 and start + len(source) > boundary:
                                boundary = start
                            else:
                                break
                        safe = pending[:boundary].replace(source, target)
                        pending = pending[boundary:]
                        self.wfile.write(safe)
                        self.wfile.flush()
                if pending:
                    self.wfile.write(pending.replace(source, target))
                    self.wfile.flush()
            else:
                payload = response.read()
                if self.path.split("?", 1)[0] == "/v1/models":
                    try:
                        parsed = json.loads(payload)
                        for item in parsed.get("data", []):
                            item["id"] = model_id
                        payload = json.dumps(parsed).encode("utf-8")
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass
                elif self.path.split("?", 1)[0] == "/v1/chat/completions":
                    payload = payload.replace(source, target)
                self.wfile.write(payload)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument("--model-id", default="trainmee")
    parser.add_argument("--upstream-model", required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(
        args.upstream, args.model_id, args.upstream_model))
    print(f"TrainMee API listening at http://{args.host}:{args.port}/v1 (model ID: {args.model_id})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
