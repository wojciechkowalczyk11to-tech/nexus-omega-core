#!/usr/bin/env python3
"""Wait until backend health endpoint reports healthy dependencies."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

HEALTHY_KEYS = ("status", "database", "redis")
REQUEST_TIMEOUT_SECONDS = 5


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _is_healthy(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return all(payload.get(key) == "healthy" for key in HEALTHY_KEYS)


def wait_for_health(url: str, timeout: int, interval: float) -> int:
    deadline = time.time() + timeout
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                _stderr(f"[wait_for_backend_health] attempt {attempt}: invalid JSON: {exc}")
            else:
                if _is_healthy(payload):
                    _stderr(
                        f"[wait_for_backend_health] backend is healthy after {attempt} attempts."
                    )
                    print(json.dumps(payload))
                    return 0
                _stderr(
                    "[wait_for_backend_health] attempt "
                    f"{attempt}: unhealthy payload {json.dumps(payload, sort_keys=True)}"
                )
        except urllib.error.URLError as exc:
            _stderr(f"[wait_for_backend_health] attempt {attempt}: request failed: {exc}")

        time.sleep(interval)

    _stderr(f"[wait_for_backend_health] timeout after {timeout}s waiting for {url}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1/health")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    return wait_for_health(args.url, args.timeout, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
