from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit, urlunsplit


def canonical_target(target: str) -> str:
    if not target:
        return target
    parsed = urlsplit(target)
    if not parsed.scheme or not parsed.netloc:
        return target
    path = "" if parsed.path == "/" else parsed.path.rstrip("/.,;)")
    if parsed.fragment and not path:
        path = "/"
    fragment = parsed.fragment.rstrip(".,;)")
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, fragment))


def action_fingerprint(action: dict) -> str:
    payload = {
        "action_kind": action.get("action_kind", "tool"),
        "tool": action.get("tool", ""),
        "profile": action.get("profile", ""),
        "target": canonical_target(action.get("target", "")),
        "args": action.get("args", {}),
        "script_template": action.get("script_template"),
        "script_goal": action.get("metadata", {}).get("script_goal"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
