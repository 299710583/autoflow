from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


class WebStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.forms: list[dict] = []
        self.inputs: list[dict] = []
        self._current_form: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "a" and values.get("href"):
            self.links.append(values["href"])
        elif tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        elif tag == "form":
            self._current_form = {
                "action": values.get("action", ""),
                "method": values.get("method", "get").lower(),
                "inputs": [],
            }
            self.forms.append(self._current_form)
        elif tag in {"input", "textarea", "select"}:
            item = {
                "name": values.get("name", ""),
                "type": values.get("type", tag),
                "id": values.get("id", ""),
            }
            if self._current_form is not None:
                self._current_form["inputs"].append(item)
            else:
                self.inputs.append(item)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "form":
            self._current_form = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()


@dataclass(frozen=True)
class WebFetchResult:
    url: str
    status_code: int
    headers: dict[str, str]
    body: str
    error: str = ""


class WebReconClient:
    def __init__(self, timeout: int = 10, max_body_bytes: int = 1_000_000) -> None:
        self.timeout = timeout
        self.max_body_bytes = max_body_bytes

    def recon(self, target_url: str) -> dict:
        normalized_url = self._normalize_url(target_url)
        homepage = self._fetch(normalized_url)
        result = {
            "target": normalized_url,
            "status_code": homepage.status_code,
            "headers": homepage.headers,
            "title": "",
            "links": [],
            "forms": [],
            "inputs": [],
            "scripts": [],
            "interesting_paths": [],
            "robots": None,
            "sitemap": None,
            "error": homepage.error,
        }
        if homepage.body:
            parser = WebStructureParser()
            parser.feed(homepage.body)
            result.update(
                {
                    "title": parser.title,
                    "links": self._absolute_unique(normalized_url, parser.links),
                    "forms": self._normalize_forms(normalized_url, parser.forms),
                    "inputs": parser.inputs,
                    "scripts": self._absolute_unique(normalized_url, parser.scripts),
                    "interesting_paths": self._interesting_paths(homepage.body, normalized_url),
                }
            )

        robots = self._fetch(urljoin(normalized_url, "/robots.txt"))
        if robots.status_code and not robots.error:
            result["robots"] = {
                "url": robots.url,
                "status_code": robots.status_code,
                "interesting_paths": self._robots_paths(robots.body, normalized_url),
            }

        sitemap = self._fetch(urljoin(normalized_url, "/sitemap.xml"))
        if sitemap.status_code and not sitemap.error:
            result["sitemap"] = {
                "url": sitemap.url,
                "status_code": sitemap.status_code,
                "locations": self._sitemap_locations(sitemap.body),
            }

        return result

    def _fetch(self, url: str) -> WebFetchResult:
        request = Request(url, headers={"User-Agent": "AutoFlow-WebRecon/0.1"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read(self.max_body_bytes)
                return WebFetchResult(
                    url=url,
                    status_code=response.status,
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=raw_body.decode("utf-8", errors="replace"),
                )
        except Exception as exc:
            return WebFetchResult(url=url, status_code=0, headers={}, body="", error=str(exc))

    def _normalize_url(self, target_url: str) -> str:
        if "://" not in target_url:
            target_url = f"http://{target_url}"
        parsed = urlparse(target_url)
        if not parsed.path:
            return f"{target_url.rstrip('/')}/"
        return target_url

    def _absolute_unique(self, base_url: str, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            absolute = urljoin(base_url, value)
            if absolute not in seen:
                seen.add(absolute)
                result.append(absolute)
        return result[:100]

    def _normalize_forms(self, base_url: str, forms: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for form in forms:
            normalized.append(
                {
                    **form,
                    "action": urljoin(base_url, form.get("action") or base_url),
                }
            )
        return normalized[:50]

    def _interesting_paths(self, body: str, base_url: str) -> list[str]:
        patterns = sorted(set(re.findall(r'["\']((?:/|api/|rest/)[A-Za-z0-9_./?=&%-]{2,})["\']', body)))
        return self._absolute_unique(base_url, patterns)[:100]

    def _robots_paths(self, body: str, base_url: str) -> list[str]:
        paths = []
        for line in body.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() in {"allow", "disallow", "sitemap"}:
                value = value.strip()
                if value:
                    paths.append(urljoin(base_url, value))
        return paths[:100]

    def _sitemap_locations(self, body: str) -> list[str]:
        return re.findall(r"<loc>(.*?)</loc>", body, flags=re.IGNORECASE)[:100]
