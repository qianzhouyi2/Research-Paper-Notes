from __future__ import annotations

import html
import json
import posixpath
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import markdown  # type: ignore  # noqa: E402
from prepare_github_wiki import main as prepare_github_wiki  # noqa: E402

EXPORT = ROOT / "github_wiki"
DOCS = ROOT / "docs"
REPO_BLOB_BASE = "https://github.com/qianzhouyi2/Research-Paper-Notes/blob/main/"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
FIRST_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
LINK_ATTR_RE = re.compile(r'(?P<attr>href|src)="(?P<url>[^"]+)"')

SITE_CSS = """\
:root {
  --paper: #f4efe5;
  --paper-strong: #efe6d7;
  --paper-deep: #e5d7bf;
  --ink: #1f2b28;
  --ink-soft: #596662;
  --accent: #0e6b63;
  --accent-soft: rgba(14, 107, 99, 0.12);
  --signal: #8b3f1f;
  --border: rgba(31, 43, 40, 0.14);
  --shadow: 0 18px 46px rgba(48, 41, 24, 0.11);
  --radius: 22px;
  --mono: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  --ui: "Trebuchet MS", "Gill Sans", "Segoe UI", sans-serif;
  --serif: "Baskerville", "Palatino Linotype", "Book Antiqua", Georgia, serif;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  color: var(--ink);
  font-family: var(--ui);
  background:
    radial-gradient(circle at top left, rgba(14, 107, 99, 0.12), transparent 28%),
    radial-gradient(circle at top right, rgba(139, 63, 31, 0.10), transparent 24%),
    linear-gradient(180deg, #f8f4ec 0%, #f1e8d9 100%);
}

a {
  color: var(--accent);
}

.site-shell {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr) 280px;
  gap: 24px;
  min-height: 100vh;
  padding: 24px;
}

.panel {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(14px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.sidebar {
  position: sticky;
  top: 24px;
  height: calc(100vh - 48px);
  padding: 24px 22px;
  overflow: auto;
  background:
    linear-gradient(180deg, rgba(239, 230, 215, 0.92) 0%, rgba(255, 255, 255, 0.76) 100%);
}

.brand {
  display: block;
  margin-bottom: 10px;
  color: var(--ink);
  text-decoration: none;
}

.brand-kicker {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.brand h1 {
  margin: 14px 0 8px;
  font-family: var(--serif);
  font-size: 31px;
  line-height: 1.05;
}

.brand p {
  margin: 0;
  color: var(--ink-soft);
  line-height: 1.55;
}

.sidebar nav {
  margin-top: 24px;
  font-size: 15px;
}

.sidebar nav ul {
  padding-left: 18px;
}

.sidebar nav li {
  margin: 9px 0;
}

.sidebar nav a {
  text-decoration: none;
}

.main {
  min-width: 0;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 22px;
  margin-bottom: 18px;
}

.topbar h2 {
  margin: 0;
  font-family: var(--serif);
  font-size: 24px;
}

.topbar p {
  margin: 4px 0 0;
  color: var(--ink-soft);
}

.topbar-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.search-box {
  position: relative;
  min-width: min(380px, 48vw);
}

.search-box input {
  width: 100%;
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.9);
  color: var(--ink);
  font: inherit;
}

.search-box input:focus {
  outline: 2px solid rgba(14, 107, 99, 0.18);
  border-color: rgba(14, 107, 99, 0.35);
}

.search-results {
  position: absolute;
  top: calc(100% + 10px);
  left: 0;
  right: 0;
  display: none;
  max-height: 420px;
  overflow: auto;
  padding: 10px;
  border-radius: 18px;
}

.search-results.visible {
  display: block;
}

.search-results a {
  display: block;
  padding: 12px 13px;
  border-radius: 14px;
  text-decoration: none;
  color: inherit;
}

.search-results a:hover {
  background: var(--accent-soft);
}

.search-results strong {
  display: block;
  margin-bottom: 4px;
}

.search-results span {
  display: block;
  color: var(--ink-soft);
  font-size: 13px;
  line-height: 1.45;
}

.breadcrumbs {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 16px;
  padding: 0 8px;
  color: var(--ink-soft);
  font-size: 14px;
}

.breadcrumbs a {
  text-decoration: none;
}

.article-wrap {
  padding: 32px clamp(22px, 3vw, 40px);
}

.article-path {
  display: inline-block;
  margin-bottom: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(139, 63, 31, 0.09);
  color: var(--signal);
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.markdown-body {
  font-family: var(--serif);
  font-size: 18px;
  line-height: 1.8;
}

.markdown-body > *:first-child {
  margin-top: 0;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4 {
  margin-top: 1.7em;
  margin-bottom: 0.65em;
  line-height: 1.2;
  color: #14201d;
}

.markdown-body h1 {
  font-size: clamp(34px, 4vw, 48px);
}

.markdown-body h2 {
  font-size: 29px;
  border-top: 1px solid rgba(20, 32, 29, 0.1);
  padding-top: 0.85em;
}

.markdown-body h3 {
  font-size: 23px;
}

.markdown-body p,
.markdown-body ul,
.markdown-body ol,
.markdown-body blockquote,
.markdown-body table,
.markdown-body pre {
  margin: 1em 0;
}

.markdown-body ul,
.markdown-body ol {
  padding-left: 1.4em;
}

.markdown-body li + li {
  margin-top: 0.35em;
}

.markdown-body code {
  font-family: var(--mono);
  font-size: 0.88em;
  background: rgba(31, 43, 40, 0.07);
  padding: 0.16em 0.38em;
  border-radius: 7px;
}

.markdown-body pre {
  overflow: auto;
  padding: 18px;
  border-radius: 16px;
  background: #172320;
  color: #f0ece3;
}

.markdown-body pre code {
  background: transparent;
  padding: 0;
  color: inherit;
}

.markdown-body blockquote {
  margin-left: 0;
  padding: 8px 18px;
  border-left: 4px solid rgba(14, 107, 99, 0.36);
  background: rgba(14, 107, 99, 0.05);
  color: #34423f;
}

.markdown-body hr {
  border: 0;
  border-top: 1px solid rgba(20, 32, 29, 0.12);
}

.markdown-body img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 18px auto;
  border-radius: 16px;
  box-shadow: 0 14px 32px rgba(31, 43, 40, 0.12);
}

.markdown-body table {
  width: 100%;
  border-collapse: collapse;
  font-size: 16px;
}

.markdown-body th,
.markdown-body td {
  padding: 12px 14px;
  border: 1px solid rgba(31, 43, 40, 0.12);
  vertical-align: top;
}

.markdown-body th {
  background: rgba(14, 107, 99, 0.08);
  text-align: left;
}

.rail {
  position: sticky;
  top: 24px;
  height: fit-content;
  padding: 24px 22px;
}

.rail h3 {
  margin: 0 0 14px;
  font-family: var(--serif);
  font-size: 22px;
}

.meta-block + .meta-block {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid var(--border);
}

.meta-label {
  margin: 0 0 8px;
  color: var(--ink-soft);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.meta-block p,
.meta-block ul {
  margin: 0;
  color: var(--ink);
  line-height: 1.6;
}

.meta-block ul {
  padding-left: 18px;
}

.chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.chip {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 13px;
  line-height: 1;
}

.mobile-toggle {
  display: none;
  border: 0;
  border-radius: 14px;
  padding: 11px 14px;
  background: var(--ink);
  color: #fff;
  font: inherit;
  cursor: pointer;
}

.footer-note {
  margin-top: 20px;
  color: var(--ink-soft);
  font-size: 13px;
}

@media (max-width: 1180px) {
  .site-shell {
    grid-template-columns: 290px minmax(0, 1fr);
  }

  .rail {
    grid-column: 1 / -1;
    position: static;
  }
}

@media (max-width: 900px) {
  .site-shell {
    grid-template-columns: 1fr;
    padding: 14px;
  }

  .sidebar {
    display: none;
    position: static;
    height: auto;
  }

  body.menu-open .sidebar {
    display: block;
  }

  .mobile-toggle {
    display: inline-flex;
  }

  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .topbar-actions,
  .search-box {
    width: 100%;
  }

  .search-box {
    min-width: 0;
  }
}
"""

SITE_JS = """\
(() => {
  const input = document.querySelector('[data-search-input]');
  const results = document.querySelector('[data-search-results]');
  const toggle = document.querySelector('[data-menu-toggle]');
  const body = document.body;
  const rootPrefix = body.dataset.rootPrefix || '';
  const currentPath = body.dataset.pagePath || '';
  let entries = [];

  function withRoot(path) {
    return `${rootPrefix}${path}`;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderResults(query) {
    if (!results) return;
    const q = query.trim().toLowerCase();
    if (!q) {
      results.classList.remove('visible');
      results.innerHTML = '';
      return;
    }

    const filtered = entries
      .filter((entry) =>
        entry.title.toLowerCase().includes(q) ||
        entry.path.toLowerCase().includes(q) ||
        entry.summary.toLowerCase().includes(q)
      )
      .slice(0, 12);

    if (!filtered.length) {
      results.classList.add('visible');
      results.innerHTML = '<div class="panel" style="padding:14px 16px;">No matching pages.</div>';
      return;
    }

    const cards = filtered
      .map((entry) => {
        const summary = entry.summary ? escapeHtml(entry.summary) : 'Open page';
        const section = escapeHtml(entry.section || 'page');
        return `
          <a class="panel" href="${encodeURI(withRoot(entry.url))}">
            <strong>${escapeHtml(entry.title)}</strong>
            <span>${section}</span>
            <span>${summary}</span>
          </a>`;
      })
      .join('');

    results.classList.add('visible');
    results.innerHTML = cards;
  }

  if (input && results) {
    fetch(withRoot('search-index.json'))
      .then((res) => res.json())
      .then((data) => {
        entries = Array.isArray(data) ? data.filter((entry) => entry.url !== currentPath) : [];
      })
      .catch(() => {
        entries = [];
      });

    input.addEventListener('input', (event) => renderResults(event.target.value));
    document.addEventListener('click', (event) => {
      if (!results.contains(event.target) && event.target !== input) {
        results.classList.remove('visible');
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        results.classList.remove('visible');
        input.blur();
      }
    });
  }

  if (toggle) {
    toggle.addEventListener('click', () => {
      body.classList.toggle('menu-open');
    });
  }
})();
"""


def normalize_path(value: str) -> str:
    value = value.replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    normalized = posixpath.normpath(value)
    return "" if normalized == "." else normalized.lstrip("/")


def parse_frontmatter(text: str) -> Tuple[Dict, str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    body = text[match.end():]
    return meta if isinstance(meta, dict) else {}, body


def plain_text_excerpt(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"`([^`]+)`", r"\1", text)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[*_>#-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[: limit - 1] + "…" if len(cleaned) > limit else cleaned


def first_heading(text: str) -> str:
    match = FIRST_HEADING_RE.search(text)
    return match.group(1).strip() if match else ""


def site_title(md_rel: str, meta: Dict, body: str) -> str:
    return str(meta.get("title") or first_heading(body) or Path(md_rel).stem)


def page_output_rel(md_rel: str) -> str:
    return normalize_path(Path(md_rel).with_suffix(".html").as_posix())


def root_prefix_for(page_rel: str) -> str:
    page_dir = posixpath.dirname(page_rel)
    if not page_dir:
        return ""
    rel = posixpath.relpath(".", page_dir)
    return "" if rel == "." else rel + "/"


def relative_url(current_html_rel: str, target_rel: str) -> str:
    current_dir = posixpath.dirname(current_html_rel) or "."
    rel = posixpath.relpath(target_rel, current_dir)
    return rel if rel != "." else posixpath.basename(target_rel)


def is_external(url: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url)) or url.startswith("//")


def rewrite_rendered_html(
    html_text: str,
    source_rel: str,
    current_html_rel: str,
    md_rel_set: set[str],
    export_files: set[str],
) -> str:
    source_dir = posixpath.dirname(source_rel)

    def repl(match: re.Match[str]) -> str:
        attr = match.group("attr")
        url = match.group("url")
        if not url or url.startswith("#") or is_external(url):
            return match.group(0)

        path_part, fragment = (url.split("#", 1) + [""])[:2]
        path_part = path_part.strip()
        if not path_part:
            return match.group(0)

        resolved = normalize_path(posixpath.join(source_dir, path_part))
        if resolved == "Home.md":
            target_rel = "index.html"
        elif resolved in md_rel_set:
            target_rel = page_output_rel(resolved)
        elif resolved in export_files:
            target_rel = resolved
        else:
            return match.group(0)

        new_url = relative_url(current_html_rel, target_rel)
        if fragment:
            new_url = f"{new_url}#{fragment}"
        return f'{attr}="{html.escape(new_url, quote=True)}"'

    return LINK_ATTR_RE.sub(repl, html_text)


def render_markdown(
    source_rel: str,
    current_html_rel: str,
    markdown_text: str,
    md_rel_set: set[str],
    export_files: set[str],
) -> str:
    engine = markdown.Markdown(
        extensions=[
            "extra",
            "fenced_code",
            "tables",
            "toc",
            "sane_lists",
            "smarty",
        ]
    )
    rendered = engine.convert(markdown_text)
    return rewrite_rendered_html(rendered, source_rel, current_html_rel, md_rel_set, export_files)


def breadcrumbs(page_rel: str) -> List[Tuple[str, str]]:
    crumbs = [("Home", "index.html")]
    if page_rel == "index.html":
        return crumbs
    parts = page_rel.split("/")
    accum: List[str] = []
    for part in parts[:-1]:
        accum.append(part)
        index_rel = "/".join(accum + ["index.html"])
        label = part.replace("-", " ").replace("_", " ").title()
        crumbs.append((label, index_rel))
    label = Path(parts[-1]).stem.replace("-", " ").replace("_", " ")
    crumbs.append((label, page_rel))
    return crumbs


def build_breadcrumb_html(current_html_rel: str) -> str:
    items = []
    crumbs = breadcrumbs(current_html_rel)
    for idx, (label, target) in enumerate(crumbs):
        text = html.escape(label)
        if idx == len(crumbs) - 1:
            items.append(f"<span>{text}</span>")
        else:
            href = html.escape(relative_url(current_html_rel, target), quote=True)
            items.append(f'<a href="{href}">{text}</a>')
    return "<span> / </span>".join(items)


def render_template(
    *,
    title: str,
    current_html_rel: str,
    page_source_rel: str,
    body_html: str,
    sidebar_html: str,
    summary: str,
    tags: List[str],
) -> str:
    prefix = root_prefix_for(current_html_rel)
    breadcrumb_html = build_breadcrumb_html(current_html_rel)
    github_url = REPO_BLOB_BASE + html.escape(page_source_rel, quote=True)
    chips = "".join(f'<span class="chip">{html.escape(tag)}</span>' for tag in tags)
    tag_block = chips if chips else '<span class="chip">untagged</span>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} | Research Paper Notes</title>
  <meta name="description" content="{html.escape(summary, quote=True)}">
  <link rel="stylesheet" href="{prefix}styles.css">
  <script defer src="{prefix}site.js"></script>
</head>
<body data-root-prefix="{prefix}" data-page-path="{html.escape(current_html_rel, quote=True)}">
  <div class="site-shell">
    <aside class="sidebar panel">
      <a class="brand" href="{prefix}index.html">
        <span class="brand-kicker">Research Wiki</span>
        <h1>Research Paper Notes</h1>
        <p>A browsable paper-reading vault published as a GitHub Pages site.</p>
      </a>
      <nav>{sidebar_html}</nav>
      <p class="footer-note">Static export from the local Obsidian-style wiki. Markdown was pre-rendered for GitHub Pages.</p>
    </aside>

    <main class="main">
      <header class="topbar panel">
        <div>
          <button class="mobile-toggle" data-menu-toggle type="button">Menu</button>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(summary or "Research note page")}</p>
        </div>
        <div class="topbar-actions">
          <div class="search-box">
            <input data-search-input type="search" placeholder="Search pages, concepts, entities...">
            <div data-search-results class="search-results"></div>
          </div>
        </div>
      </header>

      <div class="breadcrumbs">{breadcrumb_html}</div>

      <article class="article-wrap panel">
        <div class="article-path">{html.escape(page_source_rel)}</div>
        <div class="markdown-body">{body_html}</div>
      </article>
    </main>

    <aside class="rail panel">
      <h3>Page Meta</h3>
      <div class="meta-block">
        <p class="meta-label">Section</p>
        <p>{html.escape(page_source_rel.split("/", 1)[0] if "/" in page_source_rel else "root")}</p>
      </div>
      <div class="meta-block">
        <p class="meta-label">Tags</p>
        <div class="chip-list">{tag_block}</div>
      </div>
      <div class="meta-block">
        <p class="meta-label">Summary</p>
        <p>{html.escape(summary or "No summary available.")}</p>
      </div>
      <div class="meta-block">
        <p class="meta-label">Source</p>
        <ul>
          <li><a href="{github_url}">Open Markdown On GitHub</a></li>
          <li><a href="{prefix}search-index.json">Open Search Index</a></li>
        </ul>
      </div>
    </aside>
  </div>
</body>
</html>
"""


def collect_files(root: Path) -> Tuple[List[Path], set[str], set[str]]:
    md_files: List[Path] = []
    md_rel_set: set[str] = set()
    export_files: set[str] = set()
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = normalize_path(file_path.relative_to(root).as_posix())
        export_files.add(rel)
        if file_path.suffix.lower() == ".md":
            md_files.append(file_path)
            md_rel_set.add(rel)
    return sorted(md_files), md_rel_set, export_files


def copy_non_markdown_assets(export_files: set[str]) -> None:
    for rel in sorted(export_files):
        if rel.endswith(".md"):
            continue
        src = EXPORT / Path(rel)
        dst = DOCS / Path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_pages() -> List[Dict[str, str]]:
    prepare_github_wiki()
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir(parents=True, exist_ok=True)

    md_files, md_rel_set, export_files = collect_files(EXPORT)
    copy_non_markdown_assets(export_files)

    sidebar_source = (EXPORT / "_Sidebar.md").read_text(encoding="utf-8")
    entries: List[Dict[str, str]] = []

    for md_path in md_files:
        md_rel = normalize_path(md_path.relative_to(EXPORT).as_posix())
        if md_rel in {"_Sidebar.md", "Home.md"}:
            continue

        meta, body = parse_frontmatter(md_path.read_text(encoding="utf-8"))
        page_rel = page_output_rel(md_rel)
        title = site_title(md_rel, meta, body)
        summary = str(meta.get("summary") or plain_text_excerpt(body))
        tags = [str(tag) for tag in meta.get("tags", [])] if isinstance(meta.get("tags"), list) else []

        body_html = render_markdown(md_rel, page_rel, body, md_rel_set, export_files)
        sidebar_html = render_markdown("_Sidebar.md", page_rel, sidebar_source, md_rel_set, export_files)
        output_html = render_template(
            title=title,
            current_html_rel=page_rel,
            page_source_rel=md_rel,
            body_html=body_html,
            sidebar_html=sidebar_html,
            summary=summary,
            tags=tags,
        )

        out_path = DOCS / Path(page_rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_html, encoding="utf-8")

        entries.append(
            {
                "title": title,
                "url": page_rel,
                "path": md_rel,
                "section": md_rel.split("/", 1)[0] if "/" in md_rel else "root",
                "summary": summary,
            }
        )

    existing_urls = {entry["url"] for entry in entries}
    dir_set: set[str] = set()
    for entry in entries:
        dir_rel = posixpath.dirname(entry["url"])
        while dir_rel:
            dir_set.add(dir_rel)
            parent = posixpath.dirname(dir_rel)
            if parent == dir_rel:
                break
            dir_rel = parent
    ordered_dirs = sorted(dir_set)

    for dir_rel in ordered_dirs:
        index_rel = f"{dir_rel}/index.html"
        if index_rel in existing_urls:
            continue

        child_dirs = sorted(child for child in ordered_dirs if posixpath.dirname(child) == dir_rel)
        child_pages = sorted(
            (entry for entry in entries if posixpath.dirname(entry["url"]) == dir_rel and entry["url"] != index_rel),
            key=lambda item: item["title"].lower(),
        )

        title = f"{Path(dir_rel).name} Index"
        body_lines = [f"# {title}", "", f"Directory overview for `{dir_rel}`.", ""]
        if child_dirs:
            body_lines.extend(["## Subdirectories", ""])
            for child in child_dirs:
                child_index = f"{child}/index.html"
                body_lines.append(f"- [{Path(child).name}]({relative_url(index_rel, child_index)})")
            body_lines.append("")
        if child_pages:
            body_lines.extend(["## Pages", ""])
            for page in child_pages:
                body_lines.append(f"- [{page['title']}]({relative_url(index_rel, page['url'])})")
            body_lines.append("")

        body_html = render_markdown(index_rel, index_rel, "\n".join(body_lines), md_rel_set, export_files)
        sidebar_html = render_markdown("_Sidebar.md", index_rel, sidebar_source, md_rel_set, export_files)
        output_html = render_template(
            title=title,
            current_html_rel=index_rel,
            page_source_rel=dir_rel,
            body_html=body_html,
            sidebar_html=sidebar_html,
            summary=f"Directory index for {dir_rel}.",
            tags=["index", "directory"],
        )

        out_path = DOCS / Path(index_rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_html, encoding="utf-8")

        entries.append(
            {
                "title": title,
                "url": index_rel,
                "path": dir_rel,
                "section": dir_rel.split("/", 1)[0] if "/" in dir_rel else dir_rel,
                "summary": f"Directory index for {dir_rel}.",
            }
        )
        existing_urls.add(index_rel)

    not_found_html = render_template(
        title="Page Not Found",
        current_html_rel="404.html",
        page_source_rel="404",
        body_html="<h1>Page Not Found</h1><p>The requested page is not part of this export. Use search or return to the home page.</p><p><a href=\"index.html\">Back to home</a></p>",
        sidebar_html=render_markdown("_Sidebar.md", "404.html", sidebar_source, md_rel_set, export_files),
        summary="Fallback page for GitHub Pages.",
        tags=["pages", "404"],
    )
    (DOCS / "404.html").write_text(not_found_html, encoding="utf-8")
    (DOCS / "styles.css").write_text(SITE_CSS, encoding="utf-8")
    (DOCS / "site.js").write_text(SITE_JS, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    (DOCS / "search-index.json").write_text(
        json.dumps(sorted(entries, key=lambda item: item["title"].lower()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return entries


def main() -> None:
    entries = build_pages()
    html_count = sum(1 for _ in DOCS.rglob("*.html"))
    asset_count = sum(1 for p in DOCS.rglob("*") if p.is_file() and p.suffix.lower() not in {".html", ".json", ".css", ".js", ".md"})
    print(f"Pages generated: {len(entries)}")
    print(f"HTML files: {html_count}")
    print(f"Copied assets: {asset_count}")
    print(f"Output: {DOCS}")


if __name__ == "__main__":
    main()
