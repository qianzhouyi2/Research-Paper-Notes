from __future__ import annotations

import os
import posixpath
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "github_wiki"

CONTENT_DIRS = [
    "concepts",
    "entities",
    "journal",
    "notes",
    "papers_sources/Research-Paper-Notes",
    "projects",
    "references",
    "synthesis",
    "Templates",
]

ROOT_PAGES = ["index.md", "log.md"]
SKIP_DIR_NAMES = {".git", ".obsidian", "_archives", "_imports", "skills", "_notion_raw"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}

WIKILINK_RE = re.compile(r"(!?)\[\[([^\]]+)\]\]")
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)")
MD_LINK_FULL_RE = re.compile(r"(!?\[[^\]]*\]\()(<[^>]+>|[^)]+)(\))")


def norm_posix(path: str) -> str:
    path = path.replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    normalized = posixpath.normpath(path)
    if normalized in {".", ""}:
        return ""
    return normalized.lstrip("/")


def clean_anchor(anchor: str) -> str:
    text = anchor.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[`~!@#$%^&*()+=\[\]{}|\\:;\"'<>,.?/]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def parse_inner(inner: str) -> Tuple[str, Optional[str], Optional[str]]:
    parts = [p.strip() for p in inner.split("|")]
    raw_target = parts[0] if parts else ""
    alias = None
    for p in parts[1:]:
        if not p:
            continue
        if p.isdigit():
            continue
        alias = p
        break
    target, anchor = raw_target, None
    if "#" in raw_target:
        target, anchor = raw_target.split("#", 1)
    return target.strip(), alias, anchor


def list_markdown_files() -> List[Path]:
    files: List[Path] = []
    for page in ROOT_PAGES:
        p = ROOT / page
        if p.exists():
            files.append(p)
    for rel_dir in CONTENT_DIRS:
        base = ROOT / rel_dir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            rel = md.relative_to(ROOT)
            if any(part in SKIP_DIR_NAMES for part in rel.parts):
                continue
            files.append(md)
    deduped = sorted(set(files), key=lambda p: p.as_posix().lower())
    return deduped


def build_indexes(md_files: List[Path]) -> Tuple[Set[str], Dict[str, List[str]], Dict[str, str]]:
    md_rel_set: Set[str] = set()
    stem_map: Dict[str, List[str]] = {}
    no_ext_map: Dict[str, str] = {}
    for p in md_files:
        rel = p.relative_to(ROOT).as_posix()
        md_rel_set.add(rel)
        stem_map.setdefault(p.stem.lower(), []).append(rel)
        no_ext_map[p.with_suffix("").relative_to(ROOT).as_posix().lower()] = rel
    return md_rel_set, stem_map, no_ext_map


def resolve_markdown_target(
    current_rel: str,
    target: str,
    md_rel_set: Set[str],
    stem_map: Dict[str, List[str]],
    no_ext_map: Dict[str, str],
) -> Optional[str]:
    target = target.strip()
    if not target:
        return None

    if target.lower().startswith(("http://", "https://", "mailto:")):
        return None

    current_dir = posixpath.dirname(current_rel)
    raw = norm_posix(target)

    candidate_keys: List[str] = []
    if raw:
        candidate_keys.extend(
            [
                norm_posix(posixpath.join(current_dir, raw)),
                raw,
            ]
        )
        if not posixpath.splitext(raw)[1]:
            candidate_keys.extend(
                [
                    norm_posix(posixpath.join(current_dir, raw + ".md")),
                    norm_posix(raw + ".md"),
                    norm_posix(posixpath.join(current_dir, raw, "index.md")),
                    norm_posix(raw + "/index.md"),
                ]
            )

    for key in candidate_keys:
        if key in md_rel_set:
            return key
        key_no_ext = key[:-3] if key.lower().endswith(".md") else key
        if key_no_ext.lower() in no_ext_map:
            return no_ext_map[key_no_ext.lower()]

    if "/" not in raw and "." not in raw:
        options = stem_map.get(raw.lower(), [])
        if len(options) == 1:
            return options[0]
        if len(options) > 1:
            same_bucket = [x for x in options if x.split("/", 1)[0] == current_rel.split("/", 1)[0]]
            if same_bucket:
                return sorted(same_bucket)[0]
            return sorted(options)[0]

    return None


def resolve_asset_target(current_rel: str, target: str) -> Optional[str]:
    target = target.strip()
    if not target:
        return None
    if target.lower().startswith(("http://", "https://", "mailto:")):
        return None

    current_dir = posixpath.dirname(current_rel)
    raw = norm_posix(target)
    candidates = [norm_posix(posixpath.join(current_dir, raw)), raw]
    if not posixpath.splitext(raw)[1]:
        candidates.extend(
            [
                norm_posix(posixpath.join(current_dir, raw + ".md")),
                norm_posix(raw + ".md"),
                norm_posix(posixpath.join(current_dir, raw, "index.md")),
                norm_posix(raw + "/index.md"),
            ]
        )
    for cand in candidates:
        if not cand:
            continue
        src = ROOT / Path(cand)
        if src.exists() and src.is_file():
            return cand
    return None


def rel_link(from_rel: str, to_rel: str, anchor: Optional[str]) -> str:
    from_dir = posixpath.dirname(from_rel)
    relative = posixpath.relpath(to_rel, from_dir or ".")
    if relative == ".":
        relative = posixpath.basename(to_rel)
    if anchor:
        slug = clean_anchor(anchor)
        if slug:
            relative = f"{relative}#{slug}"
    return f"<{relative}>"


def convert_text(
    current_rel: str,
    text: str,
    md_rel_set: Set[str],
    stem_map: Dict[str, List[str]],
    no_ext_map: Dict[str, str],
) -> str:
    def repl(match: re.Match[str]) -> str:
        is_embed = bool(match.group(1))
        inner = match.group(2)
        target, alias, anchor = parse_inner(inner)
        label = alias or Path(target).stem or target or "link"

        resolved_md = resolve_markdown_target(current_rel, target, md_rel_set, stem_map, no_ext_map)
        if resolved_md:
            link = rel_link(current_rel, resolved_md, anchor)
            return f"[{label}]({link})"

        resolved_asset = resolve_asset_target(current_rel, target)
        if resolved_asset:
            link = rel_link(current_rel, resolved_asset, anchor)
            if is_embed or Path(resolved_asset).suffix.lower() in IMAGE_EXTS:
                return f"![{label}]({link})"
            return f"[{label}]({link})"

        if not is_embed:
            plain_like = "/" not in target and "." not in target
            if plain_like and (re.fullmatch(r"\d{4}-\d{2}-\d{2}", target) or "..." in target):
                return label

        fallback = target
        if anchor:
            fallback = f"{fallback}#{clean_anchor(anchor)}"
        fallback_link = f"<{fallback}>"
        if is_embed:
            return f"![{label}]({fallback_link})"
        return f"[{label}]({fallback_link})"

    return WIKILINK_RE.sub(repl, text)


def rewrite_existing_markdown_links(current_rel: str, text: str) -> str:
    current_dir = posixpath.dirname(current_rel)

    def repl(match: re.Match[str]) -> str:
        prefix, raw_dest, suffix = match.groups()
        dest = raw_dest.strip()
        was_wrapped = False
        if dest.startswith("<") and dest.endswith(">"):
            dest = dest[1:-1]
            was_wrapped = True

        if not dest or dest.startswith("#"):
            return match.group(0)
        if dest.lower().startswith(("http://", "https://", "mailto:")):
            return match.group(0)

        path_part, frag = (dest.split("#", 1) + [""])[:2]
        path_part = path_part.strip()
        if not path_part:
            return match.group(0)

        rel_current = norm_posix(posixpath.join(current_dir, path_part))
        root_path = norm_posix(path_part)
        candidates = [rel_current, root_path]
        if not posixpath.splitext(path_part)[1]:
            candidates.extend(
                [
                    norm_posix(rel_current + ".md"),
                    norm_posix(root_path + ".md"),
                    norm_posix(rel_current + "/index.md"),
                    norm_posix(root_path + "/index.md"),
                ]
            )

        chosen = None
        for cand in candidates:
            if not cand:
                continue
            src = ROOT / Path(cand)
            if src.exists():
                chosen = cand
                break

        if not chosen:
            return match.group(0)

        new_rel = posixpath.relpath(chosen, current_dir or ".")
        new_dest = new_rel + (f"#{frag}" if frag else "")
        if was_wrapped or " " in new_dest:
            new_dest = f"<{new_dest}>"
        return f"{prefix}{new_dest}{suffix}"

    return MD_LINK_FULL_RE.sub(repl, text)


def copy_referenced_assets(md_output_files: List[Path]) -> int:
    copied = 0
    seen: Set[str] = set()

    for out_md in md_output_files:
        content = out_md.read_text(encoding="utf-8")
        for m in MD_LINK_RE.finditer(content):
            raw = m.group(1).strip()
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            if not raw or raw.startswith("#"):
                continue
            if raw.lower().startswith(("http://", "https://", "mailto:")):
                continue

            path_only = raw.split("#", 1)[0].strip()
            if not path_only:
                continue

            out_target = (out_md.parent / Path(path_only)).resolve()
            try:
                rel_out_target = out_target.relative_to(OUT).as_posix()
            except ValueError:
                continue

            if rel_out_target.lower().endswith(".md"):
                continue
            if rel_out_target in seen:
                continue
            seen.add(rel_out_target)

            src = ROOT / Path(rel_out_target)
            if not src.exists() or not src.is_file():
                continue

            dst = OUT / Path(rel_out_target)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    return copied


def write_sidebar() -> None:
    lines = [
        "# Navigation",
        "",
        "- [Home](<Home.md>)",
        "- [Notes](<notes/index.md>)",
        "- [References](<references/index.md>)",
        "- [Concepts](<concepts/index.md>)",
        "- [Entities](<entities/index.md>)",
        "- [Synthesis](<synthesis/index.md>)",
        "- [Journal](<journal/index.md>)",
        "- [Projects](<projects/index.md>)",
        "- [Log](<log.md>)",
    ]
    (OUT / "_Sidebar.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    md_files = list_markdown_files()
    md_rel_set, stem_map, no_ext_map = build_indexes(md_files)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for src in md_files:
        rel = src.relative_to(ROOT).as_posix()
        text = src.read_text(encoding="utf-8")
        converted = convert_text(rel, text, md_rel_set, stem_map, no_ext_map)
        converted = rewrite_existing_markdown_links(rel, converted)
        dst = OUT / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(converted, encoding="utf-8")
        written.append(dst)

    if (OUT / "index.md").exists():
        shutil.copy2(OUT / "index.md", OUT / "Home.md")

    copied_assets = copy_referenced_assets(written + [OUT / "Home.md"] if (OUT / "Home.md").exists() else written)
    write_sidebar()

    print(f"Converted markdown files: {len(written)}")
    print(f"Copied linked assets: {copied_assets}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
