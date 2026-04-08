from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

NOTION_API = "https://www.notion.so/api/v3"
ROOT_PAGE_ID = "18e84951-e962-8022-bda7-e29fdeb3e883"
SITE_BASE = "https://saputello.notion.site"
READ_DATE = "2026-04-08"
TIMEOUT = 30

@dataclass
class ImportResult:
    title: str
    page_id: str
    note_path: Path
    raw_path: Path
    block_count: int
    unresolved_count: int


def post_json(path: str, payload: dict) -> dict:
    resp = requests.post(f"{NOTION_API}/{path}", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_page_chunk(page_id: str) -> Dict[str, dict]:
    payload = {
        "pageId": page_id,
        "limit": 100,
        "cursor": {"stack": []},
        "chunkNumber": 0,
        "verticalColumns": False,
    }
    data = post_json("loadCachedPageChunk", payload)
    return data.get("recordMap", {}).get("block", {}) or {}


def sync_blocks(block_ids: List[str]) -> Dict[str, dict]:
    if not block_ids:
        return {}
    payload = {"requests": [{"id": i, "table": "block", "version": -1} for i in block_ids]}
    data = post_json("syncRecordValues", payload)
    return data.get("recordMap", {}).get("block", {}) or {}


def block_value(entry: dict) -> dict:
    if not isinstance(entry, dict):
        return {}
    if isinstance(entry.get("value"), dict) and isinstance(entry["value"].get("value"), dict):
        return entry["value"]["value"]
    if isinstance(entry.get("value"), dict):
        return entry["value"]
    return {}


def rich_plain(parts: Optional[list]) -> str:
    if not parts:
        return ""
    out: List[str] = []
    for seg in parts:
        if isinstance(seg, list) and seg:
            text = seg[0]
            if isinstance(text, str):
                out.append(text)
    return "".join(out).strip()


def prop_plain(v: dict, key: str) -> str:
    props = v.get("properties") or {}
    return rich_plain(props.get(key))


def gather_referenced_ids(blocks: Dict[str, dict], root_id: str) -> Set[str]:
    refs: Set[str] = set()
    stack: List[str] = [root_id]
    seen: Set[str] = set()
    while stack:
        bid = stack.pop()
        if bid in seen:
            continue
        seen.add(bid)
        refs.add(bid)
        v = block_value(blocks.get(bid, {}))
        for child_id in v.get("content") or []:
            if isinstance(child_id, str):
                stack.append(child_id)
    return refs


def resolve_page_blocks(page_id: str) -> Tuple[Dict[str, dict], Set[str]]:
    blocks = fetch_page_chunk(page_id)
    for _ in range(20):
        refs = gather_referenced_ids(blocks, page_id)
        missing = sorted(x for x in refs if x not in blocks)
        if not missing:
            return blocks, set()
        before = len(blocks)
        for i in range(0, len(missing), 100):
            synced = sync_blocks(missing[i:i+100])
            for bid, entry in synced.items():
                if bid not in blocks and block_value(entry):
                    blocks[bid] = entry
        if len(blocks) == before:
            return blocks, set(missing)
    refs = gather_referenced_ids(blocks, page_id)
    return blocks, {x for x in refs if x not in blocks}


def block_title(v: dict) -> str:
    return prop_plain(v, "title")


def block_url(v: dict) -> str:
    fmt = v.get("format") or {}
    props = v.get("properties") or {}
    for k in ("display_source", "source"):
        if isinstance(fmt.get(k), str) and fmt.get(k):
            return fmt[k]
    for k in ("source", "link", "url"):
        txt = rich_plain(props.get(k))
        if txt:
            return txt
    return ""


def bool_prop(v: dict, key: str) -> bool:
    return prop_plain(v, key).lower() in {"yes", "true", "1", "checked", "x", "已完成", "是"}


def render_block(bid: str, blocks: Dict[str, dict], visited: Set[str], indent: int = 0) -> List[str]:
    if bid in visited:
        return []
    visited.add(bid)
    if bid not in blocks:
        return [f"{'  ' * indent}- [缺失块] `{bid}`"]
    v = block_value(blocks[bid])
    btype = v.get("type", "unknown")
    title = block_title(v)
    children = [c for c in (v.get("content") or []) if isinstance(c, str)]
    pre = "  " * indent
    lines: List[str] = []

    if btype == "text":
        if title:
            lines.append(f"{pre}{title}")
    elif btype == "header":
        lines.append(f"{pre}# {title}")
    elif btype == "sub_header":
        lines.append(f"{pre}## {title}")
    elif btype == "sub_sub_header":
        lines.append(f"{pre}### {title}")
    elif btype == "bulleted_list":
        lines.append(f"{pre}- {title or '[空项]'}")
    elif btype == "numbered_list":
        lines.append(f"{pre}1. {title or '[空项]'}")
    elif btype == "to_do":
        lines.append(f"{pre}- [{'x' if bool_prop(v, 'checked') else ' '}] {title or '[空项]'}")
    elif btype == "quote":
        lines.append(f"{pre}> {title}")
    elif btype == "callout":
        lines.append(f"{pre}> [!note] {title}")
    elif btype == "divider":
        lines.append(f"{pre}---")
    elif btype in {"code", "codepen"}:
        lang = (v.get("format") or {}).get("code_language") or ""
        lines.append(f"{pre}```{lang}")
        if title:
            lines.append(title)
        lines.append(f"{pre}```")
    elif btype in {"image", "video", "audio", "file", "pdf", "bookmark", "embed"}:
        url = block_url(v)
        caption = prop_plain(v, "caption")
        row = f"{pre}- [{btype}] {url or '(无链接)'}"
        if caption:
            row += f" | {caption}"
        lines.append(row)
    elif btype == "page":
        lines.append(f"{pre}- [子页] {title or bid}")
    else:
        lines.append(f"{pre}- [未专门解析:{btype}] {title or '(无标题)'}")

    for cid in children:
        lines.extend(render_block(cid, blocks, visited, indent + 1))
    return lines


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[<>:\"/\\\\|?*]", "", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(".")
    return (name or "untitled")[:180]


def normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def pick_note_filename(title: str, existing_stems: Dict[str, Path]) -> str:
    full = sanitize_filename(title)
    key = normalize_key(full)
    if key in existing_stems:
        return existing_stems[key].name
    main = sanitize_filename(title.split(":")[0].strip())
    mkey = normalize_key(main)
    if mkey in existing_stems:
        return existing_stems[mkey].name
    return f"{full}.md"


def note_template(title: str, url: str, raw_text: str, page_id: str, block_count: int, unresolved: Set[str], raw_rel: str) -> str:
    unresolved_lines = "\n".join(f"- `{u}`" for u in sorted(unresolved)) if unresolved else "- 无"
    return f"""# {title}

- 阅读日期：[[{READ_DATE}]]
- 状态：已读
- 标签：#paper #imported #notion

## 基本信息

- 题目：{title}
- 链接：[{title}]({url})
- 作者：待从原笔记补充
- 单位：待从原笔记补充
- 发表：待从原笔记补充
- 关键词：待补

## 研究问题

- 论文关注的核心问题：待从原笔记补充
- 为什么这个问题重要：待从原笔记补充
- 论文要优化或解决的目标：待从原笔记补充

## 为什么这个问题难

- 难点 1：待从原笔记补充
- 难点 2：待从原笔记补充
- 难点 3：待从原笔记补充

## 核心思路

- 方法名称：待从原笔记补充
- 整体思路：待从原笔记补充
- 代理目标 / 损失 / 优化目标：待从原笔记补充

## 方法细节

### 1. 模块一

- 做了什么：待从原笔记补充
- 为什么这么设计：待从原笔记补充
- 关键实现：待从原笔记补充

### 2. 模块二

- 做了什么：待从原笔记补充
- 为什么这么设计：待从原笔记补充
- 关键实现：待从原笔记补充

## 主要贡献

- 贡献 1：待从原笔记补充
- 贡献 2：待从原笔记补充
- 贡献 3：待从原笔记补充

## 实验设置

- 数据集：待从原笔记补充
- 模型：待从原笔记补充
- 对比方法：待从原笔记补充
- 评价指标：待从原笔记补充
- 关键超参数：待从原笔记补充

## 关键结果

- 结果 1：待从原笔记补充
- 结果 2：待从原笔记补充
- 结果 3：待从原笔记补充

## 消融实验结论

- 消融点 1：待从原笔记补充
- 消融点 2：待从原笔记补充
- 消融点 3：待从原笔记补充

## 我的理解

- 直观理解：待从原笔记补充
- 最值得关注的设计：待从原笔记补充
- 和已有方法相比的新意：待从原笔记补充
- 我认为最强的一点：待从原笔记补充

## 可能的局限性

- 局限 1：待从原笔记补充
- 局限 2：待从原笔记补充
- 局限 3：待从原笔记补充

## 可以继续思考的问题

- 问题 1：待从原笔记补充
- 问题 2：待从原笔记补充
- 问题 3：待从原笔记补充

## 一句话总结

- 基于 Notion 原始笔记自动导入，待你二次精读后补全结构化内容。

## 原始笔记（Notion 完整转录）

{raw_text}

## 导入完整性

- 源页面 ID：`{page_id}`
- 抓取块数量：`{block_count}`
- 未解析块引用：`{len(unresolved)}`
- 原始 JSON：`{raw_rel}`
- 未解析块 ID：
{unresolved_lines}

## 待确认问题

- [ ] 是否需要我把“原始笔记完整转录”进一步整理成结构化条目？
- [ ] 是否需要我补齐作者、单位、发表信息？
- [ ] 是否需要我把图表链接整理为本地附件？
"""


def import_root_notes(root: Path) -> List[ImportResult]:
    notes_dir = root / "notes"
    raw_dir = notes_dir / "_notion_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_stems: Dict[str, Path] = {}
    for p in notes_dir.glob("*.md"):
        existing_stems[normalize_key(p.stem)] = p

    root_blocks = fetch_page_chunk(ROOT_PAGE_ID)
    root_v = block_value(root_blocks.get(ROOT_PAGE_ID, {}))
    child_ids = [x for x in (root_v.get("content") or []) if isinstance(x, str)]

    results: List[ImportResult] = []
    for cid in child_ids:
        entry = root_blocks.get(cid)
        if not entry:
            synced = sync_blocks([cid])
            entry = synced.get(cid)
            if entry:
                root_blocks[cid] = entry
        if not entry:
            continue
        child_v = block_value(entry)
        if child_v.get("type") != "page":
            continue
        title = block_title(child_v) or cid

        blocks, unresolved = resolve_page_blocks(cid)
        visited: Set[str] = set()
        rendered = render_block(cid, blocks, visited, indent=0)
        raw_text = "\n".join(x for x in rendered if x.strip()).strip() or "- [空白页面]"

        filename = pick_note_filename(title, existing_stems)
        note_path = notes_dir / filename
        raw_name = sanitize_filename(title) + ".json"
        raw_path = raw_dir / raw_name

        raw_dump = {
            "title": title,
            "page_id": cid,
            "root_page_id": ROOT_PAGE_ID,
            "blocks": blocks,
            "unresolved_block_ids": sorted(unresolved),
        }
        raw_path.write_text(json.dumps(raw_dump, ensure_ascii=False, indent=2), encoding="utf-8")

        note = note_template(
            title=title,
            url=f"{SITE_BASE}/{cid.replace('-', '')}",
            raw_text=raw_text,
            page_id=cid,
            block_count=len(blocks),
            unresolved=unresolved,
            raw_rel=str(raw_path.relative_to(root)).replace("\\", "/"),
        )
        note_path.write_text(note, encoding="utf-8")

        existing_stems[normalize_key(note_path.stem)] = note_path
        results.append(ImportResult(title, cid, note_path, raw_path, len(blocks), len(unresolved)))

    return results


def main() -> None:
    root = Path.cwd()
    results = import_root_notes(root)
    report = {
        "root_page_id": ROOT_PAGE_ID,
        "imported_count": len(results),
        "items": [
            {
                "title": r.title,
                "page_id": r.page_id,
                "note_path": str(r.note_path).replace("\\", "/"),
                "raw_path": str(r.raw_path).replace("\\", "/"),
                "block_count": r.block_count,
                "unresolved_count": r.unresolved_count,
            }
            for r in results
        ],
    }
    report_path = root / "notes" / "_notion_raw" / "import_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
