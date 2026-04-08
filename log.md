---
title: Wiki 日志
category: log
tags:
  - log
  - wiki
updated: 2026-04-08
summary: 记录 wiki 初始化、ingest 与维护操作的时间线。
---

# Wiki 日志

- [2026-04-08 11:13:50 +08:00] INIT vault_path="D:/paper_reading/paper_reading" sources="D:/paper_reading/paper_reading/papers_sources" preserved_existing_folders="notes,papers_sources,Templates"
- [2026-04-08 11:36:20 +08:00] INGEST source="papers_sources/Research-Paper-Notes" pages_updated=2 pages_created=6 mode=append
- [2026-04-08 12:40:32 +08:00] INGEST source="papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation" pages_updated=8 pages_created=10 mode=append
- [2026-04-08 16:35:11 +08:00] INGEST source="notes/_notion_raw/Are aligned neural networks adversarially aligned.json" pages_updated=2 pages_created=1 mode=append
- [2026-04-08 16:37:09 +08:00] INGEST source="notion:Maintaining-Plasticity-in-Deep-Continual-Learning" pages_updated=1 pages_created=1 mode=append notes="template 1-15, 19 images localized, arxiv source downloaded"
- [2026-04-08 16:47:00 +08:00] LINT source="notes/Are aligned neural networks adversarially aligned.md" pages_updated=1 pages_created=0 mode=append notes="added 15.3 self-check; verified image links and no legacy dump sections"
- [2026-04-08 16:48:00 +08:00] LINT source="notes/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature.md" pages_updated=1 pages_created=0 mode=append notes="added 15.3 self-check; verified image links and no legacy dump sections"
- [2026-04-08 16:48:27 +08:00] INGEST source="notes" pages_updated=3 pages_created=0 mode=append
- [2026-04-08 17:06:10 +08:00] INGEST source="papers_sources/Are aligned neural networks adversarially aligned 2306.15447" pages_updated=5 pages_created=3 mode=append
- [2026-04-08 17:06:35 +08:00] INGEST source="papers_sources/Hyena Hierarchy 2302.10866" pages_updated=4 pages_created=2 mode=append
- [2026-04-08 17:06:58 +08:00] INGEST source="papers_sources/Maintaining Plasticity in Deep Continual Learning" pages_updated=4 pages_created=3 mode=append
- [2026-04-08 17:07:24 +08:00] INGEST source="papers_sources/MeanSparse 2406.05927" pages_updated=5 pages_created=2 mode=append
- [2026-04-08 17:07:41 +08:00] INGEST source="papers_sources/2306.15447.tar" pages_updated=3 pages_created=0 mode=append
- [2026-04-08 17:07:57 +08:00] INGEST source="papers_sources/2302.10866.tar" pages_updated=3 pages_created=0 mode=append
- [2026-04-08 17:08:15 +08:00] INGEST source="papers_sources/2402.01220-source.tar" pages_updated=2 pages_created=0 mode=append
- [2026-04-08 17:18:30 +08:00] UPDATE source="workspace/wiki-update-2026-04-08" pages_updated=3 pages_created=1 mode=append notes="added reusable playbook for manual append workflow and sync checks"
- [2026-04-08 17:29:05 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-deep" pages_updated=7 pages_created=37 mode=append notes="expanded notes distillation into references/concepts/entities/synthesis and refreshed all major indexes"
- [2026-04-08 17:29:20 +08:00] CROSS_LINK pages_scanned=110 links_added=148 pages_modified=38 orphans_remaining=0
- [2026-04-08 17:34:54 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-authors" pages_updated=8 pages_created=4 mode=append notes="added cross-paper author entities and back-linked related references"
- [2026-04-08 17:35:05 +08:00] CROSS_LINK pages_scanned=110 links_added=23 pages_modified=12 orphans_remaining=0
- [2026-04-08 17:40:21 +08:00] AUDIT source="workspace/wiki-coverage-audit-2026-04-08" pages_updated=2 pages_created=1 mode=append notes="full gap audit completed for notes->references/concepts/entities/synthesis coverage"
- [2026-04-08 17:55:47 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-gap-fill" pages_updated=20 pages_created=20 mode=append notes="closed concept/synthesis hard gaps and expanded author/model entities"
- [2026-04-08 17:56:10 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-note-link-pass" pages_updated=25 pages_created=0 mode=append notes="added per-note Wiki association block to all notes"
- [2026-04-08 17:56:22 +08:00] CROSS_LINK pages_scanned=135 links_added=140 pages_modified=53 orphans_remaining=0
- [2026-04-08 18:43:08 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-papers-sources-deep" pages_updated=38 pages_created=35 mode=full notes="paper-by-paper deepening from papers_sources: refined concepts/entities/synthesis, upgraded 9 references and 9 note Wiki links"
- [2026-04-08 19:31:34 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-notes-fine-scan" pages_updated=27 pages_created=11 mode=append notes="fine-scanned notes and strengthened concept/entity/synthesis links; created 5 concepts, 3 entities, 3 synthesis pages"
- [2026-04-08 20:36:22 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-notes-fine-scan-2" pages_updated=19 pages_created=11 mode=append notes="second semantic pass on notes: added probabilistic inference, soft monotonic alignment, multimodal two-stage CoT, PEFT anchors and linked related entities/synthesis"
- [2026-04-08 21:00:23 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-notes-fine-scan-3" pages_updated=23 pages_created=18 mode=append notes="third semantic pass on notes: raised all notes to >=3 concept/entity anchors, added temporal/process synthesis pages, and filled author/dataset entities"
- [2026-04-08 21:52:23 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-notes-fine-scan-4" pages_updated=13 pages_created=2 mode=append notes="fourth semantic pass on notes: added adaptation/plasticity and representation-capacity synthesis pages, and raised all notes to >=3 synthesis anchors"
