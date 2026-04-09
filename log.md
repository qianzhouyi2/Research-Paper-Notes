---
title: Wiki 日志
category: log
tags:
  - log
  - wiki
updated: 2026-04-10
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
- [2026-04-08 22:48:28 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-online-refresh" pages_updated=17 pages_created=10 mode=append notes="online metadata refresh: corrected author/entity attribution for CIF, implicit CoT, GPT-ST, Math-Shepherd, MoA, and SoT; removed 7 misattributed entity pages"
- [2026-04-08 23:10:38 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-online-refresh-2" pages_updated=24 pages_created=0 mode=append notes="completed online verification for the remaining 20 reference cards, added 4 author-entity anchors, and closed reference-level metadata coverage to 26/26"
- [2026-04-08 23:43:13 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-concepts-online-refresh" pages_updated=31 pages_created=0 mode=append notes="reviewed all 50 concept pages against primary sources and added online supplement sections to 27 lightweight concept cards"
- [2026-04-09 00:07:09 +08:00] UPDATE source="workspace/wiki-update-2026-04-08-concepts-online-refresh-2" pages_updated=27 pages_created=0 mode=append notes="closed the remaining concept-layer online supplements across adversarial attack, plasticity, Hyena, GPT-ST, CIF, and world-model probing clusters"
- [2026-04-09 00:41:56 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-entities-online-refresh" pages_updated=71 pages_created=0 mode=append notes="added online supplement sections to all 67 entity pages and synced entities/index, home index, coverage audit, and manifest metadata"
- [2026-04-09 09:28:17 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-synthesis-online-refresh" pages_updated=25 pages_created=0 mode=append notes="added online supplement sections to all 21 synthesis pages and synced synthesis/index, home index, coverage audit, and manifest metadata"
- [2026-04-09 11:57:24 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-dnn-concept" pages_updated=6 pages_created=1 mode=append notes="added English-named concept card Deep Neural Network (DNN), linked from segmentation source note, and synced concept/reference/note indexes with manifest"
- [2026-04-09 12:01:44 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-semantic-segmentation-concept" pages_updated=6 pages_created=1 mode=append notes="added concept card Semantic Segmentation with online supplement, linked it from source/note/reference pages, and synced index plus manifest metadata"
- [2026-04-09 12:08:46 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-segmentation-model-entities" pages_updated=8 pages_created=4 mode=append notes="added FCN/PSPNet/DeepLabv3/MaskFormer entity cards, cross-linked SegFormer and segmentation note/reference/concept pages, and synced entities index plus manifest metadata"
- [2026-04-09 12:19:30 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-vit-blackbox-online" pages_updated=4 pages_created=4 mode=append notes="added concept cards for ViT/SignSGD/SimBA/Square Attack with online verification, refreshed query-efficient evaluation links, and synced concept/home indexes with manifest"
- [2026-04-09 12:37:22 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-segmentation-attack-priors" pages_updated=7 pages_created=2 mode=append notes="added concept cards for Indirect Local Attack in Segmentation and SegPGD, linked them into the Delving segmentation attack thread, and synced concept/home indexes with manifest"
- [2026-04-09 12:36:48 +08:00] TAG_NORMALIZE tags_renamed=8 pages_modified=10 new_tags_added=1 notes="reconstructed _meta/taxonomy.md, normalized ambiguous legacy tags, and backfilled frontmatter on journal/projects index pages"
- [2026-04-09 16:12:38 +08:00] UPDATE source="workspace/wiki-update-2026-04-09-nes-linf-calibration" pages_updated=10 pages_created=2 mode=append notes="added concept cards for NES and L-infinity Norm Ball, calibrated Random Attack / NES / l-infinity explanations in the Delving segmentation attack thread, and synced concept/home indexes with manifest"
- [2026-04-09 22:01:44 +08:00] UPDATE source="papers_sources/semantic_segmentation_robustness_20260409" pages_updated=4 pages_created=2 mode=append notes="synced 28-paper semantic segmentation robustness corpus (22 arXiv TeX sources + 6 PDF fallback) into synthesis/journal and refreshed indexes"
- [2026-04-10 11:30:00 +08:00] INGEST source="workspace/wiki-ingest-2026-04-10-seg-robustness-batches" pages_updated=7 pages_created=125 mode=append notes="per-paper verified ingest for 28 segmentation robustness papers; added 30 concepts, 60 entities, and 6 synthesis pages"
- [2026-04-10 12:05:00 +08:00] UPDATE source="workspace/wiki-update-2026-04-10-seg-robustness-deepening" pages_updated=30 pages_created=1 mode=append notes="deepened 28 per-paper reference cards with local abstract evidence and added a batch reading matrix synthesis page"
- [2026-04-10 12:26:00 +08:00] UPDATE source="workspace/wiki-update-2026-04-10-seg-robustness-structure-pass" pages_updated=28 pages_created=0 mode=append notes="added local TeX section-structure evidence to per-paper reference cards and saved structure extraction artifact"
- [2026-04-10 13:10:00 +08:00] UPDATE source="workspace/wiki-update-2026-04-10-full-online-pass" pages_updated=100 pages_created=0 mode=append notes="completed one-by-one online supplement for extracted concepts/entities/synthesis pages in segmentation robustness corpus"
- [2026-04-10 13:32:00 +08:00] UPDATE source="workspace/wiki-update-2026-04-10-online-pass-encoding-fix" pages_updated=104 pages_created=0 mode=append notes="replaced garbled online-supplement sections with clean one-by-one online supplement blocks"
- [2026-04-10 13:45:00 +08:00] UPDATE source="workspace/wiki-update-2026-04-10-global-online-pass" pages_updated=252 pages_created=0 mode=append notes="normalized online supplement sections for all concepts/entities/synthesis pages"
- [2026-04-10 13:52:00 +08:00] UPDATE source="workspace/wiki-update-2026-04-10-index-cleanup" pages_updated=7 pages_created=0 mode=append notes="removed garbled heading blocks from index pages and normalized updated dates"
