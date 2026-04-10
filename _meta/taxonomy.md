---
title: Tag Taxonomy
category: meta
tags:
  - taxonomy
  - wiki
  - workflow
  - audit
updated: 2026-04-10
summary: Authoritative canonical tag vocabulary, usage rules, and migration guidance for this vault.
sources:
  - workspace/wiki-update-2026-04-10-global-lint-remediation
created: 2026-04-09
---

# Tag Taxonomy

## Scope

This file is the authoritative source for frontmatter `tags` in this vault.

- It applies to `concepts/`, `references/`, `entities/`, `synthesis/`, `journal/`, `projects/`, and future normalized `notes/`.
- New tags should be added here before they are used in content pages.
- Raw `_notion_raw` exports are not part of the controlled vocabulary unless they are promoted into user-facing pages.

## Audit Snapshot (2026-04-09)

- Pages scanned: `213`
- Pages with frontmatter tags: `187`
- Pages without frontmatter tags: `26`
- Pages over the `5`-tag limit: `0`
- Live canonical tags after this pass: `82`
- Deprecated or normalized legacy tags in this pass:
  - `coverage`
  - `wiki-update`
  - `adversarial`
  - `gnn`
  - `feature-engineering`
  - `feature-sparsification`
  - `imported`
  - `temporal`
- Outstanding backlog:
  - `24` imported note pages still use body text without frontmatter tags
  - `2` raw `_notion_raw` exports are intentionally left outside normalization

## Global Rules

- Use lowercase ASCII tags only.
- Use hyphenated multi-word tags, not spaces or camelCase.
- Maximum `5` tags per page.
- Every tagged page must have exactly one primary page-role tag:
  - `index`
  - `note`
  - `paper`
  - `concept`
  - `entity`
  - `synthesis`
  - `journal`
  - `project`
- Index pages should use `index` plus the section tag when that section has one, such as `reference`, `journal`, or `project`.
- Entity pages must use `entity` plus exactly one subtype:
  - `author`
  - `model`
  - `dataset`
  - `benchmark`
- Workflow tags such as `wiki`, `audit`, `workflow`, `ingest`, and `taxonomy` are for vault-maintenance pages, not ordinary research content.
- Do not use person names, model names, dataset names, venues, or years as tags. Represent them with pages in `entities/`.
- Only use abbreviations when they are explicitly canonical in this vault:
  - `llm`
  - `cot`
  - `mcts`
  - `asr`

## Recommended Tag Recipes

- Concept page:
  - `concept` + `2-4` stable method or topic tags
- Reference card:
  - `paper` + `2-4` topic or method tags
- Entity page:
  - `entity` + one subtype tag + `1-3` domain tags
- Synthesis page:
  - `synthesis` + `2-4` thematic tags
- Journal or audit page:
  - `journal` + optional `audit` and `wiki`
- Project page:
  - `project` + `2-4` execution-theme tags
- Future normalized note page:
  - `note` + `2-4` topic tags

## Canonical Tags

### Page Role And Governance

- `index`: section landing page or directory entry page.
- `note`: long-form reading note or imported paper note.
- `paper`: single-paper summary or reference card.
- `concept`: reusable method, term, metric, or protocol card.
- `entity`: author, model, dataset, or benchmark card.
- `synthesis`: cross-paper synthesis page.
- `journal`: process log, audit note, or maintenance journal.
- `project`: execution-oriented project page.
- `reference`: section-level bibliography tag; use on `references/index.md`, not on individual paper cards.
- `wiki`: vault-maintenance or knowledge-base operation page.
- `workflow`: reusable operating playbook.
- `audit`: coverage, quality, or consistency audit page.
- `ingest`: import or append workflow page.
- `taxonomy`: tag-governance page.

### Entity Subtype

- `author`: person entity page.
- `model`: model or architecture-family entity page.
- `dataset`: dataset entity page.
- `benchmark`: benchmark or leaderboard entity page.

### Research Domains And Problem Areas

- `llm`: large language model content.
- `llm-safety`: safety, alignment risk, or harmful-behavior evaluation for LLM systems.
- `multimodal`: content spanning more than one modality.
- `multi-agent`: coordinated multi-agent systems or methods.
- `collaboration`: collaboration mechanism or cooperation pattern.
- `long-context`: long-context modeling or evaluation.
- `spatio-temporal`: spatio-temporal structure or prediction.
- `sequence-modeling`: sequence-modeling as the main research problem family.
- `graph-learning`: graph-structured learning or graph-based representation.
- `diffusion`: diffusion-model content.
- `continual-learning`: continual learning or plasticity-retention work.
- `semantic-segmentation`: semantic segmentation task or benchmark content.
- `speech`: speech domain content.
- `traffic`: traffic forecasting or traffic data domain content.

### Threat, Robustness, And Evaluation Context

- `adversarial-attack`: attack construction, attack family, or threat-model page.
- `adversarial-robustness`: robustness property, defense, or robustness-evaluation page.
- `black-box-attack`: black-box attack setting or method.
- `query-efficient`: query-budget-sensitive attacks or evaluations.
- `robustness`: general robustness theme not limited to adversarial settings.
- `security`: broader security research context.
- `model-security`: model-centric security or adversarial model-security context.
- `perturbation`: perturbation design or perturbation interaction as a central mechanism.
- `evaluation`: evaluation protocol, metric family, or benchmarking method.
- `diagnostics`: diagnostic tools or measurement procedures.
- `interpretability`: probing, explanation, or interpretability methods.
- `asr`: automatic speech recognition in this vault, not attack success rate.

### Core Method And Training Tags

- `architecture`: architecture or backbone design.
- `representation`: representation learning or representation geometry.
- `optimization`: optimization method or objective design.
- `efficiency`: training or inference efficiency.
- `inference`: inference-time behavior or control.
- `generation`: generative modeling or generation procedure.
- `reasoning`: reasoning method, reasoning structure, or reasoning evaluation.
- `search`: explicit search procedure.
- `routing`: compute or path routing.
- `orchestration`: multi-step system orchestration.
- `pretraining`: pretraining method or setup.
- `finetuning`: fine-tuning or adaptation after pretraining.
- `parameter-efficient`: parameter-efficient tuning or adaptation.
- `verification`: explicit verification or checking component.
- `process-supervision`: supervision on intermediate steps or traces.
- `probabilistic`: probabilistic formulation or probabilistic control.
- `sampling`: sampling-centric method.
- `reinforcement-learning`: RL framing, author lineage, or RL-style optimization.
- `methodology`: methodology page about how to evaluate or compare methods.
- `adaptation`: adaptation strategy or adaptation synthesis.
- `alignment`: alignment or monotonic alignment mechanism.
- `post-training`: post-training intervention or enhancement.
- `survey`: survey or review paper.

### Technical Motifs

- `transformer`: Transformer architecture is central to the page.
- `convolution`: convolutional mechanism is central to the page.
- `graph`: graph structure as a modeling object.
- `hypergraph`: hypergraph structure as a modeling object.
- `mcts`: Monte Carlo Tree Search.
- `cot`: Chain-of-Thought as a central method object.
- `sparsification`: sparsification is central to the method.
- `plasticity`: plasticity or loss-of-plasticity topic.
- `nonlinearity`: nonlinearity design or compensation.
- `scaling`: scaling law, scale effect, or system scaling.
- `capacity`: capacity, rank, or representational-capacity topic.
- `sequence`: sequence object or sequence transduction, when narrower than `sequence-modeling`.
- `compositionality`: compositional generation or compositional reasoning.
- `bayesian`: Bayesian framing or Bayesian inference.
- `probing`: probing methodology.

## Canonical Usage Notes

- `paper` vs `reference`:
  - Use `paper` on individual reference cards.
  - Use `reference` only for the `references/` section landing page or future bibliography-only index pages.
- `graph` vs `graph-learning`:
  - Use `graph` when the object itself is graph-structured.
  - Use `graph-learning` when the page is about the broader graph-learning research area.
- `sequence` vs `sequence-modeling`:
  - Use `sequence` for narrower sequence or transduction mechanics.
  - Use `sequence-modeling` for the broader model family or synthesis theme.
- `security` vs `model-security`:
  - Use `security` for broader security context or security researchers.
  - Use `model-security` when the core topic is model-centric adversarial or robustness security.
- `transformer`:
  - Use only when Transformer structure is part of what the page is about.
  - Do not add it to every LLM page by default.

## Legacy Tags, Aliases, And Migration Rules

| Legacy tag | Canonical handling | Rule |
| --- | --- | --- |
| `coverage` | `audit` | Coverage pages are audits. Rename directly. |
| `wiki-update` | `wiki` | If the page is also an operational guide, add `workflow`; if it is an audit, add `audit`. |
| `gnn` | `graph-learning` | Use `graph-learning` unless a future dedicated graph-neural-network tag is introduced. |
| `feature-sparsification` | `sparsification` | Direct rename. |
| `feature-engineering` | manual choice | Replace with `representation`, `optimization`, or `sparsification` depending on the page. |
| `adversarial` | manual choice | Replace with `adversarial-attack` or `adversarial-robustness`; do not keep the ambiguous form. |
| `temporal` | manual choice | Replace with `spatio-temporal` or `sequence-modeling`. |
| `imported` | `ingest` or remove | Use `ingest` on workflow or import-index pages; otherwise remove it. |

## Current Normalization Decisions (2026-04-09)

- `journal/Wiki Coverage Audit 2026-04-08.md`:
  - `coverage` -> removed in favor of `audit`
- `journal/Papers Sources Deep Gap Audit 2026-04-08.md`:
  - `wiki-update` -> `wiki`
- `synthesis/Robust Representation and Adversarial Dynamics.md`:
  - `adversarial` -> `adversarial-robustness`
- `references/GPT-ST Spatio-Temporal Pretraining.md`:
  - `gnn` -> `graph-learning`
- `references/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification.md`:
  - `feature-sparsification` -> `sparsification`
- `concepts/Mean-Centered Feature Sparsification.md`:
  - `feature-engineering` -> `sparsification`
- `notes/Research-Paper-Notes 导入索引.md`:
  - `imported` -> `ingest`
- `synthesis/Temporal Structure Learning in Sequence Models.md`:
  - `temporal` -> `sequence-modeling`
- Added reserved canonical tag:
  - `project`

## Backlog

- Normalize frontmatter for imported `notes/*.md` pages when you want note-level tag search to become reliable.
- Keep `_notion_raw` exports unnormalized unless they are promoted into visible knowledge pages.
- Re-run a full tag audit after the next large ingest batch or after a bulk note-frontmatter normalization pass.
