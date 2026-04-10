---
title: LibriSpeech Dataset
category: entity
tags:
- entity
- dataset
- asr
- speech
sources:
- notes/CIF Continuous Integrate-and-Fire.md
created: 2026-04-08
updated: 2026-04-10
summary: LibriSpeech 是英语语音识别常用基准，在 CIF 相关实验中用于评估 WER 与流式对齐效果。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# LibriSpeech Dataset

## 联网补充

- OpenSLR 把 LibriSpeech 定义为约 1000 小时、16kHz 的英文朗读语音语料，来源于 LibriVox 有声书并经过仔细切分与对齐。
- 在本库里，LibriSpeech 是 CIF 这条语音对齐线的基础评测语料，重点在声学帧与输出 token 的对齐质量。

## 关联页面

- [[references/CIF Continuous Integrate-and-Fire]]
- [[concepts/Soft Monotonic Alignment for Sequence Transduction]]
- [[synthesis/Temporal Structure Learning in Sequence Models]]
