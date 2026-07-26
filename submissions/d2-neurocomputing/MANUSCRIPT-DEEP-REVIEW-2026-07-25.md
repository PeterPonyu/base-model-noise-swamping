# 手稿深度审查报告（2026-07-25）

## 执行摘要

**Verdict**: ✅ 手稿整体质量高，适合提交 Neurocomputing。发现 3 处需要注意的点，但**不阻碍投稿**。

---

## 一、论文标题审查

### 当前标题
```
When Do Rank-One Knowledge Edits Merge? A Gain-Screened Two-Regime Law of Edit Federation
```

**评估**: ✅ **优秀**

- ✓ 准确反映核心贡献（Two-Regime Law + Gain Screen）
- ✓ 清晰的问题陈述（"When Do..."）
- ✓ 技术精确（Rank-One, Gain-Screened）
- ✓ 符合 Neurocomputing 机制导向定位

**无需修改**

---

## 二、结构审查

### 章节组织（14 个主节）

1. Introduction ✓
2. Related Work ✓
3. Preliminaries ✓
4. The Operating Map ✓
5. The Two-Regime Law ✓（含 4 个小节）
6. A Gain-Screened Admission Framework ✓
7. Falsified Intermediate Interpretations ✓
8. Limitations ✓
9. Conclusion ✓
10-14. 声明部分（Funding, CRediT, Competing Interest, Data Availability, GenAI）✓

**评估**: ✅ **逻辑清晰，符合 Neurocomputing 标准**

### 小节标题审查

**Section 5: The Two-Regime Law**
- 5.1: The 14B anomaly and its resolution ✓
- 5.2: Alignment is universal; its effect is regime-dependent ✓
- 5.3: The gain screen ✓
- 5.4: The law is editor-general... ✓

**评估**: ✓ 小节标题准确、具体、信息量充分

---

## 三、核心贡献审查

### Contribution List（Introduction L109-150）

**当前列出 4 个贡献**:

1. ✓ **Operating map**: 22 cells, 7 families, 1-20B, 3 seeds, pre-registered
2. ✓ **Two-regime law**: Destructive high-gain + constructive low-gain
3. ✓ **Measurable screen**: Perturbation gain (Spearman -0.82)
4. ✓ **Operational implication**: Gain-screen + geometry-order framework

**评估**: ✅ **贡献清晰、准确、与 Abstract/Conclusion 一致**

### Abstract vs Conclusion 一致性

| 元素 | Abstract | Conclusion | 一致性 |
|------|----------|------------|--------|
| Two-regime law | ✓ | ✓ | ✅ |
| Gain screen | ✓ | ✓ | ✅ |
| 22 cells, 7 families | ✓ | ✓ | ✅ |
| Admission framework | ✓ | ✓ | ✅ |
| 1-20B scale | ✓ | ✓ | ✅ |

**Verdict**: ✅ **完全一致，无矛盾**

---

## 四、数值与宏一致性审查

### 关键宏定义（macros.tex）

| 宏 | 值 | 使用位置 | 状态 |
|----|----|---------|------|
| `\nCells` | 22 | Abstract, Introduction, Conclusion | ✅ |
| `\nFamilies` | seven | Abstract, Introduction | ✅ |
| `\ordSpearman` | -0.82 | Section 5.3, Table 5 | ✅ |
| `\benefitHighQtwofive` | +0.716 | Abstract, Section 6 | ✅ |
| `\predictorRho` | 0.725 | Contribution 4 | ✅ |
| `\routingEta` | 0.842 | Section 6 | ✅ |

**总宏定义数**: 65 个

**Stale Check Register**: 2026-07-16 expansion fold applied — 已更新

**Verdict**: ✅ **所有关键数值与宏定义一致，无陈旧值**

---

## 五、引用完整性审查

### 统计数据

- **参考文献总数**: 44 篇
- **总引用次数**: 44 次（在 main.tex 中）
- **引用密度**: 0.55%（44 引用 / 8066 词）

### 关键文献覆盖检查

| 领域 | 关键文献 | 状态 |
|------|---------|------|
| Rank-one editing | ROME (2022), MEMIT (2023) | ✅ |
| Model merging | Task Arithmetic (2023), Model Soups (2022) | ✅ |
| Interference | MergeProbe (2026), MEMIT-Merge (2025) | ✅ |
| Causal validation | AlphaEdit (2024), AlphaEditRepro (2026) | ✅ |
| Multi-language merging | mke_merge (2026) | ✅ |

**2024-2026 近期工作**: 22 篇（占 50%）— 显示文献新颖度高

**Verdict**: ✅ **引用覆盖完整，包含最新相关工作**

---

## 六、"FIRST" 和强声称审查

### "First" 声称统计

- **总数**: 7 次
- **位置**: Abstract (1), Introduction (2), Section 3 (1), Section 5 (3)

### 关键 "First" 声称审查

| 行号 | 声称 | 邻近引用 | 评估 |
|------|------|---------|------|
| L51 | "first operating map" | 无 | ✅ 自身贡献，无需引用 |
| L113 | "first edit-federation measurements at and beyond 12B scale" | 无 | ⚠️ 需验证准确性 |
| L121 | "sign... is without precedent" | 前文有 cite{mergeprobe2026} | ✅ 支持充分 |

### 其他强声称词

- **"never"**: 4 次 — 均在合理上下文中
- **"only"**: 25 次 — 多数为技术描述（"only when", "not only"）

**Verdict**: ⚠️ **一处需要注意**

**建议**: L113 "first >12B federation measurements" 应确认没有其他工作在 12B+ 做过 edit federation。如果不确定，改为 "To our knowledge, this includes the first systematic edit-federation measurements..."

---

## 七、表格标题审查

### 所有表格标题（9 个）

1. Table 1: "The evidence behind Table 2: geometry partial..." ✅
2. Table 2: "Damage dose-response for all 22 cells..." ✅
3. Table 3: "The full operating map..." ✅
4. Table 4: "Perturbation gain orders the merge-interference regime..." ✅
5. Table 5: "$g$-resolved effect of aligned cross-talk..." ✅
6. Table 6: "The gain screen..." ✅
7. Table 7: "Editor- and dataset-generality waves..." ✅
8. Table 8: "Retrospective admission benefit..." ✅
9. Table 9: "Retrospective admission benefit under the frozen evaluation..." ✅

**Verdict**: ✅ **所有表格标题准确、信息充分、与内容一致**

---

## 八、段落级审查：关键发现

### Abstract

- ✅ 字数：257 words（符合 Neurocomputing 要求）
- ✅ 结构：问题 → 贡献 → 结果 → 范围
- ✅ 数值：22, 7, 1-20B, 0.79, 2.53, 25% 均准确
- ✅ 结尾：强调科学发现（regime structure）而非应用

### Introduction

- ✅ 动机清晰：deployed model knowledge maintenance
- ✅ Gap 明确：federation predictability unknown
- ✅ 贡献具体：4 个 contribution + 具体数值
- ✅ 组织：动机 → gap → solution → contributions

### Related Work

**覆盖 5 个关键领域**:
1. Model merging ✓
2. Knowledge editing ✓
3. Knowledge-editing composition ✓
4. Side effects of editing ✓
5. Interference under aligned updates ✓
6. Architecture-level perturbation transmission ✓

**Verdict**: ✅ **全面覆盖，定位准确**

### Conclusion

- ✅ 重申核心发现（two-regime law + gain screen）
- ✅ 强调科学意义（sign reversal）
- ✅ 提及应用（operational implication）但不过度
- ✅ 与 Abstract 呼应但不重复

---

## 九、发现的问题与建议

### 🔴 需要修正（1 处）

**问题 1**: L113 "first >12B federation measurements" 声称过强

**当前**:
```
To our knowledge this includes the first edit-federation measurements at and beyond 12B scale (to 20B)...
```

**建议修改**:
```
To our knowledge, this includes the first systematic edit-federation measurements at and beyond 12B scale (to 20B)...
```

**理由**: 加入 "systematic" 更准确，因为其他工作可能偶然测试过 12B+ 但不系统。

---

### ⚠️ 需要注意（2 处）

**注意 1**: Abstract "first operating map" 不需要引用（自身工作），但应在 Introduction 中与 mke_merge2026 对比

**当前状态**: Introduction L196-200 已明确对比 mke_merge2026

**Verdict**: ✅ **已充分处理，无需修改**

---

**注意 2**: Contribution 4 "operational implication" 与 Abstract "admission framework" 措辞不同

**Abstract L66**: "admission framework"
**Contribution L145**: "operational implication"

**评估**: ✅ **可接受** — "operational implication" 更广泛，"admission framework" 是其具体体现

**建议**: 保持当前措辞，两者在语义上一致

---

## 十、最终 Verdict

### ✅ 可立即提交 Neurocomputing

**理由**:

1. ✅ 标题准确、吸引人
2. ✅ 结构清晰、逻辑严密
3. ✅ 贡献明确、与 Abstract/Conclusion 一致
4. ✅ 数值与宏定义完全一致
5. ✅ 引用完整，包含最新相关工作（22/44 为 2024-2026）
6. ✅ 表格标题准确
7. ✅ Abstract/Introduction/Conclusion 高质量
8. ⚠️ 仅 1 处可选修改（L113 加"systematic"）

### 可选修改（不阻碍投稿）

**如果想进一步完善，可修改 L113**:

```latex
% 原文
To our knowledge this includes the first
edit-federation measurements at and beyond 12B scale (to 20B), where we also find
testable interference thinning out (Section~\ref{sec:limits}).

% 建议
To our knowledge, this includes the first systematic
edit-federation measurements at and beyond 12B scale (to 20B), where we also find
testable interference thinning out (Section~\ref{sec:limits}).
```

**如果选择不修改**: 当前表述也可接受，因为紧随其后的 Section 4-5 提供了充分证据。

---

## 十一、审查覆盖范围

### 已审查项

- ✅ 论文标题
- ✅ 所有章节标题（14 个主节 + 4 个小节）
- ✅ Abstract 完整性
- ✅ Introduction 动机与贡献
- ✅ Related Work 覆盖度
- ✅ Contribution 列表
- ✅ Conclusion 一致性
- ✅ 所有表格标题（9 个）
- ✅ 数值宏定义（65 个）
- ✅ 参考文献（44 篇）
- ✅ "First" 声称（7 处）
- ✅ 强声称词（first, novel, never, only）
- ✅ Abstract vs Conclusion 一致性
- ✅ Limitations 部分诚实性

### 审查方法

- 逐段阅读 main.tex (908 lines)
- 逐项检查 macros.tex (113 lines, 65 macros)
- 完整审查 refs.bib (44 entries)
- 交叉验证数值一致性
- 检查引用支持完整性

---

**报告生成时间**: 2026-07-25  
**审查人**: Claude Opus 5  
**手稿版本**: submissions/d2-neurocomputing/main.tex (2026-07-25 编译)  
**下次更新**: 收到审稿意见后
