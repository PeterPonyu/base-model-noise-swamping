# Neurocomputing 提交检查清单（2026-07-25）

## 一、文件准备（本地完成）

### 核心文件
- [x] `main.pdf` (28 pages, ~400 KB) — 已编译，journal 字段已更新为 Neurocomputing
- [x] `main.tex` — 已修改：删除摘要 L69 "Prospective deployment is future work"
- [x] `main.tex` — 已修改：Section 6 标题改为 "A Gain-Screened Admission Framework"
- [x] `main.tex` — 已修改：contribution 4 改为 "An operational implication"
- [x] `main.tex` — 已修改：Conclusion 改为 "operational implication"
- [x] `cover-letter.txt` — 机制导向，443 words
- [x] `refs.bib` — 44 references
- [x] `macros.tex` — 数值宏定义
- [x] Figures — figA.pdf 到 figE.pdf（5 个主图）

### 元数据文件
- [x] `portal-metadata.txt` — 需修改 journal name，其他保持
- [x] `highlights.txt` — KBS 版本可复用

### 数据可用性
- [x] Zenodo DOI: 10.5281/zenodo.21405273
- [x] GitHub: https://github.com/PeterPonyu/edit-federation-map
- [x] 已在 main.tex L896-898 声明

### GenAI 声明
- [x] 已在 main.tex L900-904 包含（Elsevier 两轨制政策）

---

## 二、关键修改确认

| 位置 | 原文（KBS） | 新文（Neurocomputing） | 状态 |
|------|------------|----------------------|------|
| 摘要 L69 | "Prospective deployment is future work." | "The regime structure and screening law hold across seven architecture families from 1B to 20B parameters." | ✓ 已改 |
| Section 6 标题 | "A Gain-Screened Admission Rule" | "A Gain-Screened Admission Framework" | ✓ 已改 |
| Contribution 4 | "A deployable admission rule" | "An operational implication for knowledge maintenance" | ✓ 已改 |
| Conclusion L876 | "actionable admission rule" | "operational implication" | ✓ 已改 |
| 摘要 L66 | "admission rule" | "admission framework" | ✓ 已改 |
| Journal field | Knowledge-Based Systems | Neurocomputing | ✓ 已改 |

---

## 三、Portal 提交步骤（用户手动完成）

### 3.1 进入 Neurocomputing Editorial Manager
- URL: https://www.editorialmanager.com/neucom/
- 或从 KBS rejection email 点击 "Transfer to Neurocomputing"（但不要直接 final submit）

### 3.2 选择文章类型
- **Research Paper** (标准选项)
- **不选** Review Article / Short Communication

### 3.3 上传文件
1. **Main manuscript**: `main.pdf`
2. **Cover letter**: `cover-letter.txt`
3. **Highlights** (可选): `highlights.txt`
4. **LaTeX source** (revision 时提供，初投只需 PDF)

### 3.4 填写 Metadata
- **Title**: When Do Rank-One Knowledge Edits Merge? A Gain-Screened Two-Regime Law of Edit Federation
- **Abstract**: 从 main.pdf 复制（已去掉 "Prospective deployment" 那句）
- **Keywords**: 建议 6-8 个
  - knowledge editing
  - model merging
  - neural network dynamics
  - rank-one updates
  - edit federation
  - perturbation gain
  - learning systems
  - constructive interference

### 3.5 作者信息
- **Corresponding author**: Zeyu Fu
- **Affiliation**: Army Medical University (Third Military Medical University), Chongqing, China
- **ORCID**: 0009-0001-8329-0108
- **Email**: [用户邮箱]

### 3.6 **关键选项：Open Access**
- ⚠️ **选择 "No, I do not wish to publish open access"**
- 这会走 **Subscription** 路径，无需支付 APC
- Neurocomputing 是 hybrid journal，支持订阅制免费发表

### 3.7 Data Availability Statement
```
The code, edit vectors, and result artifacts supporting the findings of this study are openly available at Zenodo (https://doi.org/10.5281/zenodo.21405273) and GitHub (https://github.com/PeterPonyu/edit-federation-map).
```

### 3.8 Declarations
- **Funding**: None
- **Competing interests**: None
- **Generative AI**: Already included in manuscript (Section after References)

---

## 四、不要做的事

- ❌ **不要**原样点击 KBS transfer email 后直接 final submit
- ❌ **不要**选择 Open Access（会强制收 APC ~$2,930）
- ❌ **不要**上传 KBS 冻结版 PDF（`main.submitted-20260717.pdf`）
- ❌ **不要**使用 KBS 的旧 cover letter
- ❌ **不要**在 Portal 填表时再写 "Prospective deployment is future work"

---

## 五、预期时间线（Neurocomputing 官方数据）

- **Submission to first decision**: ~8 days (median)
- **Acceptance to publication**: ~8 days
- **Total (if accept)**: ~2-3 weeks

---

## 六、如果收到审稿意见

### 可能要求的补充实验（优先级）

1. **不太可能要求**：
   - ✗ Prospective deployment validation — 机制研究不要求
   - ✗ Downstream/global damage 全量测试 — Limitations 已明确边界
   - ✗ 补充架构族 — 当前 8 族已超论文声称

2. **中等可能**：
   - ⚠️ Mistral-Nemo-12B 额外层（如果质疑 12B 证据）
     - 成本：~2-3 GPU-h 或 ¥30-60 云端
     - 当前本地有 12B 证据但可能审稿人想看更多层
   - ⚠️ Norm-based gain estimator (L864 future work)
     - 成本：~1 天 CPU
     - 可在 revision response 补充

3. **低可能**：
   - ⚠️ 更多种子重复（当前 3-seed 已是标准）
   - ⚠️ 额外数据集（当前 CounterFact + zsRE 已覆盖主流）

---

## 七、远端 Box 需求评估

### 当前结论：**初投不需要任何远端 Box**

理由：
1. 本地已有 697 结果文件，覆盖 8 架构族、1-20B、3-seed
2. 核心证据表齐全：C4_causal_table.json, C1_mechanism_sc_table.json, D3_benefit_predictor_eval.json
3. Neurocomputing 作为机制研究期刊，不要求 prospective deployment

### 如果审稿人要求补充（仅在 revision 时）

**场景 A：补 Mistral-Nemo-12B 额外层（如 L20/L25/L30）**
- **需求**：1x A100-40GB 或 2x A6000-48GB
- **时长**：2-3 小时（每层 ~30-45 分钟 × 3 seeds）
- **成本**：¥30-60 云端 或 本地 GPU 免费

**场景 B：补 Norm-based gain estimator**
- **需求**：纯 CPU，无 GPU
- **时长**：~1 天（统计分析 + 验证）
- **成本**：¥0

**场景 C：（极低概率）审稿人要求 prospective validation**
- **策略**：在 revision response 解释这超出当前资源范围，提供 detailed protocol 供未来验证
- **不需要**：实际跑实验

---

## 八、提交后跟踪

- [ ] 记录 submission ID
- [ ] 记录 submission date
- [ ] 更新项目记忆：D2 转投 Neurocomputing 2026-07-25
- [ ] 等待 editorial decision（预期 ~8 天）

---

**清单创建时间**: 2026-07-25  
**下次更新**: 收到 Neurocomputing 首轮决定后
