# Neurocomputing 转投版本准备完毕（2026-07-25）

## 状态：✅ 本地准备完成，等待用户手动提交

---

## 一、已完成的工作

### 1.1 核心稿件修改（耗时 ~45 分钟，零实验成本）

| 修改项 | 位置 | 状态 |
|-------|------|------|
| 删除 "Prospective deployment is future work" | 摘要 L69 | ✅ 已删除 |
| 改为科学发现收尾 | 摘要 L69 | ✅ 新句："The regime structure and screening law hold across seven architecture families from 1B to 20B parameters." |
| Section 标题降调 | L726 | ✅ "Admission Rule" → "Admission Framework" |
| Contribution 降调 | L145-148 | ✅ "deployable admission rule" → "operational implication for knowledge maintenance" |
| Conclusion 降调 | L876 | ✅ "actionable admission rule" → "operational implication" |
| 摘要用词统一 | L66 | ✅ "admission rule" → "admission framework" |
| Journal 字段 | main.tex | ✅ Knowledge-Based Systems → Neurocomputing |

### 1.2 新文件创建

- ✅ `cover-letter.txt` (401 words) — 机制导向，主打 two-regime law、sign reversal、preregistered map
- ✅ `SUBMISSION-CHECKLIST.md` (173 lines) — 完整提交指南
- ✅ `main.pdf` (28 pages, ~401 KB) — 已编译验证
- ✅ `READY-TO-SUBMIT.md` (本文件) — 最终摘要

### 1.3 保留的 KBS 文件（未修改）

- ✅ `refs.bib` (44 references) — 无需改动
- ✅ `macros.tex` — 数值宏
- ✅ `figA.pdf` 到 `figE.pdf` — 5 个主图
- ✅ `highlights.txt` — 可复用
- ✅ `portal-metadata.txt` — 只需改 journal name

---

## 二、与 KBS 版本的差异

```bash
# SHA256 对比
KBS:           4761d10aae25df0d85b5dd0f0e4797682ac911435e33a91ad10972c66b863658
Neurocomputing: 38c6c7f4a924615e653544c4b3f3e8755a6397b24e6f6dc4d36375f36cff4744
```

**差异内容**：
1. 摘要最后一句（去掉 future work 自我削弱）
2. 5 处 "admission rule" → "admission framework" / "operational implication"
3. Journal 字段
4. Cover letter 完全重写（mechanism-first 定位）

**保持不变**：
- 28 页
- 44 参考文献
- 所有图表
- Limitations 部分（诚实保留 future work 声明，但不在摘要）
- 实验数据和统计量

---

## 三、远端 Box 需求：**当前为零**

### 实验覆盖已充分

| 维度 | 本地验证 | 论文声称 | 状态 |
|------|---------|---------|------|
| 结果文件 | 697 个 | - | ✅ |
| 架构族 | 8 families | 7 families | ✅ **超出** |
| 规模 | 1-20B | 1-20B | ✅ |
| 种子 | s0=202, s1=139, s2=127 | 3 seeds | ✅ |
| 核心表 | C4_causal, C1_mechanism, D3_benefit | - | ✅ |
| 编辑器 | ROME, MEMIT, AlphaEdit, FT, GRACE | 5 editors | ✅ |
| 数据集 | CounterFact, zsRE, MQuAKE, RippleEdits | 4 datasets | ✅ |

### 如果审稿人要求补充（revision 时）

**场景 A：Mistral-Nemo-12B 额外层**
- GPU: 1x A100-40GB 或 2x A6000-48GB
- 时长: 2-3 小时
- 成本: ¥30-60 云端 或 本地免费
- 概率: **中等**（如果审稿人质疑 12B 证据）

**场景 B：Norm-based gain estimator**
- CPU only
- 时长: ~1 天
- 成本: ¥0
- 概率: **低**（L864 future work，非 critical gap）

**场景 C：Prospective validation**
- 策略: Revision response 解释超出资源范围
- 不需要: 实际跑实验
- 概率: **极低**（机制研究不要求）

---

## 四、下一步行动（用户手动）

### 4.1 Portal 提交步骤

1. **进入 Neurocomputing Editorial Manager**
   - URL: https://www.editorialmanager.com/neucom/
   - 或从 KBS rejection email 点击 "Transfer"（但不直接 final submit）

2. **上传文件**
   - Main manuscript: `submissions/d2-neurocomputing/main.pdf`
   - Cover letter: `submissions/d2-neurocomputing/cover-letter.txt`
   - Highlights (可选): `submissions/d2-neurocomputing/highlights.txt`

3. **关键选项：Open Access**
   - ⚠️ **选择 "No, I do not wish to publish open access"**
   - 这会走 Subscription 路径，**无需支付 APC**
   - Neurocomputing 支持 hybrid 订阅制免费发表

4. **Metadata**
   - Title: 从 main.pdf 复制
   - Abstract: 从 main.pdf 复制（**已去掉 "Prospective deployment" 那句**）
   - Keywords: knowledge editing, model merging, neural network dynamics, rank-one updates, edit federation, perturbation gain, learning systems, constructive interference
   - Author: Zeyu Fu, ORCID 0009-0001-8329-0108
   - Affiliation: Army Medical University, Chongqing, China

5. **Data Availability**
   ```
   The code, edit vectors, and result artifacts supporting the findings of this study are openly available at Zenodo (https://doi.org/10.5281/zenodo.21405273) and GitHub (https://github.com/PeterPonyu/edit-federation-map).
   ```

6. **Declarations**
   - Funding: None
   - Competing interests: None
   - Generative AI: Already in manuscript

### 4.2 提交后跟踪

```bash
# 记录 submission ID 和日期
echo "Submission ID: [从 portal 获取]" >> submissions/d2-neurocomputing/SUBMISSION-LOG.txt
echo "Date: $(date +%Y-%m-%d)" >> submissions/d2-neurocomputing/SUBMISSION-LOG.txt
```

### 4.3 更新项目记忆

```bash
# 在下次 Claude 会话中告知：
# "D2 已于 2026-07-25 转投 Neurocomputing，submission ID [xxx]"
```

---

## 五、预期时间线

- **Submission to first decision**: ~8 days (Neurocomputing 官方中位数)
- **如果 accept**: 另需 ~8 days 到发表
- **如果 minor/major revision**: 取决于补充工作量
  - Minor（文字修改）：1-2 周
  - Major（补 Mistral-Nemo）：2-3 周

---

## 六、禁止事项

- ❌ **不要**上传 `main.submitted-20260717.pdf`（KBS 冻结版）
- ❌ **不要**选择 Open Access（会收 ~$2,930 APC）
- ❌ **不要**使用 KBS 的旧 cover letter
- ❌ **不要**在初投前补 Mistral-Nemo 实验（等审稿人要求）
- ❌ **不要**在 Portal 再写 "Prospective deployment is future work"

---

## 七、关键文件清单

### 必须上传
- `main.pdf` (28 pages, 401 KB) ✅
- `cover-letter.txt` (401 words) ✅

### 可选上传
- `highlights.txt` ✅

### Portal 手动填写
- Title, Abstract, Keywords, Author info, Data availability ✅ (从本文档复制)

### 保留在本地（revision 时提供）
- `main.tex`
- `refs.bib`
- `macros.tex`
- `figA.pdf` - `figE.pdf`

---

## 八、成本总结

| 项目 | 成本 |
|------|------|
| 稿件修改 | ¥0（~45 分钟人工） |
| PDF 编译 | ¥0 |
| Cover letter 撰写 | ¥0（~30 分钟人工） |
| Submission（订阅制） | ¥0 |
| **总计** | **¥0** |

**如果审稿人要求补充**：
- Mistral-Nemo 12B: ¥30-60 云端 或 本地免费
- Norm estimator: ¥0（纯 CPU）

---

## 九、成功标志

✅ 所有必改项已完成  
✅ PDF 编译成功（28 页）  
✅ Cover letter 机制导向  
✅ Subscription 路径明确  
✅ 实验覆盖充分（无需补充）  
✅ KBS 冻结版未被覆盖  
✅ 提交检查清单完整  

---

**状态**: 🟢 **READY TO SUBMIT**  
**准备时间**: 2026-07-25 10:11-10:13 (本地 ~2 小时，实际操作 ~45 分钟)  
**下次行动**: 用户登录 Neurocomputing Editorial Manager 并按 Section IV 步骤提交
