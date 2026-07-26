# Neurocomputing Portal 提交字段 — 复制粘贴指南

**文件位置**: `submissions/d2-neurocomputing/PORTAL-FIELDS/`

所有字段已准备好，按顺序复制到 Neurocomputing Editorial Manager。

---

## 📋 提交流程

### Step 1: Attach Files

- 上传 `main.pdf` (28 pages, 392 KB)
- 上传 `cover-letter.txt` (401 words)
- 可选：上传 `highlights.txt`

### Step 2: General Information

#### 1️⃣ Title

**文件**: `01-title.txt`

```
When Do Rank-One Knowledge Edits Merge? A Gain-Screened Two-Regime Law of Edit Federation
```

#### 2️⃣ Abstract

**文件**: `02-abstract-raw.txt`  
**字数**: 257 words  
**字符数**: 1,833

```
Maintaining a deployed language model's knowledge increasingly relies on rank-one weight edits that correct individual facts without retraining. As this matures, independently authored updates must be federated into one model, and a maintainer must know, before merging, whether they will interfere. We present the first operating map of rank-one edit federation: 22 model-layer cells across 7 architecture families (1-20B), merge group sizes 2-20 (saturation confirmed through 100), three seeds, pre-registered gates. The map shows two regimes. In high-gain regimes, merged edits damage one another at every group size, predicted by key-geometry coherence. In low-gain regimes—deep layers of some architectures—small-group cross-talk is instead constructive: aligned interference raises the members' target logits, then crosses to damage at an architecture-dependent group size. Because edits also install poorly there, the constructive regime marks where not to place edits; its scientific value is the sign reversal itself. One measurable scalar, the layer's perturbation gain, estimated per model from a small calibration merge set on a staging copy, screens the regimes (rank correlation 0.79 with constructive fraction across the 22 cells, p<0.001, 95% CI [0.54, 0.91]; gate, threshold, and per-cell predictions frozen before the runs), is depth-gated within a model, and is neither scale- nor family-specific. The map yields a gain-screened, geometry-ordered admission framework for knowledge maintenance; in retrospective selection evaluations it avoids 2.53 logits of damage per admitted edit (at those edits' own facts) versus random admission at 25% budget in high-gain regimes, beating magnitude-only ordering. The regime structure and screening law hold across seven architecture families from 1B to 20B parameters.
```

#### 3️⃣ Keywords

**文件**: `04-keywords.txt`  
**数量**: 8 个

⚠️ **注意**: Portal 通常要求逐个输入，不要一次性粘贴全部。

```
knowledge editing
model merging
neural network dynamics
rank-one updates
edit federation
perturbation gain
learning systems
constructive interference
```

#### 4️⃣ Section/Category

**文件**: `07-category.txt`

在下拉菜单选择：**Neural Networks**

---

### Step 3: Authors

**文件**: `05-author-info.txt`

- **Name**: Zeyu Fu
- **Email**: fuzeyu09@gmail.com
- **ORCID**: 0009-0001-8329-0108
- **Affiliation**: Army Medical University (Third Military Medical University)
- **City**: Chongqing
- **Country**: China
- **Role**: Corresponding Author

---

### Step 4: Review Preferences

⚠️ **关键选项**：

**Open Access**: 选择 **"No, I do not wish to publish open access"**

（走 Subscription 路径，无需支付 APC ~$2,930）

---

### Step 5: Additional Information

#### 1️⃣ Data Availability Statement

**文件**: `03-data-availability.txt`  
**字符数**: 123（限制 200）

在下拉菜单选择：**"Other (please explain: e.g. 'I have shared the link to my data/code at the Attach File step')."**

然后在文本框粘贴：

```
Data/code at Zenodo: https://doi.org/10.5281/zenodo.21405273 and GitHub: https://github.com/PeterPonyu/edit-federation-map
```

#### 2️⃣ Free Preprint Service (SSRN)

**文件**: `08-ssrn-preprint.txt`

选择：**"NO, I don't want to share my research early and openly as a preprint."**

#### 3️⃣ Funding Information

**文件**: `06-funding.txt`

```
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.
```

---

## ✅ 最终检查清单

在点击 "Submit" 前确认：

- [ ] Title 已填写
- [ ] Abstract 已粘贴（257 words）
- [ ] Keywords 已逐个输入（8 个）
- [ ] Section/Category 选择 **Neural Networks**
- [ ] Author 信息完整（ORCID included）
- [ ] **Open Access 选择 "No"**（关键！）
- [ ] Data Availability 选择 "Other" 并粘贴链接
- [ ] SSRN Preprint 选择 "NO"
- [ ] Funding 声明已填写
- [ ] `main.pdf` 和 `cover-letter.txt` 已上传

---

## 📊 字段统计

| 字段 | 字数/字符数 | 限制 | 状态 |
|------|------------|------|------|
| Title | 90 chars | - | ✅ |
| Abstract | 257 words / 1,833 chars | - | ✅ |
| Keywords | 8 个 | - | ✅ |
| Data Availability | 123 chars | 200 chars | ✅ |
| Funding | 125 chars | - | ✅ |

---

**准备时间**: 2026-07-25  
**下次更新**: 提交后记录 Submission ID
