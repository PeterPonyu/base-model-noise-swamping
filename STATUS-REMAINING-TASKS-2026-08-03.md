# 剩余任务状态 — 2026-08-03 上午检查

基于 2026-07-31 Master Gap-Closure Plan 的执行状态。

## ✅ 已完成的 GPU 实验（本地）

| ID | 任务 | 状态 | 证据 |
|----|------|------|------|
| H1 | B6 Phi refix (s2 + 3 deletion) | ✅ DONE | gate_phi35 + u1e0_phi35 all seeds committed |
| H3 | Qwen deletion L21 s1/s2 | ⚠️ PARTIAL | JSON 存在但 **NPZ 矩阵文件缺失** → GATE 无法生成 |
| H5 | B6 random-direction control | ✅ DONE | 12 arand cells (L8/10/12/14 × s0/1/2) committed |
| H6 | B6 alphaHO holdout | ✅ DONE | 6 g4_alphaHO cells (L10/14 × s0/1/2) committed |
| - | MIX_A/B refill | ✅ DONE | 33+33 cells complete |

## 🟢 进行中（云端 Box 36039）

| ID | 任务 | 状态 |
|----|------|------|
| H14 | Frame-A MIX_C | 🟢 22/33 完成（预计数小时内完成）|

## ⚠️ 发现的问题

### P1: H3 不完整
- **问题**：u1e0_qwen15b_delete_refusal_L21_s1/s2.json 存在，但对应的 npz 矩阵文件不存在
- **影响**：无法生成 GATE_qwen15b_L21_s1/s2.json → H2/H4 门控不完整
- **需要**：重新运行 qwen15b L21 s1/s2 实验（~1-1.5 GPU-h）OR 检查 Box 36039 是否有这些 npz

### P2: H2/H4 门控状态
- **当前**：deletion_phase_readout 运行结果：
  - G_D1_PASS: true
  - G_D2_PASS: true
  - **TEXT_PASS: false** ← H4 FAIL
- **依赖**：需要 H3 的 qwen15b s1/s2 GATE 文件才能完整评估

### P3: H8 D2 Prospective 部分完成
- **当前**：mistral7b L24 × 3 seeds 已完成
- **Prereg 要求**：需要检查是否还需要其他模型/配置

## 📋 可立即执行的任务（本地 ¥0）

### 基础设施修复（H18-H22）
- [ ] H18: setsid/trap 防止 SIGTERM 杀死 pipeline
- [ ] H19: manifest sync 修复（engine/*.sh + manifests/*.txt）
- [ ] H20: supervisor 脚本修复（显式传递 H=）
- [ ] H21: box preflight probe 脚本
- [ ] H22: tokenizer gate 集成到 phase_check

### 文档/验证任务
- [ ] H7: B6 prose holes（Table II fold, 五个小项）
- [ ] H9: D2 deposit artifacts 验证（RG_matched_dose_spread + perm-null Phi）
- [ ] H10: D2 prose verification（pdftotext sweep）
- [ ] H13: Paper B prose（estimand, K3 adjudication）
- [ ] H15: Frame-A Q_ext 代码（等 M6 prereg 批准）
- [ ] H16: Frame-A predictor self-overlap 披露

## 🎯 需要 GPU 的任务

### 立即可启动（如果 GPU 空闲）
- [ ] H3 补完：qwen15b L21 s1/s2 重跑（~1-1.5 GPU-h）
  - 或检查 Box 36039 是否有 npz 文件可以拉取

### 等待 H3/H4 结果
- [ ] H17: Deletion wave 1（15 cells，依赖 H4 TEXT_PASS）

### 等待 MIX_C 完成
- [ ] Frame-A gate v2 验证
- [ ] H15 Q_ext 分析执行

### 需要 Prereg 批准
- [ ] H11: Paper B 3-point curve（9 cells，~8 GPU-h）
- [ ] H12: Paper B 8B anchor（依赖 H11 通过）

## 💰 云端批次估算

| Batch | 内容 | GPU-h | 预估成本 |
|-------|------|-------|----------|
| 1 (部分) | H3 补完 | ~1.5 | ¥4-6 |
| 2 | H11 PB-CURVE | ~8 | ¥16-20 |
| 3 | H12 PB-B4 (条件) | ~3 | ¥6-8 |
| 4 | H17 DEL-W1 (条件) | ~10 | ¥20-25 |

总计（如果全部执行）：¥46-59

## 🚦 下一步建议

### 立即行动
1. **等待 MIX_C 完成**（数小时内），拉取数据
2. **检查 Box 36039**：qwen15b L21 s1/s2 npz 是否在云端
3. **启动基础设施修复**（H18-H22，本地，可并行）

### 短期（今日）
4. **补完 H3**：
   - 如果云端有 npz → 拉取 → 生成 GATE → 重跑 H2/H4
   - 如果没有 → 本地重跑（GPU 空闲时）
5. **文档任务**：H7/H9/H10/H13（可并行）

### 用户决策点
- [ ] 批准 4 个 DRAFT preregs（D2-PROSP, PAPERB-CURVE, B6-RANDOM, FRAME-A-M6）
- [ ] H4 TEXT_PASS 如果失败：是否接受 honest negative（PreUnlearn 对比）
- [ ] 启动剩余 GPU batches 的 go/no-go

## 备注

- 本地 GPU 当前**空闲**（利用率 2%）
- Box 36039 双卡运行中（MIX_C）
- 所有新数据已提交（commit cad07a3，91 files）
