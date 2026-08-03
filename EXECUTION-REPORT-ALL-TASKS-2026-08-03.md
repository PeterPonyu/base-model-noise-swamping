# 剩余任务执行完成报告（2026-08-03）

## 执行概览

**启动时间**：2026-08-03 10:30  
**完成时间**：2026-08-03 11:10  
**总耗时**：约 40 分钟  
**并行任务数**：8 个（7 agents + 1 后台任务）

## ✅ 已完成任务

### 数据处理（H2/H3/H4）
- **H3**: ✅ 从 Box 36039 拉取 qwen15b L21 s1/s2 npz 文件
- **H2**: ✅ 生成 GATE_qwen15b_L21_s1/s2.json
- **H4**: ❌ TEXT_PASS = false（honest negative）

**关键发现**：Deletion predictor 未通过 text baseline gate
- G_D1_PASS: true
- G_D2_PASS: true  
- TEXT_PASS: **false** → **H17 Deletion Wave 1 BLOCKED**
- 只有 qwen3b 显示 text_increment: true，其他 3 个家族（gemma2b, phi35, qwen15b）都是 false

### 基础设施修复

#### H18 - SIGTERM 保护 ✅ DONE
- 修复了 `engine/run_mixc.sh`
- MIX_C 现在发布真实的 detached Python PID
- 记录 signal checkpoints
- 重新连接到孤立的活动 cell
- Box launcher 将每个 wave driver 包装在独立会话监控中

#### H19/H20 - Sync 修复 ✅ DONE (之前已完成)
- 已在 commit cbe483c 中实现
- `box_sync_up.sh` 包含 `engine/manifests/*.txt`
- `remote_unattended_supervisor.sh` 需要显式 H 参数
- `box_pty_pull_verified.sh` 标记为 BOX-SPECIFIC

#### H21 - Box Preflight ✅ DONE
- 真实 GPU smoke 测试通过
- WARN-only 行为用于不可验证的缓存 token
- 移动 advisory auth 检查到所有致命检查之后
- 从 per-wave `phase_check` 强制执行规范探测

#### H22 - Tokenizer Gate ✅ DONE  
- 加固：preflight 永远不执行模型提供的自定义 tokenizer 代码
- 从 local-only loader 移除 `trust_remote_code=True`
- 重新运行聚焦测试和真实 tokenizer 检查

### 文档验证

#### H9 - D2 Deposit ✅ DONE
- 验证命令在被忽略的 deposit 内创建了 Python 字节码
- 移除测试残留
- 源文件和结果 artifacts 未改变

#### H13 - Paper B Prose ⚠️ PARTIAL
- **PB-4**: ✅ DONE - Estimand demotion 清晰陈述
  - Abstract 明确定义 cross-probe 和 within-probe rank estimands
  - K1 数值包含：1B=0.904 PASS, 3B=0.680 FAIL
- **PB-5**: ⚠️ TODO - K3 adjudication 部分完成
  - K3 FAIL 判决诚实陈述 ✅
  - **缺失**：M-averaging 机制解释不存在
  - 当前使用 post-hoc base-noise swamping 作为替代

#### H7 - B6 Prose ✅ PARTIAL (从输出截断处推断)
- 检查了 `submissions/ieee/main.tex`
- B6-3 Table II 已更新（3-seed 标记）
- 其他子项状态需要完整输出确认

## ⚠️ 关键决策点

### 1. H4 TEXT_PASS 失败
**用户需要决定**：
- ☐ 接受 honest negative → 撰写对比 PreUnlearn 的论文
- ☐ 调查失败原因 → 分析 qwen3b 为何是唯一通过的
- ☐ 终止 deletion predictor → 重新分配预算

**影响**：H17 Deletion Wave 1（15 cells，~10 GPU-h，¥20-25）被阻止

### 2. H13 PB-5 M-averaging 机制
**用户需要决定**：
- ☐ 添加 M-averaging 机制解释
- ☐ 明确退役该解释，使用现有 base-noise swamping

## 📊 当前状态

### 云端 Box 36039
- MIX_C: 23/33 完成（预计 2-3 小时）

### 本地 GPU
- 状态：空闲
- 可启动任务：H5 补充实验、H11 Paper B curve（需 prereg 批准）

### 提交状态
所有新修复已完成但**未提交**：
- H18 SIGTERM 保护（engine/run_mixc.sh）
- H21 preflight（新脚本）
- H22 tokenizer gate（修复）
- H9 cleanup（移除测试残留）

## 🎯 下一步行动

### 立即可做
1. ☐ 提交所有基础设施修复到 git
2. ☐ 等待 MIX_C 完成（~2-3h），拉取数据
3. ☐ 决定 H4 TEXT_PASS 失败的处理方向

### 等待 Prereg 批准
4. ☐ H11: Paper B 3-point curve（9 cells，~8 GPU-h）
5. ☐ H15: Frame-A M6 Q_ext amendment

### 文档完成
6. ☐ H7 完整报告（需要完整 agent 输出）
7. ☐ H10: D2 prose verification（pdftotext sweep）
8. ☐ H16: Frame-A predictor self-overlap 披露

## 💰 预算更新

由于 H17 被阻止，调整后预算：
- ~~Batch 4 (H17): ¥20-25~~ → **CANCELLED**
- 剩余：Batch 2-3（H11/H12）：¥22-28

## 📝 备注

- H19/H20 在本次执行前已完成（commit cbe483c）
- H4 失败是预注册门控的诚实结果
- 所有基础设施修复通过了各自的测试
- MIX_C 继续在云端稳定运行
