# H13 Paper B Prose 检查结果（2026-08-03）

## PB-4: ✅ DONE - Estimand Demotion

**状态**：已完成，陈述清晰

**证据**：
- Abstract 中明确定义了 cross-probe 和 within-probe rank estimands（main.tex:110-123）
- 降级说明明确：预注册的 cross-probe 阈值在 4-bit 全模型上失败于 3 个模型中的 2 个（main.tex:985-1002）
- K1 数值包含：1B = 0.904 PASS, 3B = 0.680 FAIL（阈值 0.85，main.tex:989-990）
- 宏定义在 macros.tex:257-278

## PB-5: ⚠️ TODO - K3 Adjudication

**已完成部分**：
- K3 FAIL 判决诚实陈述（main.tex:906-914）
- 原因说明清晰：仅 0.001-0.004 参数超过 bin width，编辑实际上是 sub-bin-width
- 分母警告和测量轴限定明确（main.tex:916-928）
- 门控表重复 "KILLED (measured axis)"（main.tex:948-963）

**缺失部分**：
- **M-averaging 机制解释缺失**
- 当前 main.tex 定义了 r_func 和 r_param，但明确称其为描述性而非机制声明（main.tex:930-940）
- 替代解释：post-hoc base-noise swamping 作为机制（main.tex:740-780）

**建议**：
- 如果 M-averaging 是预期的机制故事，需要添加
- 或明确退役该解释，使用现有的 base-noise swamping 叙述

## 总结

- PB-4: ✅ 完全完成
- PB-5: ⚠️ 部分完成 - K3 判决正确，但机制解释需要确认/补充
