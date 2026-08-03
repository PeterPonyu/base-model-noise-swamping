# H2/H3/H4 Deletion Phase 最终结果（2026-08-03）

## 执行摘要

- **H2**: ✅ COMPLETE - Phi35 Phase L readout 重新生成完成
- **H3**: ✅ COMPLETE - Qwen15b L21 s1/s2 GATE 文件生成完成
- **H4**: ❌ FAIL - TEXT_PASS gate 失败

## 详细结果

### 状态
- Status: COMPLETE（所有 GATE 文件齐全）
- Missing files: []（无缺失）

### 门控结果
| Gate | Result | 说明 |
|------|--------|------|
| G_D1 | ✅ PASS | Deletion gate decidable |
| G_D2 | ✅ PASS | Variance present |
| **TEXT** | ❌ **FAIL** | **Text baseline increment 不足** |

### 家族详情

| Family | Decidable | Variance | Text Increment |
|--------|-----------|----------|----------------|
| gemma2b | ✅ true | ✅ true | ❌ false |
| phi35 | ✅ true | ✅ true | ❌ false |
| qwen3b | ✅ true | ✅ true | ✅ **true** |
| qwen15b | ✅ true | ✅ true | ❌ false |

**关键发现**：只有 qwen3b 通过 text_increment，其他 3 个家族都失败。

## 影响分析

### 立即影响
- ❌ **H17 Deletion Wave 1 BLOCKED** - 依赖 TEXT_PASS，无法启动
- ⚠️ **Deletion predictor 不超过 text baseline** - 方法论问题或 honest negative

### 科学价值
✅ **这是可发表的 honest negative result**
- 与 PreUnlearn 对比有价值
- 展示了预注册门控的诚实性
- 4 个家族中 3 个失败 → 系统性问题，不是个例

### 决策点

**用户需要决定**：
1. **接受 honest negative** → 撰写对比 PreUnlearn 的 negative result paper
2. **调查失败原因** → 可能的方向：
   - Text baseline 定义是否合理？
   - 是否需要重新审视预注册门控标准？
   - Qwen3b 为何是唯一通过的？（架构特异性？）
3. **终止 deletion predictor 方向** → 将资源转移到其他方向

## 数据完整性

所有必需的 GATE 文件已生成：
```
✅ GATE_gemma2b_L13_s0/1/2.json
✅ GATE_phi35_L16_s0/1/2.json
✅ GATE_qwen3b_L18_s0/1/2.json
✅ GATE_qwen15b_L21_s0/1/2.json
```

## 下一步建议

### 如果接受 negative
1. 检查 deletion_text_baseline.py 实现
2. 运行对比分析：deletion vs text baseline 效果大小
3. 撰写 honest negative 段落

### 如果调查
1. 分析 qwen3b 为何通过（唯一的 true）
2. 检查其他 3 个家族的 text baseline 数值
3. 考虑放宽门控标准（需要 prereg amendment）

### 如果终止
1. 更新 master plan，标记 H17 为 CANCELLED
2. 重新分配预算到其他批次
