# D2 Companion Paper Citation Strategy (2026-07-24)

> **问题：** D2@KBS如何引用B6@TETCI companion paper？
> **结论：** KBS是SINGLE-anonymized → 可以正常引用，无需匿名化处理。

---

## 背景

**D2 (此文):** Edit federation @ KBS, 提交 2026-07-17
**B6 (companion):** Single-edit damage mechanism @ IEEE TETCI, 提交 2026-07-10

---

## KBS匿名性政策 (已验证)

**Review model:** SINGLE-blind (非双盲)
- 作者身份对审稿人可见
- 审稿人可以查阅作者的其他工作
- → **B6可正常引用**

**来源:** canonical.md line 13-15 (verified 2026-07-16)

---

## 推荐方案

### **方案A: 正常引用 (推荐)**

在References中添加:
```bibtex
@article{fu2026geometry,
  title={When and Why Does Key Geometry Predict Locate-then-Edit 
         Collateral Damage? ...},
  author={Fu, Zeyu},
  journal={IEEE Trans. Emerg. Topics Comput. Intell.},
  note={Under review},
  year={2026}
}
```

在§2 Related Work引用:
```
A companion line~\cite{fu2026geometry} establishes that single-edit 
collateral damage is key-geometry-predictable within a family...
```

### **Contingency Plan: 如果B6被拒**

**选项1:** 更新为arXiv preprint引用
**选项2:** 折叠background到D2 §3 (增加2-3段)

---

## TODO-ANON状态

**canonical.md标记:** `[TODO-ANON]` at line 495
**实际状态:** RESOLVED by KBS single-blind policy
**行动:** 可以从TODO列表移除

---

## 实施检查清单

- [ ] 检查main.tex中companion引用的当前措辞
- [ ] 确认References中的bibtex entry格式
- [ ] 如果B6有更新(接受/拒稿)，相应更新D2引用
- [ ] 从canonical.md TODO列表中移除此项

---

**创建日期:** 2026-07-24
**策略来源:** KBS submission guidelines + canonical.md verification
