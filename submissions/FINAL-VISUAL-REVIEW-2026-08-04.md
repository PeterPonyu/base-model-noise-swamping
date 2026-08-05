# Final Visual Review Index — 2026-08-04

This index is for visual inspection only. Every review PDF must display
`HONEST-STATE REVIEW DRAFT — NOT SUBMISSION CANDIDATE` on page 1. Frozen or
previously submitted artifacts remain separate and are not replaced by these files.

## Package status

| Package | Review state | Review PDF | Full-page QA | In-manuscript figure QA |
|---|---|---|---|---|
| D2 / Neurocomputing | READY FOR VISUAL REVIEW; prospective result remains mixed (P3 passes 1/3) | `d2-neurocomputing/main-honest-review.pdf` | `d2-neurocomputing/figures-qa/contact-sheet.png` (35 pages) | `d2-neurocomputing/figures-qa/figures-contact-sheet.png` (5 composite figures) |
| B6 / IEEE TETCI | READY FOR VISUAL REVIEW; revision review only, submitted artifact frozen | `ieee/flat/main-honest-review.pdf` | `ieee/figures-qa/contact-sheet.png` (14 pages) | `ieee/figures-qa/figures-contact-sheet.png` (7 figures) |
| Frame-A / ESWA | READY FOR VISUAL REVIEW; preregistered router gate is KILL (`0.103 < 0.5`) | `frame-a-eswa/main-honest-review.pdf` | `frame-a-eswa/figures-qa/contact-sheet.png` (18 pages) | `frame-a-eswa/figures-qa/figures-contact-sheet.png` (10 figures) |
| Paper B / Neurocomputing | READY FOR VISUAL REVIEW; H11 complete (9/9 cells), G-S3 PASS ($\rho=-0.900$) | `paper-b-neurocomputing/main-honest-review.pdf` | `paper-b-neurocomputing/figures-qa/contact-sheet.png` (29 pages) | `paper-b-neurocomputing/figures-qa/figures-contact-sheet.png` (5 manuscript figures) |

## D2 / Neurocomputing

- Scientific status: honest-state review draft; not a submission candidate.
- Boundary: P1/P2/P4 pass; P3 passes 1/3 seeds; full confirmation is blocked.
- PDF manifest: `d2-neurocomputing/figures-qa/manifest.json`
- Figure manifest: `d2-neurocomputing/figures-qa/figures-manifest.json`
- Review PDF SHA256: `bd5e2f57382b49953e6a6fb76fe4c7a20e31d63fae17c03fa53b649db8608a1a`.
- Full-page contact sheet SHA256: `3062ab55ec5e391d9e35b3eb55d6c1f9d669ec44a1f18ce8026a3922aab5451a`.
- In-manuscript figure contact sheet SHA256: `c6f7b475bc6db16361ba8a421de4e3e60b4cc8f58da9354f0fc1e6ba4b833536`.
- Frozen `main.pdf` SHA256: `5c48fa92ec69138da61f29e16ffde68c48bfe375d06b1c4194a8ca61703b9a18`.
- Automated checks: fonts embedded; text/path leak scan clean; 0 LaTeX errors;
  0 undefined references; one 2.61108 pt overfull warning retained for visual review.

## B6 / IEEE TETCI

- Scientific status: honest-state revision review draft; not a submission candidate.
- Boundary: SxC is an exact rank-one reduction, not a faithful true-influence rank
  surrogate; signed law is Llama-family scoped; strong causal law is not claimed at scale.
- PDF manifest: `ieee/figures-qa/manifest.json`
- Figure manifest: `ieee/figures-qa/figures-manifest.json`
- Review PDF SHA256: `0858794f832877800ad93d2ed194058a10912919743c2be0f291fb6dc1498d73`.
- Full-page contact sheet SHA256: `626e57aeaa6c9ef6ca10c1a09f97d482df3dae3facadbfaf9af7bbedfedf15b9`.
- In-manuscript figure contact sheet SHA256: `2e53354e956ef9be02d93349a67fcf409659ebeb25e1d2a4e0e284075854b6fd`.
- Automated checks: fonts embedded; text/path leak scan clean; 0 LaTeX errors;
  0 undefined references; seven 0.94–1.45 pt figure overfull warnings retained for visual review.
- Frozen `main-as-submitted.pdf` SHA256:
  `9fe0eb55adad0bf935db54188ddc8a84440f6df8482de3f3259212830bff5145`.
- Frozen `flat/TETCI_main_manuscript.zip` SHA256:
  `cad4851f6b792ada599e7c6a38c309ebe0d754763f87e8ac5ce4430c33f9f0c8`.

## Frame-A / ESWA

- Scientific status: honest-state review draft; not a submission candidate.
- Boundary: the preregistered router gate is **KILL**; T4 is `0.103 < 0.5`.
  The review copy keeps the measured KILL outcome and does not launder it into PASS.
- PDF manifest: `frame-a-eswa/figures-qa/manifest.json`
- Figure manifest: `frame-a-eswa/figures-qa/figures-manifest.json`
- Review PDF SHA256: `538bb8652ade48db4f9822b9751df366c81e8aef29fb1415dc9f85e11790984f`.
- Full-page contact sheet SHA256: `28c6d2f9de7f99b5f4f9146f17e48dc4e0067e65d8eb73c359cd34ce9fbc5f3e`.
- In-manuscript figure contact sheet SHA256: `0997c5ed48a13c60920b41917502361d50d417cd42c0db704751e6aff1bf2c6e`.
- Frozen `main.pdf` SHA256: `a16063536e1a318aee25104c7c6526e2201d1d369a1316cfc9b90c3bbefa673b`.
- Automated checks: 18 nonblank pages, 10 nonblank standalone figures, all fonts embedded,
  0 LaTeX errors, 0 undefined references, 0 overfull boxes, and 0 missing graphics for
  the selected honest-review source. The generic text scan reports two uses of
  “placeholder” in explicit negations (“not a placeholder” and “synthetic-cost
  placeholders”), not unfinished artifacts.

## Paper B / Neurocomputing

- Scientific status: honest-state review draft; not a submission candidate.
- Boundary: H11 replication grid complete (gemma2b L19, qwen3b L27, phi35 L24 × 3 seeds);
  G-S3 PASS with $\rho=-0.900$ (threshold $-0.3$) on the 9-cell NEW grid.
  Phi-3.5 s2 was recovered from box 36039 (out_dir misplacement fixed, pair validated).
  Llama-3.2-3B L24 s1 ran as a parallel supplementary cell (table only, npz overwritten
  by phi35 s2 sharing the same out_dir — known limitation, not part of the 9-cell gate).
- PDF manifest: `paper-b-neurocomputing/figures-qa/manifest.json`
- Figure manifest: `paper-b-neurocomputing/figures-qa/figures-manifest.json`
- Review PDF SHA256: `a29de6a66ee03ac1fe78ad5c71f6cd012344a9b4ac2152536123a33008a52f1e`.
- Full-page contact sheet SHA256: `017d87187964bb0a371533fc5571e4afa710193d4ef01654bf5740e5033e4e56`.
- In-manuscript figure contact sheet SHA256: `e6c61e0d40e945bc8061356c698e6670885e56dfc78d89c6c2b18aad2a29b1f8`.
- Frozen `main.pdf` SHA256: `b79bd033e1da5d1b05df845f9ed014b4aaaed94d5ce8a9f47786753c13c0299a`.
- Automated checks: 29 pages, 0 LaTeX errors, 0 undefined references, 0 overfull boxes.
  figF3 (noise-to-signal vs rank survival, 18 cells, 4 families) inserted after
  Table 2 in §6.2; G-S3 PASS annotation rendered in-figure.


   clipping, and legibility.
2. Open each full-page `contact-sheet.png` and inspect float placement, blank space,
   headings, tables, and banner placement.
3. Use the per-page PNGs in the same `figures-qa/` directory for any page that needs
   full-resolution inspection.
4. Read `manifest.md` for the build-warning counts and `manifest.json` for PDF SHA256,
   exact page list, font status, and leak-scan result.
