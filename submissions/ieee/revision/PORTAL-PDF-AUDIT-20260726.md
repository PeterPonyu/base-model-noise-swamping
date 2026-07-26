# PORTAL PDF AUDIT — which PDF is actually under review at IEEE TETCI?

**Date:** 2026-07-26
**Scope:** identify the manuscript PDF uploaded to the TETCI portal on 2026-07-10, so the
revision round diffs against the right baseline.
**Verdict:** **DETERMINED from disk** (with one harmless residual ambiguity, see §4).

---

## 1. Conclusion

The PDF under review at TETCI is the **2026-07-10 01:16–01:17 build**, whose content is
carried identically by two byte-different files on disk:

- `submissions/ieee/main-as-submitted.pdf` — md5 `0b682102efa4b6c13feed28484c98c4b`
- `submissions/ieee/flat/main.pdf` — md5 `5b87fcc975d018e6b290c92e613f450d`

These two differ **only in typesetting micro-placement** (hierarchical vs flattened
source, 34 s apart); their extracted text is identical apart from horizontal whitespace
runs inside two figure panels. Either one is a faithful representation of the submitted
manuscript.

**`submissions/ieee/main.pdf` (md5 `677546f784bc62964174093cd2147e49`) is NOT the
submitted manuscript.** It is a post-submission revision draft built 2026-07-16 20:50.
Its abstract is fully rewritten (single-lead-claim framing, scope caps hoisted). Do not
use it as the "as submitted" baseline.

**`submissions/ieee/main.submitted-20260716.pdf` is a MISNAMED FILE.** It is byte-identical
to the 07-16 revision draft (`677546f7…`) and was created 2026-07-20 07:05, in the same
second as `d2-federation/main.submitted-20260717.pdf`. It is a batch labelling copy that
picked up whatever `main.pdf` held at the time — which by then was the unsubmitted draft.
The `submitted-20260716` in its name asserts a 07-16 submission event that **did not
happen** (see §3). Recommend renaming it to `main.revision-draft-20260716.pdf`.

---

## 2. Evidence table

| File | md5 (first 8) | Embedded PDF CreationDate | filesystem mtime | Pages | Bytes | Role |
|---|---|---|---|---|---|---|
| `main-as-submitted.pdf` | `0b682102` | **Fri Jul 10 01:16:54 2026 EDT** | Jul 16 20:38 | 14 | 343 186 | **SUBMITTED content** (hierarchical build) |
| `flat/main.pdf` | `5b87fcc9` | **Fri Jul 10 01:17:28 2026 EDT** | Jul 10 01:17 | 14 | 343 340 | **SUBMITTED content** (flat build; the copy inside the portal zip) |
| `main.pdf` | `677546f7` | Thu Jul 16 20:50:16 2026 EDT | Jul 16 20:50 | 14 | 343 333 | post-submission revision draft |
| `main.submitted-20260716.pdf` | `677546f7` | Thu Jul 16 20:50:16 2026 EDT | Jul 20 07:05 | 14 | 343 333 | mislabelled copy of the draft |

Supporting artifacts:

| Artifact | Finding |
|---|---|
| `flat/TETCI_main_manuscript.zip` | 5 entries dated 2026-07-10 01:17: `main.tex` (205 022 B, flattened), `references.bib`, `main.pdf`, `IEEEtran.cls`, `IEEEtran.bst`. The zipped `main.pdf` hashes to `5b87fcc9…` — i.e. **identical to `flat/main.pdf`**. This is the source package prepared for the portal. |
| `submissions/ieee/portal/` | Contains only the metadata deliverables (`title_page.md/.docx`, `cover_letter.md`, `conflict_of_interest.md/.docx`, `CODE-DATA-PLAN.md`), all dated 2026-07-09 23:48–23:58. **No PDF, no SHA256SUMS, no upload manifest.** `CODE-DATA-PLAN.md` records only the code-availability answer ("Yes, I have code associated" — do not upload now). |
| SHA256SUMS files | Exist for `d2-federation/` and `d2-neurocomputing/` only. **No checksum manifest was ever written for `submissions/ieee/`** — the direct provenance record we would have preferred does not exist. |
| `build_revision2.log` | Latexmk rerun triggered by `figures/figA2.tex`, `figures/figE.tex`, `figures/figG.tex` — the 07-16 axis-font bump. Confirms `main.pdf` is a rebuild, not the submission artifact. |
| No `.git` in workspace | No VCS history available to arbitrate; timestamps + embedded PDF metadata are the primary evidence. |

---

## 3. Was anything resubmitted on 2026-07-16?

**No.** Three independent lines agree:

1. **Session memory (decisive).** `~/.claude/projects/…/memory/paper-scores-typeset-referee-20260716.md`
   records, verbatim: *"B6 revision draft APPLIED + engineer-verified in submissions/ieee/
   (as-submitted snapshot `main-as-submitted.pdf` created FIRST; **NOT resubmitted — draft
   only**)."* The snapshot was taken at 20:38 precisely so the 20:50 rebuild would not
   destroy the submitted state.
2. **Content diff.** `main-as-submitted.pdf` → `main.pdf` rewrites the entire abstract
   (old: "We ask when and why the cosine…"; new: "Our central finding is that this damage
   is both predictable and causally removable from geometry alone…"). This is the
   referee-driven "hoist the caveats / single lead claim" edit recommended on 07-16 night —
   a revision-round change, not a submission-day change.
3. **Process.** TETCI (ScholarOne) does not accept a silent file swap on a manuscript that
   is out with reviewers; a 07-16 upload would have required a withdraw-and-resubmit, of
   which there is no trace in `portal/` or `REVISION-PREP-20260710.md`. That doc still
   reads "Status: B6 SUBMITTED, under review as of 2026-07-10", with 07-16 additions filed
   under a section explicitly headed "RESPONSE-POINTS (added 2026-07-16, revision-draft
   pass)".

---

## 4. Residual ambiguity (harmless)

Disk cannot distinguish **which of the two content-identical 07-10 builds** was attached as
the portal's main-document PDF: the hierarchical build (`main-as-submitted.pdf`,
01:16:54) or the flat build (`flat/main.pdf`, 01:17:28, the one inside
`TETCI_main_manuscript.zip`).

Circumstantial weight favours the **flat build**: it is 34 s newer, it was flattened
specifically for portal upload (single self-contained `main.tex`), and it is the copy
bundled into the manuscript zip. But there is no upload log, no portal receipt, and no
checksum manifest to confirm it.

**This does not matter for the revision.** The two builds have identical text, identical
page count (14), identical numbers, identical figures, and identical line/page breaks at
the level a reviewer or an editor's diff would see. Any revision-vs-submission comparison
gives the same answer against either file.

If exact byte-level provenance is ever needed (e.g. an editor asks for a tracked-changes
diff against the exact uploaded file), **the user must check the TETCI/ScholarOne portal
UI** — the submitted-files list shows the uploaded filename and size; match it against
343 186 B (hierarchical) vs 343 340 B (flat).

---

## 5. Recommended actions

| # | Action | Cost |
|---|---|---|
| 1 | Treat `main-as-submitted.pdf` as the canonical review baseline. Freeze it; do not rebuild it. | 0 |
| 2 | Rename `main.submitted-20260716.pdf` → `main.revision-draft-20260716.pdf`. The current name is a factual error that will mislead the next session. | 0 |
| 3 | Write a `SHA256SUMS` for `submissions/ieee/` so this audit never has to be repeated. | 0 |
| 4 | (Optional, user-only) Open the TETCI portal, read the submitted-files list, and record the exact filename + byte size here. Resolves §4 completely. | user, ~2 min |

Note that actions 2 and 3 modify `submissions/ieee/` and are **deliberately not performed
by this audit** — the standing instruction is that nothing in `submissions/ieee/` is
touched outside `revision/` while the paper is under review. They are listed for the user
to authorize.
