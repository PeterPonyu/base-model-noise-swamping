# CAMERA-READY-ONLY content — DO NOT include in any uploaded review source/PDF

This file holds everything that must be pasted into `main.tex` ONLY AFTER
acceptance (IEEE TETCI review is double-anonymous; the uploaded source must
stay fully anonymous, with no de-anonymizing content even behind toggles or
in comments). Real author details filled 2026-07-10; they match the
submitted `portal/title_page.docx`.

## 1. Author block (replaces `\author{Anonymous Author(s)}`)

```latex
\author{Zeyu~Fu
\thanks{Z.~Fu is with the State Key Laboratory of Trauma and Chemical
  Poisoning, Institute of Combined Injury, Chongqing Engineering Research
  Center for Nanomedicine, College of Preventive Medicine, Army Medical
  University, Chongqing, China (e-mail: fuzeyu09@gmail.com).}%
}
```

Also restore the surname in the right running head of `\markboth`
("Anonymous \MakeLowercase{\textit{et al.}}" → "Fu: When and Why Does ..."
— sole author, so no "et al.").

## 2. REQUIRED AI-generated-text disclosure (IEEE policy — binding)

Per the IEEE policy acknowledged at submission, AI-generated text must be
disclosed in the Acknowledgments with a citation to the AI system. This
wording matches the title-page acknowledgments submitted for review. Add
before the references:

```latex
\section*{Acknowledgments}
The author acknowledges the use of Claude (Anthropic)~\cite{claude} for
assistance with manuscript drafting, data visualization, and analysis-code
development, under the author's direction; the author designed the
experiments, verified all reported results against the underlying data, and
takes full responsibility for the content.
```

with a `references.bib` entry for the AI system (fill the model/version used):

```bibtex
@misc{claude,
  title        = {Claude (large language model)},
  author       = {{Anthropic}},
  year         = {2026},
  howpublished = {\url{https://www.anthropic.com}}
}
```

Funding acknowledgments (if any) also go here at camera-ready — never in the
review version.

## 3. Dual-submission history note (for the record, not for the paper)

The conference draft this manuscript grew out of was NEVER submitted anywhere
(the planned ARR cycle was forfeited for a journal-first strategy), so there
is no prior/simultaneous submission to disclose and this is a standalone
first submission. If the conference version is ever actually submitted or
published elsewhere, a dual-submission/extension-disclosure `\thanks` note
must be added with the real citation and venue.

## 4. Page-charge facts (TAI, verified 2026-07-10)

Hybrid journal: subscription track has NO APC ($2{,}800$ only if OA is
chosen); regular papers get 10 pages free with a MANDATORY \$200/page
overlength charge at acceptance (14 pp ⇒ ~\$800, either track).
