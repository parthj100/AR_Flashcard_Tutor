# Licenses: datasets, models and software

Checked November 11, 2026. **None of the datasets or models below has a No-Derivatives (ND) term**, so
photographing, hand-copying and degrading problems (our method) is allowed for all of them. Two
restrictions do apply: non-commercial use (NC) and share-alike (SA) on some datasets, and AGPL-3.0 on one
software dependency.

How each license was checked:
- **Repo:** the LICENSE file in the official repository
- **Card:** the license field on the official Hugging Face page
- **Site:** the official download page
- **Paper:** the license statement in the authors' paper
- **pip:** the installed package's metadata

Reference numbers point to [`RELATED_WORK.md`](RELATED_WORK.md).

## Datasets

| Dataset | What we would use | License | ND? | What it allows | Checked by |
|---|---|---|---|---|---|
| **GSM8K** [1] | Main problem source: about 150 word problems | **MIT** | No | Modify, redistribute and use commercially, with the copyright notice kept | Repo ([LICENSE](https://github.com/openai/grade-school-math/blob/master/LICENSE)), card |
| **Our own problems** | Problems written to cover signs, decimals, fractions, exponents and units | Ours to choose; we suggest **CC BY 4.0** | No | — | — |
| **MATH** (Hendrycks et al.) [2] | **Recommended** second, notation-heavy source: full competition problems with fractions, exponents, roots and Greek letters | **MIT** | No | Same as GSM8K | Repo ([LICENSE](https://github.com/hendrycks/math/blob/main/LICENSE)) |
| **CROHME** 2013–2019 [5] | Handwritten single expressions (pen strokes) | **CC BY-NC-SA 3.0** | No | Modify for non-commercial use; anything we share from it must carry the same license | Site ([TC-11](https://tc11.cvc.uab.es/datasets/ICFHR-CROHME-2016_1/gt_1_1)) |
| **MathWriting** [4] | Handwritten and synthetic expressions (pen strokes) | **CC BY-NC-SA 4.0** | No | Same as CROHME | Paper |
| **UniMER-1M / UniMER-Test** [6] | Printed and handwritten expression images | **Apache-2.0** on the card | No | Modify, redistribute, commercial use | Card. The HME100K part is *not* distributed under it; see below |
| **HME100K** [3] | Camera photos of handwritten expressions | **Not confirmed** | Unknown | — | No official license found. A third-party Figshare mirror labels its copy CC BY 4.0, which doesn't bind the original. UniMER's card asks users to get HME100K from its provider "for copyright compliance." **Email the authors before using it** |
| **Aida Calculus** (Kaggle) | Synthetic handwritten calculus expressions | Not named in what we found | Unknown | — | Check the Kaggle page before use |
| **GSM8K-V** [7] | Not planned (GSM8K as visual scenes, not text photos); related work only | MIT (code repo) | No | — | Repo |

**Recommendation for "use multiple datasets":** add a slice of **MATH** next to GSM8K. It's MIT like
GSM8K, and it's full word problems, so it fits our photograph-a-problem setup. It also brings the special
characters our professor is worried about. CROHME and MathWriting are allowed for a course project
(non-commercial), but they're single expressions stored as pen strokes. Rendered strokes look synthetic,
not like photos. Leave HME100K out until its license is confirmed.

**What the NC / SA terms mean for us:** a course research project is non-commercial, so NC is fine. If
we publish any images derived from CROHME or MathWriting (for example in a GitHub dataset release), we
must release them under the same CC BY-NC-SA license and credit the source. Keep GSM8K/MATH-derived
images and CROHME/MathWriting-derived images in separate folders with their own license notes.

## OCR models and their weights

| Component | Role | License | Checked by |
|---|---|---|---|
| **EasyOCR** 1.7.2 | OCR model A | **Apache-2.0** | pip, repo |
| EasyOCR weights (`craft_mlt_25k`, `english_g2`) | Text detection and recognition | Ship with EasyOCR. Not separately checked | — |
| **Pix2Text** 1.1.7 | OCR model B (math-aware) | **MIT** | pip, repo |
| Pix2Text formula detector `pix2text-mfd-1.5` | Finds formulas in the image | **MIT** | Card |
| Pix2Text formula recognizer `pix2text-mfr-1.5` | Formula → LaTeX | **MIT** | Card |
| CnOCR / CnSTD 2.3 / 1.2 | Pix2Text's text reader and detector | **Apache-2.0** | pip |
| PP-OCRv3 English weights (via CnOCR/CnSTD) | Text detection and recognition | **Apache-2.0** | Card |
| **Ultralytics** 8.4 | Loaded by Pix2Text's formula detector | **AGPL-3.0** ⚠️ | pip; Ultralytics' [license page](https://ultralytics.com/license) |

**The AGPL-3.0 issue.** Ultralytics says AGPL-3.0 covers its code *and* models trained with it, and that
running such models through other code forms a combined work under AGPL. Pix2Text's own weights are
labeled MIT, so the two claims conflict. What it means for us:
- Running it for research on our own machines is allowed. Ultralytics lists non-profit and research use
  as not needing a commercial license.
- If we **distribute** the app or **serve it to other users over a network** (for example a hosted
  Lens), AGPL requires releasing that app's full source under AGPL. Our repo is already public, so this is
  manageable, but it rules out a closed-source or commercial Lens without an Ultralytics license.
- Say this on the license slide rather than leaving it out.

## Other software

| Component | License | Checked by |
|---|---|---|
| Phi-4-mini (solver, later stages) | **MIT** | Card |
| PyTorch 2.15 | Permissive mix: BSD-3-Clause, Apache-2.0, MIT and others (no copyleft) | pip |
| ONNX Runtime 1.31 | MIT | pip |
| FastAPI, Uvicorn | MIT, BSD-3-Clause | pip |

## Still to do

1. **HME100K:** email the authors of [3] for the license, or drop it.
2. **Aida:** read the license on the Kaggle page if we want it.
3. **Our own problems and photos:** choose a license (CC BY 4.0 suggested), and get consent from anyone
   whose handwriting is photographed if we ever publish the images.
4. **Our repo:** it has no LICENSE file. Add one (MIT matches most of our dependencies).
