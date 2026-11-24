# Related work: is this study new?

Searched November 11, 2026, with web search over arXiv, the ACL Anthology, the CVF open access site and
dataset pages. This was a targeted search, not a systematic review. Before the presentation, repeat the
key queries on Google Scholar and Semantic Scholar (see "Still to check").

## Short answer

**Not entirely new, but it fills a real gap.** No single part of our method is new: OCR for handwritten
math, the effect of OCR errors on downstream tasks, CER missing numeric errors, and energy per image have
each been studied. What we did **not** find is a study that:

1. photographs the **same full word problems** under **three real capture conditions** (printed,
   handwritten, on a screen), with synthetic images as a separate baseline;
2. scores OCR with a **rubric of quantity-changing errors** (number, decimal, sign or operator, fraction
   or exponent, unit or currency) and tests whether **CER understates** them (our H4);
3. reports the **accuracy-energy trade-off** as **joules per transcript with no critical errors**,
   comparing a general OCR model with a math-aware one on the same hardware.

The honest framing: **a new evaluation (benchmark plus metric) of an existing problem, not a new model.**
Lead with the critical-error metric and the energy-per-usable-transcript number, since those are what
existing work lacks. Name the closest work below, so the "new" claim survives the professor's stricter
grading of improvements.

## Update, November 15, 2026: second search

A second round of searches (about a dozen queries; the closest papers were then read) found one paper
that is closer to us than anything above, and narrowed two of our three claims.

**New closest work: MathReal [18].** 2,000 K–12 math questions as **real phone photos** uploaded by
users (textbooks, exam papers, printed exercises), tagged with 14 kinds of real-world noise (blur,
glare, shadow, rotation, handwriting on the page, and so on). It also scores how well each model
transcribes the question, with a metric that includes **exact agreement on numbers and units** as one
of five equal parts (checked in the paper's appendix).

What this changes:

- **"Real photos of math problems" is not new.** MathReal did it at a much larger scale.
- **"Checking numbers separately from character accuracy" is not entirely new either.** Their number
  score is one fifth of a blended score. It isn't reported on its own, doesn't split errors by kind
  (sign, decimal, fraction, unit), and part of the blended score is judged by an LLM.
- **What is still ours:** MathReal tests large vision-language models, not small OCR tools that run
  on a laptop. Its questions are in Chinese. Each problem has one user photo (plus a clean copy for
  some), not the same problem captured under controlled conditions. It has no energy or time numbers.

**Other new work**

- **OCR-Robust [19]:** renders GSM8K problems as images, adds computer-made blur and other damage, and
  measures the final answer of vision-language models. No real photos, no OCR-level scoring, no classic
  OCR tools, no energy. Close to our typeset sharp-vs-blurred benchmark, so cite it there.
- **InduOCRBench [20]:** shows that good OCR scores don't guarantee good results in the next step
  (document question answering), for example 82.9% OCR accuracy against about 53% answer accuracy in
  one category. Supports H4 in another field. No number-error categories.
- **CDM [21]:** a formula-recognition metric built because edit distance is unfair to formulas. It
  checks whether the formula looks right, not whether the quantities are right.
- **Energy:** two more searches found OCR comparisons that report speed only. We still found no paper
  that reports energy per image for OCR tools, so that claim stands for now.
- **Photo-to-solve apps:** no peer-reviewed accuracy study of Photomath or similar turned up.

**The ResearchGate entry is still unresolved.** The page blocks automated reading. Its title and ID
match the well-known survey by Chan and Yeung (2000), which is a literature review and could not have
tested modern OCR systems, so the "12 systems, 1,200 expressions" text is probably a wrong
auto-generated description. **This is a guess; open the page in a browser to confirm.**

**Revised claim:** as far as we can tell, nobody has compared small on-device OCR tools on the same
English word problems across controlled capture conditions, with errors counted by kind and energy
measured. Say "builds on MathReal" rather than "first to use real photos."

## Closest work, and how ours differs

| Work | What it does | How ours differs |
|---|---|---|
| **GSM8K-V** [7] (EMNLP 2026) | Turns GSM8K problems into **multi-image visual scenes** and tests 34 VLMs; best is 59% vs 91% for humans. OCR text alone was a weak baseline | Their quantities are hidden in pictures (count the apples). Ours are **written text** in a photo, and we measure the OCR step itself, not end-to-end solving |
| **PINK / over-correction** [8] (2026) | Shows VLMs silently "fix" students' mistakes when transcribing multi-line handwritten math; proposes the LLM-graded PINK metric instead of BLEU | Same concern (string metrics miss meaning-changing errors), different target: **student solutions** vs **problem statements**. Our rubric is deterministic and quantity-focused, not LLM-graded |
| **OmniHandwritingOCR** [9] (CIKM 2026) | Benchmark of 77.6K handwritten text and math images; 13 OCR/MLLM systems; reports hallucinated corrections | Handwriting only, single images. No printed-vs-screen comparison of the same problem, no energy |
| **FERMAT** [10] (ACL 2025) | Tests whether VLMs can find and fix errors in handwritten math solutions | About grading solutions, not reading problems |
| **Arabic/English/French OCR evaluation** [11] (2026) | Compares cloud and self-hosted OCR, including math expression recognition. Specialist formula OCR drops sharply from printed to handwritten | Supports our H1 and H2. Measures expressions, not word problems; no critical-error rubric, no energy |
| **van Strien et al.** [12] (2020) | OCR errors consistently hurt downstream NLP tasks; finds quality thresholds | The classic citation for "OCR quality affects downstream use." Historical newspapers, not math |
| **Correia & Luck** [13] (2022) | Digitizing balance sheets: high character accuracy still leaves many numbers wrong | Closest argument for our H4 (CER hides numeric errors), in finance rather than math education |
| **Math expression recognition** [3]–[6] | HME100K, CROHME, MathWriting, UniMERNet: datasets and models for single formulas | Formulas in isolation. We use full problems where text and math mix |
| **Energy of vision models** [14] | Per-image energy for edge VLMs: output length, not image size, dominates | Methodology support for joules per image. Not OCR-specific |

**Unverified possible overlap:** a ResearchGate entry titled "Mathematical Expression Recognition: A
Survey" has an abstract describing 12 OCR systems (7 general, 5 math-specific) on 1,200 handwritten,
printed and photographed expressions, scored on more than CER. Its metadata is inconsistent, and we
couldn't open the paper. **Find and read this before claiming novelty.** If real, it's the closest prior
comparison of general vs math OCR across capture conditions. Our differences would still be word problems
rather than expressions, the quantity-error rubric, and energy.

## How this maps to our hypotheses

| Hypothesis | Prior support | What we add |
|---|---|---|
| H1: handwriting is hardest | [9], [11] | Same problems across three real conditions, so condition is the only thing that changes |
| H2: math-aware OCR beats EasyOCR on handwriting | [11] (specialist vs general) | Tested on full word problems, CPU only |
| H3: math-aware OCR costs more time and energy | No direct OCR evidence found | Joules per image and per critical-error-free transcript |
| H4: CER understates the problem | [8], [13] in other domains | First test on math problem statements, with a deterministic rubric checked against hand labels |

## Still to check

1. The ResearchGate "12 OCR systems / 1,200 expressions" study (above).
2. Google Scholar for: "OCR" "math word problems" photographed; "critical errors" OCR numbers;
   "energy" OCR comparison; Pix2Text evaluation.
3. Whether any education-technology paper evaluates photo-to-solve apps (Photomath and similar) on OCR
   accuracy. None turned up here.

## References (IEEE style)

[1] K. Cobbe *et al.*, "Training verifiers to solve math word problems," arXiv:2110.14168, 2021.

[2] D. Hendrycks *et al.*, "Measuring mathematical problem solving with the MATH dataset," in *Proc.
NeurIPS Datasets and Benchmarks Track*, 2021, arXiv:2103.03874.

[3] Y. Yuan *et al.*, "Syntax-aware network for handwritten mathematical expression recognition," in
*Proc. IEEE/CVF CVPR*, 2022, arXiv:2203.01601.

[4] P. Gervais, A. Fadeeva, and A. Maksai, "MathWriting: A dataset for handwritten mathematical
expression recognition," arXiv:2404.10690, 2024.

[5] H. Mouchère, C. Viard-Gaudin, R. Zanibbi, and U. Garain, "ICFHR2016 CROHME: Competition on
recognition of online handwritten mathematical expressions," in *Proc. ICFHR*, 2016. Data: TC-11, CC
BY-NC-SA 3.0.

[6] B. Wang, Z. Gu, G. Liang, C. Xu, B. Zhang, B. Shi, and C. He, "UniMERNet: A universal network for
real-world mathematical expression recognition," arXiv:2404.15254, 2024.

[7] F. Yuan, Y. Yan, Y. Jiang, H. Zhao, T. Feng, J. Chen, Y. Lou, W. Zhang, W. Lu, J. Xiao, Y. Zhuang,
and Y. Shen, "GSM8K-V: Can vision language models solve grade school math word problems in visual
contexts," in *Proc. EMNLP*, 2026, arXiv:2509.25160.

[8] J. Seong, W. Liermann, M. Kim, J. Shin, and S. Lim, "When VLMs 'fix' students: Identifying and
penalizing over-correction in the evaluation of multi-line handwritten math OCR," arXiv:2604.22774,
2026.

[9] Z. Guo, M. Zhang, and B. Jiang, "OmniHandwritingOCR: A diagnostic benchmark for evaluating
multimodal LLMs in handwritten OCR scenarios," in *Proc. CIKM*, 2026, arXiv:2608.18586.

[10] O. Nath, H. Bathina, M. S. U. R. Khan, and M. M. Khapra, "Can vision-language models evaluate
handwritten math?," in *Proc. 63rd Annu. Meeting ACL (Vol. 1: Long Papers)*, 2025, pp. 14784–14814,
doi: 10.18653/v1/2025.acl-long.720.

[11] J. Alsubhi *et al.*, "Evaluating cloud and self-hosted OCR systems for Arabic, English, and French
documents: Bounding box detection, text recognition, and mathematical expression recognition,"
arXiv:2610.25235, 2026.

[12] D. van Strien, K. Beelen, M. C. Ardanuy, K. Hosseini, B. McGillivray, and G. Colavizza, "Assessing
the impact of OCR quality on downstream NLP tasks," in *Proc. ICAART*, 2020, pp. 484–496, doi:
10.5220/0009169004840496.

[13] S. Correia and S. Luck, "Digitizing historical balance sheet data: A practitioner's guide,"
arXiv:2204.00052, 2022.

[14] "Seeing is free, speaking is not: Uncovering the true energy bottleneck in edge VLM inference,"
arXiv:2607.09520, 2026. *(Authors not yet checked.)*

[18] J. Feng, Z. Wang, Z. Zhang, Y. Guo, Z. Zhou, X. Chen, Z. Li, and D. Yin, "MathReal: We keep it
real! A real scene benchmark for evaluating math reasoning in multimodal large language models,"
arXiv:2508.06009, 2025.

[19] Y. Cheng, Y. Wu, and Y. Chang, "How robust is OCR-reasoning? Evaluating OCR-reasoning robustness
of vision-language models under visual perturbations," arXiv:2606.26041, 2026.

[20] L. Sun *et al.*, "When good OCR is not enough: Benchmarking OCR robustness for retrieval-augmented
generation," arXiv:2605.00911, 2026.

[21] B. Wang *et al.*, "Image over text: Transforming formula recognition evaluation with character
detection matching," in *Proc. IEEE/CVF CVPR*, 2025, arXiv:2409.03643. *(Author list not checked.)*

[15] JaidedAI, "EasyOCR," GitHub, 2020. [Online]. Available: https://github.com/JaidedAI/EasyOCR

[16] Breezedeus, "Pix2Text," GitHub, 2022. [Online]. Available: https://github.com/breezedeus/Pix2Text

[17] Microsoft, "Phi-4-mini-instruct," Hugging Face, 2025. [Online]. Available:
https://huggingface.co/microsoft/Phi-4-mini-instruct
