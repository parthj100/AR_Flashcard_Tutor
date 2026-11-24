# Dataset licenses

Every item in this test set is under a permissive license, or was made by us.

| What | Dataset / source | License | Copyright | Where it's used |
|---|---|---|---|---|
| 20 word problems | **GSM8K** (test split), [openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT | (c) 2021 OpenAI | `problems.csv` rows `gsm8k_*` |
| 10 math-symbol problems | **MATH** (Hendrycks et al.), [hendrycks/math](https://github.com/hendrycks/math) | MIT | (c) 2021 Dan Hendrycks | `problems.csv` rows `math_*` |
| ↳ taken from the 500-problem test split | **MATH-500**, [openai/prm800k](https://github.com/openai/prm800k) (`prm800k/math_splits/test.jsonl`) | MIT | (c) 2023 OpenAI | source file of the `math_*` rows |
| Photos | Taken by the project team (Alexis Juarez Gomez, Parthkumar Joshi) | Our own work | The team | `photos/` |
| Rendered images | Typeset by us from the problem text (KaTeX + headless Chrome screenshot, plus a Gaussian blur copy). **No AI image generation.** | Our own work | The team | `photos/rendered/`, `photos/rendered_blur/` |
| Math typesetting on the photo sheet | **KaTeX** 0.16.11, [KaTeX/KaTeX](https://github.com/KaTeX/KaTeX) | MIT | (c) 2013-2020 Khan Academy and other contributors | `sheets.html` |

Source files were checked on 2026-11-15. `sources/math500_test.jsonl` has SHA-256
`35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`, which is the hash in the
PRM800K repository's Git LFS pointer.

None of these licenses has a "No Derivatives" or "Non-Commercial" clause. Photographing,
transcribing and redistributing the problems is allowed, as long as the copyright notices and
the permission notice below stay with them.

We do **not** use the Hugging Face mirror `HuggingFaceH4/MATH-500`. Its content is identical,
but its dataset card states no license, so we download from the MIT-licensed PRM800K original.

---

## MIT License (identical text for GSM8K, MATH, PRM800K and KaTeX)

Copyright (c) 2021 OpenAI (GSM8K)
Copyright (c) 2021 Dan Hendrycks (MATH)
Copyright (c) 2023 OpenAI (PRM800K / MATH-500 split)
Copyright (c) 2013-2020 Khan Academy and other contributors (KaTeX)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
