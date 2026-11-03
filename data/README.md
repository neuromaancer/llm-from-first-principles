# Teaching data

Small data should be easy to inspect, share, and reproduce. The JSONL files here
are literal exports of the lesson examples, not a new benchmark or additional
training data. Existing notebooks keep their original inline definitions.

## Included in Git

| File | Rows | Origin and purpose |
|---|---:|---|
| [SFT training](toy/sft_train.jsonl) | 4 | Lesson 22 `training_pairs`: hello, farewell, thanks, yes. |
| [SFT held-out probes](toy/sft_heldout.jsonl) | 3 | Lesson 22 `evaluation_cases`: welcome, no, please. |
| [Preference training](toy/preferences_train.jsonl) | 3 | Lesson 23 `preferences`; reused by Lesson 25 and the small integration demo. |
| [Preference held-out probes](toy/preferences_heldout.jsonl) | 3 | Lesson 23 `heldout_preferences`; reused by Lesson 25 evaluation. |

The exports were checked against the source notebooks at
[commit d9154aa](https://github.com/neuromaancer/llm-from-first-principles/tree/d9154aac4da7b6bc61875c4122717d69dbb09d07):
[Lesson 22](../lessons/22_supervised_fine_tuning.ipynb), code cells 26 and 40,
and [Lesson 23](../lessons/23_preference_data_and_reward_modeling.ipynb), code
cells 1 and 35 (zero-based notebook cell indices). Values, labels, and row order
are preserved. The integration demo's three inline `PAIRS` match
`preferences_train.jsonl` exactly; the demo remains independently runnable.

### Schemas and serialization

Each UTF-8 line is one JSON object:

```json
{"prompt": "Say hello", "response": "Hello"}
{"prompt": "Say yes", "chosen": "Yes", "rejected": "No"}
```

The first schema is for SFT; the second is for preferences. Stored response
strings do **not** contain the terminating newline. The lesson serialization is:

```python
prefix = f"User: {row['prompt']}\nAssistant: "
completion = row["response"] + "\n"  # Or chosen / rejected for a preference pair.
```

Only completion targets contribute to supervision/scoring. The newline is an
ordinary character in the character tokenizer, not a dedicated EOS token.

Loading a file requires no additional dependency:

```python
import json
from pathlib import Path

# From the repository root; notebooks in lessons/ use ../data/ instead.
pairs = [
    json.loads(line)
    for line in Path("data/toy/preferences_train.jsonl").read_text().splitlines()
    if line.strip()
]
```

### What held-out means

Splits are **lesson-specific**. For example, `Say no` is held out in Lesson 22
but becomes a training preference in Lessons 23/25. `Say welcome` is held out
from preference training but appears in the PPO experiment. These tiny synthetic
labels were written for teaching, not collected from human raters. They are not
an untouched generalization benchmark, and training accuracy is not a claim
about real instruction-following ability.

## Tiny Shakespeare: provenance matters

Lessons 16 and 22 read `data/raw/tinyshakespeare.txt` (using a `../data/` prefix
from `lessons/`). The original ignored file was not accessible in the repository.
The saved Lesson 16 output reports **1,115,390 characters** and a vocabulary of
65 characters. The canonical upstream snapshot below has **1,115,394 characters**
and the same vocabulary size. A matching prefix and vocabulary do not prove
byte-for-byte identity. No four characters have been silently removed to make
the counts agree.

The raw corpus is **not bundled in this commit**. A reproducible public reference
is available through [prepare_tinyshakespeare.py](prepare_tinyshakespeare.py):

```bash
python data/prepare_tinyshakespeare.py
```

The script uses only the standard library, verifies the size and SHA-256, and
writes to the notebook's expected path. It verifies an already matching file
without downloading again and **refuses to overwrite a different local file**.
A separate destination can be passed with `--output`.

| Field | Pinned upstream reference |
|---|---|
| Source repository | [karpathy/char-rnn](https://github.com/karpathy/char-rnn) |
| Source path | [data/tinyshakespeare/input.txt](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt) |
| Source commit | `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e` |
| UTF-8 bytes / characters | `1,115,394` / `1,115,394` |
| Unique characters | `65` |
| SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |

This reference enables a new, documented run. It must not be presented as exact
reproduction of the original local corpus or its saved training results. Lesson
16 creates a contiguous 90%/10% token split; it does not require separate
train/validation text files. Checkpoints and tokenized/processed derivatives
remain local generated artifacts.

## Version-control policy

The four reviewed JSONL datasets are committed. Root `.gitignore` also allows
the single `data/raw/tinyshakespeare.txt` path so the original small corpus can
be added after its contents and provenance are checked. Other raw files,
processed datasets, checkpoints, and run outputs remain ignored. Publishing
the local original later should record its own checksum rather than changing
this upstream checksum to imply that the files were always identical.
