"""Fetch a pinned public Tiny Shakespeare snapshot; never replace a local corpus.

Run from the repository root: python data/prepare_tinyshakespeare.py
Uses only the Python standard library. See data/README.md for the four-character
mismatch between this upstream reference and the original saved lesson output.
"""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

SOURCE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    f"{SOURCE_COMMIT}/data/tinyshakespeare/input.txt"
)
EXPECTED_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
EXPECTED_BYTES = 1_115_394
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "raw" / "tinyshakespeare.txt"


def prepare_corpus(destination: Path) -> str:
    """Download or verify the pinned reference, preserving any different file."""
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != EXPECTED_SHA256:
            raise FileExistsError(
                f"Preserved existing corpus: {destination}\n"
                f"Its SHA-256 is {actual}, not the pinned upstream checksum.\n"
                "Do not replace your original lesson data blindly. Use --output "
                "with a different path to obtain a separate upstream reference."
            )
        return f"Already verified: {destination}"

    with urlopen(SOURCE_URL, timeout=30) as response:
        content = response.read(EXPECTED_BYTES + 1)
    actual = hashlib.sha256(content).hexdigest()
    if len(content) != EXPECTED_BYTES or actual != EXPECTED_SHA256:
        raise ValueError("Downloaded content does not match the pinned snapshot.")
    text = content.decode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents replacing a file created during download.
    with destination.open("xb") as output:
        output.write(content)
    return (
        f"Downloaded upstream reference: {destination}\n"
        f"{len(text):,} characters; {len(set(text))} unique characters; "
        f"SHA-256 {actual}\n"
        "This is not verified as byte-identical to the original lesson corpus."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        print(prepare_corpus(args.output))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
