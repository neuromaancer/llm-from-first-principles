"""Small tokenizers used by the learning project."""

from collections.abc import Sequence


class CharacterTokenizer:
    """Deterministic character-level tokenizer.

    The tokenizer assigns one integer ID to every character in a fixed
    vocabulary. It is intentionally simple so that pretraining experiments can
    focus on the language-model pipeline rather than tokenizer infrastructure.

    Args:
        vocabulary: Ordered collection of unique single-character tokens.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
    ) -> None:
        if not vocabulary:
            raise ValueError(
                "vocabulary must not be empty."
            )

        tokens: tuple[str, ...] = tuple(
            vocabulary
        )

        if any(len(token) != 1 for token in tokens):
            raise ValueError(
                "CharacterTokenizer vocabulary entries must each contain "
                "exactly one character."
            )

        if len(set(tokens)) != len(tokens):
            raise ValueError(
                "vocabulary must not contain duplicate characters."
            )

        self.vocabulary: tuple[str, ...] = tokens
        self.token_to_id: dict[str, int] = {
            token: index
            for index, token in enumerate(tokens)
        }
        self.id_to_token: dict[int, str] = {
            index: token
            for token, index in self.token_to_id.items()
        }

    @classmethod
    def from_text(
        cls,
        text: str,
    ) -> "CharacterTokenizer":
        """Build a deterministic vocabulary from all characters in text.

        Args:
            text: Corpus used to define the character vocabulary.

        Returns:
            Character tokenizer whose vocabulary is sorted by character.
        """
        if not text:
            raise ValueError(
                "text must not be empty."
            )

        return cls(
            sorted(set(text))
        )

    @property
    def vocab_size(self) -> int:
        """Return the number of character tokens."""
        return len(self.vocabulary)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """Convert characters into token IDs.

        Args:
            text: Text containing only characters in the vocabulary.

        Returns:
            Integer token IDs.

        Raises:
            ValueError: If the input contains an unknown character.
        """
        token_ids: list[int] = []

        for character in text:
            if character not in self.token_to_id:
                raise ValueError(
                    f"Unknown character: {character!r}"
                )

            token_ids.append(
                self.token_to_id[character]
            )

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> str:
        """Convert token IDs back into characters.

        Args:
            token_ids: Character-level integer token IDs.

        Returns:
            Decoded text.

        Raises:
            ValueError: If an ID lies outside the vocabulary.
        """
        characters: list[str] = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            characters.append(
                self.id_to_token[token_id]
            )

        return "".join(characters)


__all__ = [
    "CharacterTokenizer",
]
