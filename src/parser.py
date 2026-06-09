from __future__ import annotations
from .ir import MathNode, var, const as make_const


class ParseError(Exception):
    pass


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> MathNode:
        node = self._expr()
        self._ws()
        if self.pos < len(self.text):
            raise ParseError(
                f"Unexpected input at pos {self.pos}: {self.text[self.pos:]!r}"
            )
        return node

    def _expr(self) -> MathNode:
        self._ws()

        # Try numeric literal first (int, float, negative)
        num = self._try_numeric()
        if num is not None:
            return MathNode(op="const", args=(), value=num)

        name = self._name()
        self._ws()

        if self.pos < len(self.text) and self.text[self.pos] == "(":
            self.pos += 1  # consume '('
            # Special-case: const(...) extracts value directly
            if name == "const":
                return self._parse_const_body()
            args = self._parse_args()
            return MathNode(op=name, args=tuple(args))

        return var(name)

    # ── Numeric literal ───────────────────────────────────────────────────

    def _try_numeric(self) -> int | float | None:
        """
        Try to parse an int or float literal (including negative).
        Returns None and restores pos if not a number.
        """
        start = self.pos
        # Optional leading minus — only if followed by a digit
        if self.pos < len(self.text) and self.text[self.pos] == "-":
            if self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit():
                self.pos += 1
            else:
                return None
        if self.pos >= len(self.text) or not self.text[self.pos].isdigit():
            self.pos = start
            return None
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        # Optional decimal part
        is_float = False
        if self.pos < len(self.text) and self.text[self.pos] == ".":
            is_float = True
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        s = self.text[start : self.pos]
        return float(s) if is_float else int(s)

    # ── const(...) special form ───────────────────────────────────────────

    def _parse_const_body(self) -> MathNode:
        """Parse the body of const(...) — already consumed '('."""
        self._ws()
        num = self._try_numeric()
        if num is not None:
            self._ws()
            self._expect(")")
            return MathNode(op="const", args=(), value=num)
        # Allow const(name) where name is a string value
        name = self._name()
        self._ws()
        self._expect(")")
        # Attempt numeric conversion
        for conv in (int, float):
            try:
                return MathNode(op="const", args=(), value=conv(name))
            except ValueError:
                pass
        return MathNode(op="const", args=(), value=name)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _parse_args(self) -> list[MathNode]:
        """Parse comma-separated exprs until ')'. '(' already consumed."""
        args: list[MathNode] = []
        self._ws()
        if self.pos < len(self.text) and self.text[self.pos] == ")":
            self.pos += 1
            return args
        while True:
            args.append(self._expr())
            self._ws()
            if self.pos >= len(self.text):
                raise ParseError("Unclosed parenthesis")
            if self.text[self.pos] == ")":
                self.pos += 1
                break
            if self.text[self.pos] == ",":
                self.pos += 1
            else:
                raise ParseError(
                    f"Expected ',' or ')' at pos {self.pos}: {self.text[self.pos:]!r}"
                )
        return args

    def _name(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        if self.pos == start:
            raise ParseError(
                f"Expected identifier at pos {self.pos}: "
                f"{self.text[self.pos:self.pos+5]!r}"
            )
        return self.text[start : self.pos]

    def _ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _expect(self, ch: str) -> None:
        if self.pos >= len(self.text) or self.text[self.pos] != ch:
            got = self.text[self.pos : self.pos + 3] if self.pos < len(self.text) else "EOF"
            raise ParseError(f"Expected {ch!r} at pos {self.pos}, got {got!r}")
        self.pos += 1


def parse(text: str) -> MathNode:
    return _Parser(text).parse()
