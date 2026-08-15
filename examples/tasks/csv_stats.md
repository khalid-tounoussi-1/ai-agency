Build a small CSV statistics library.

Provide a function `load_rows(text: str) -> list[dict]` that parses CSV text
with a header row into a list of dictionaries, and a function
`column_stats(rows: list[dict], column: str) -> dict` that returns
`{"count", "min", "max", "mean"}` for a numeric column.

Requirements:
- Values that cannot be parsed as numbers are skipped by `column_stats`, not
  treated as zero.
- `column_stats` on a column with no numeric values returns count 0 and None
  for min, max and mean.
- Requesting a column that does not exist raises `KeyError`.
- `mean` is a float; `min` and `max` keep the numeric type they were parsed as.
- Standard library only.
