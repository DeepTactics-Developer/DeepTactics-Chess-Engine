# Opening Book Data

Optional opening data goes in `openings.tsv`.

Preferred row format:

```text
ECO<TAB>Opening Name<TAB>SAN moves separated by spaces
```

Legacy two-column rows still work:

```text
Opening Name<TAB>SAN moves separated by spaces
```

Example:

```text
C60	Ruy Lopez	e4 e5 Nf3 Nc6 Bb5
B20	Sicilian Defense	e4 c5
D06	Queen's Gambit	d4 d5 c4
```

Opening detection is not self-play learning. It is a lookup against known opening sequences, usually from ECO/opening book data.

If you have ECO TSV files split as `a.tsv` through `e.tsv`, import them together:

```powershell
python -m chess_engine import-openings --input path\to\a.tsv path\to\b.tsv path\to\c.tsv path\to\d.tsv path\to\e.tsv --output assets/openings/openings.tsv
```

