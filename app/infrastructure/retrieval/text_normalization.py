"""Tokenization and light orthographic normalization for the keyword index.

This lives in its own module rather than inside the retriever because it now carries
real linguistic policy with its own edge cases, and because index-time and query-time
text must be processed by the exact same code path: BM25 can only match tokens that
were produced identically on both sides. `tokenize` is that single path.
"""

import re
import unicodedata

# Unicode-aware word characters, minus the underscore. The previous `[a-z0-9]+` dropped
# every non-ASCII script entirely; plain `\w+` would fuse "2024_ok" into one junk token.
# For pure-ASCII text this pattern produces exactly the same tokens as `[a-z0-9]+`,
# which is what keeps English tokenization -- and the English retrieval baseline --
# unchanged. (`re.UNICODE` is the default for str patterns in Python 3; passing it
# explicitly would only imply it does something.)
_TOKEN_RE = re.compile(r"[^\W_]+")

# Tashkeel (U+064B-U+0652), superscript alef (U+0670) and tatweel/kashida (U+0640).
# Tashkeel are combining marks (Unicode category Mn), which `\w` does NOT match: without
# this step a vowelled word shatters into one token per letter cluster instead of one
# token. Tatweel is a letter (category Lm), so it IS matched and would otherwise stay
# glued inside the token.
_ARABIC_STRIP_RE = re.compile("[ً-ْٰـ]")

# Orthographic variants Arabic writers use interchangeably. This is exactly Lucene's
# ArabicNormalizationFilter (plus U+0670), so the fixtures behave here the way they will
# behave in the OpenSearch index that eventually replaces them -- which is what makes
# today's measured Arabic baseline transferable rather than an artifact of a bespoke
# normalizer.
_ARABIC_FOLD = str.maketrans(
    {
        "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
        "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
        "آ": "ا",  # ALEF WITH MADDA ABOVE -> ALEF
        "ٱ": "ا",  # ALEF WASLA            -> ALEF
        "ى": "ي",  # ALEF MAKSURA          -> YEH
        "ة": "ه",  # TEH MARBUTA           -> HEH
    }
)


def normalize(text: str) -> str:
    """Lowercase, NFKC-fold, then apply Arabic light normalization.

    Every step is a no-op for ASCII -- NFKC leaves all 128 ASCII codepoints untouched and
    the Arabic tables only name Arabic codepoints -- so English text comes back unchanged.
    NFKC also folds Arabic Presentation Forms (U+FB50-U+FEFF) back to their base letters,
    which is why those ranges are not listed explicitly.

    Deliberately NOT done here: definite-article ("al-") stripping, light stemming and
    stop-word removal. The first two are morphological rather than orthographic and need
    a real stemmer with a stop-list -- a crude prefix strip mangles words that legitimately
    begin with alef-lam. Stop-word removal is redundant against BM25's IDF and actively
    risky on a corpus this small. Each is a separate, measurable decision.
    """
    folded = unicodedata.normalize("NFKC", text.lower())
    return _ARABIC_STRIP_RE.sub("", folded).translate(_ARABIC_FOLD)


def tokenize(text: str) -> list[str]:
    """The one tokenization path, used at index time and at query time alike."""
    return _TOKEN_RE.findall(normalize(text))
