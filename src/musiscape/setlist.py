"""Aligning detected pieces with the setlist.

A concert's running order says what was planned; the recording says what happened, in
what order, and what was dropped. Given the pieces a segmenter found and the text of the
spoken introduction before each (from any transcriber), this matches names heard to
names planned and fills the rest in running order. Names the host *thanks* belong to
the act that just finished and are ignored; a name right after "vær så god" or "ved"
is the act being introduced and counts extra. Acts nobody was matched to come back as
``not_detected``: cancelled, or not a musical number.

The setlist is a JSON list of acts ``{"nr", "act", "performers", "work", "composer",
"contact"}`` or a ``.docx`` whose first table has such columns (the IMV *kjøreplan*
template: Nr. / Innslag / Komponist / Låt / Medvirkende).
"""
from __future__ import annotations

import difflib
import html
import json
import re
import zipfile
from pathlib import Path

HEADER_ALIASES = {
    "nr": ["nr", "nr.", "#", "no", "no."],
    "act": ["innslag", "act", "title", "tittel"],
    "composer": ["komponist / tekst", "komponist", "composer"],
    "work": ["låt / stykke", "låt", "stykke", "work", "piece", "verk"],
    "performers": ["medvirkende", "performers", "musicians", "utøvere"],
    "contact": ["kontaktperson", "contact"],
}

_CAP = re.compile(r"\b[A-ZÆØÅ][a-zæøåé\-]{2,}")
_WORD = re.compile(r"\b[A-Za-zÆØÅæøåéÉ][a-zæøåé\-]{2,}")
_THANKS = re.compile(r"[Tt]akk,?\s+(?:til\s+|for\s+\w+,?\s+)?((?:[A-ZÆØÅ][\wæøåé\-]+\s*){1,3})")
_CUE = re.compile(r"(?i)(?:vær så god|få høre|få et innslag med|innslag med|velkommen|presentere|ønske velkommen|\bved"
                  r"|call upon|calls upon|invite|please welcome|welcome|come forward|the floor|over to)[,:]?\s*")
# Words a running order capitalises that are not names: the parts of a defence or a seminar and the
# fillers of a concert plan. A word here never matches on its own.
_GENERIC = {"solo", "piano", "duo", "band", "div", "allsang", "spiller", "fra", "og", "ikke", "bestemt",
            "egen", "improvisert", "alle", "fellesnummer", "untitled", "study", "the", "for", "med", "ved",
            "first", "second", "third", "fourth", "trial", "thesis", "lecture", "introduction", "opponent",
            "committee", "announcement", "candidate", "break", "part", "session", "panel", "keynote",
            "opening", "closing", "discussion", "questions", "welcome", "university", "professor"}
# A title this long is a sentence, not a name: its words count only as pairs.
_TITLE_AS_PHRASE = 3


def _docx_tables(path: Path) -> list[list[list[str]]]:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    tables = []
    for tbl in re.findall(r"<w:tbl>.*?</w:tbl>", xml, flags=re.S):
        rows = []
        for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, flags=re.S):
            cells = []
            for tc in re.findall(r"<w:tc>.*?</w:tc>", tr, flags=re.S):
                tc = re.sub(r"<w:br/>", " / ", tc)
                paras = [html.unescape(re.sub(r"<[^>]+>", "", p)) for p in re.findall(r"<w:p[ >].*?</w:p>", tc, flags=re.S)]
                cells.append(" / ".join(p.strip() for p in paras if p.strip()))
            rows.append(cells)
        tables.append(rows)
    return tables


def load_setlist(path: str | Path) -> list[dict]:
    """Acts from a ``.json`` list or the first suitable table of a ``.docx``."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() != ".docx":
        raise ValueError("setlist must be .json or .docx")
    for rows in _docx_tables(path):
        if not rows:
            continue
        header = [c.strip().lower() for c in rows[0]]
        col = {}
        for key, aliases in HEADER_ALIASES.items():
            for i, h in enumerate(header):
                if h in aliases:
                    col[key] = i
                    break
        if "act" not in col:
            continue
        acts = []
        for r in rows[1:]:
            get = lambda k: r[col[k]].strip() if k in col and col[k] < len(r) else ""
            if not get("act") and not get("performers"):
                continue
            acts.append({k: get(k) for k in ("nr", "act", "composer", "work", "performers", "contact")})
        if acts:
            return acts
    raise ValueError("no table with an act/innslag column found")


def thanked_names(text: str) -> set[str]:
    """Names the host thanks: the previous act. A transcript that is nothing but the thanks is left alone."""
    out = set()
    for m in _THANKS.finditer(text or ""):
        if len(text) - m.end() < 3:
            continue
        out |= {t.lower() for t in _CAP.findall(m.group(1))}
    return out


def intro_names(text: str | None, cues: bool = True) -> list[tuple[str, float, float]]:
    """``(token, relative position, cue bonus)`` for the words of an introduction that could be names.

    A word said right after a hand-over cue ("vær så god", "I now call upon") carries a bonus;
    ``cues=False`` scores the words alone, which is how ``align_setlist`` tells a cued name apart."""
    if not text:
        return []
    skip = thanked_names(text)
    cue_spans = [(m.end(), m.end() + 45) for m in _CUE.finditer(text)] if cues else []
    toks = []
    for m in _WORD.finditer(text):
        t = m.group(0).lower()
        if t in skip:
            continue
        bonus = 0.1 if any(a <= m.start() <= b for a, b in cue_spans) else 0.0
        toks.append((t, m.start() / max(1, len(text)), bonus))
    return toks


def _plan_tokens(act: dict) -> tuple[set[str], set[str]]:
    """The names an act can be recognised by: single words and pairs of consecutive capitalised words.

    People (``performers``, ``composer``, ``contact``) are matched on either, since a host says a
    surname alone. A title (``act``, ``work``) with ``_TITLE_AS_PHRASE`` or more capitalised words is a
    sentence, and its words are ordinary words that turn up in any speech, so it matches only as a
    phrase: "Machine Synchresis", never "Machine". A short title is a name and matches as one."""
    uni: set[str] = set(); bi: set[str] = set()
    for field in ("act", "work", "performers", "composer", "contact"):
        caps = [t.lower() for t in _CAP.findall(str(act.get(field, "") or ""))]
        phrase_only = field in ("act", "work") and len(caps) >= _TITLE_AS_PHRASE
        if not phrase_only:
            uni |= {t for t in caps if t not in _GENERIC}
        bi |= {f"{a} {b}" for a, b in zip(caps, caps[1:]) if a not in _GENERIC and b not in _GENERIC}
    return uni, bi


def plan_tokens(acts: list[dict]) -> list[tuple[set[str], set[str]]]:
    """``_plan_tokens`` for every act, with the names that two or more acts share taken out of all of
    them: a candidate who gives both the trial lecture and the introduction is named in both acts, and
    hearing that name says nothing about which of the two is starting."""
    toks = [_plan_tokens(a) for a in acts]
    shared_uni = {t for i, (u, _) in enumerate(toks) for t in u if any(t in u2 for j, (u2, _) in enumerate(toks) if j != i)}
    shared_bi = {t for i, (_, b) in enumerate(toks) for t in b if any(t in b2 for j, (_, b2) in enumerate(toks) if j != i)}
    return [(u - shared_uni, b - shared_bi) for u, b in toks]


def name_score(intro_text: str | None, act: dict, tokens: tuple[set[str], set[str]] | None = None,
               cues: bool = True) -> tuple[float, float]:
    """``(score, position)``: best fuzzy match between the intro and the act; full names count more.

    ``tokens`` is the act's entry from ``plan_tokens(acts)``; without it the act is scored on its own
    names, including any it shares with another act. ``cues=False`` leaves out the hand-over bonus."""
    names = intro_names(intro_text, cues)
    if not names:
        return 0.0, 0.0
    uni, bi = tokens if tokens is not None else _plan_tokens(act)
    best, pos = 0.0, 0.0
    for k, (a, p, bonus) in enumerate(names):
        for b in uni:
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r >= 0.75 and r + bonus > best:
                best, pos = r + bonus, p
        if k + 1 < len(names):
            pair = f"{a} {names[k + 1][0]}"
            for b in bi:
                r = difflib.SequenceMatcher(None, pair, b).ratio()
                if r >= 0.8 and r + 0.1 + bonus > best:
                    best, pos = r + 0.1 + bonus, p
    return best, pos


def align_setlist(pieces: list[dict], acts: list[dict], min_score: float = 0.8, reading_acts: int = 3) -> dict:
    """Match detected pieces (``{"id", "intro": text}`` in playing order) to acts.

    Returns ``assignments`` (piece id -> act index or None), ``how`` (``name`` / ``order`` /
    ``continues``), ``not_detected`` (act indices) and the per-act ``scores`` of every piece.

    An introduction that names ``reading_acts`` or more acts is the programme being read out, the
    chair listing the committee or the host running through the evening, and says nothing by itself
    about which act is starting. If one of those names follows a hand-over cue ("I now call upon"),
    that act is the one; otherwise the names are set aside and the piece is placed by running order.
    Two names in one introduction are still a hand-over, where the one said last wins.
    """
    n, m = len(pieces), len(acts)
    toks = plan_tokens(acts)
    scored = [[name_score(p.get("intro"), a, toks[j]) for j, a in enumerate(acts)] for p in pieces]
    plain = [[name_score(p.get("intro"), a, toks[j], cues=False) for j, a in enumerate(acts)] for p in pieces]
    cued = [{j for j in range(m) if scored[i][j][0] >= min_score and scored[i][j][0] > plain[i][j][0]} for i in range(n)]
    reading = {i for i in range(n) if sum(sc >= min_score for sc, _ in scored[i]) >= reading_acts}
    assign: list[int | None] = [None] * n
    used: set[int] = set()
    cands = sorted(((sc, pos, i, j) for i in range(n) for j in range(m) for sc, pos in [scored[i][j]]
                    if sc >= min_score and (i not in reading or j in cued[i])),
                   key=lambda c: (c[0], c[1]), reverse=True)
    for sc, pos, i, j in cands:                       # confident names; the one said last wins a tie
        if assign[i] is None and j not in used:
            assign[i] = j; used.add(j)
    how = {i: "name" for i in range(n) if assign[i] is not None}
    for i in range(n):                                # leftovers by running order between matched neighbours
        if assign[i] is not None:
            continue
        lo = max([assign[k] for k in range(i) if assign[k] is not None], default=-1)
        hi = min([assign[k] for k in range(i + 1, n) if assign[k] is not None], default=m)
        free = [j for j in range(lo + 1, hi) if j not in used]
        if free:
            assign[i] = free[0]; used.add(free[0]); how[i] = "order"
    for i in range(1, n):                             # nobody named and nothing free: the act continues
        if assign[i] is None and assign[i - 1] is not None and ((i in reading and not cued[i]) or max((sc for sc, _ in scored[i]), default=0.0) < min_score):
            assign[i] = assign[i - 1]; how[i] = "continues"
    return {"assignments": {pieces[i]["id"]: assign[i] for i in range(n)},
            "how": {pieces[i]["id"]: how.get(i) for i in range(n)},
            "not_detected": [j for j in range(m) if j not in set(a for a in assign if a is not None)],
            "scores": {pieces[i]["id"]: [round(sc, 2) for sc, _ in scored[i]] for i in range(n)}}


def act_title(act: dict) -> str:
    """``"3. Menuett fra Suite op. 20 – Øyvin Dybsand"``: the work when there is one, else the act name."""
    work = act.get("work") or ""
    work = work.split(":", 1)[1].strip() if ":" in work else work
    work = " / ".join(w.strip(" ,/") for w in work.split("/") if w.strip(" ,/"))
    label = work if work and work not in ("–", "-", "Ikke bestemt") else act.get("act", "")
    who = act.get("performers") or ""
    title = f"{act.get('nr', '')}. {label}".strip(". ")
    return title + (f" – {who}" if who and who not in ("–", "-") else "")
