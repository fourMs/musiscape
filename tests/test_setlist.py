"""Setlist alignment: names heard against names planned, thank-yous ignored, order for the rest."""
import json
import zipfile

import pytest

from musiscape import setlist as sl

ACTS = [
    {"nr": "1", "act": "Karen og Kristina", "work": "Sonatina", "composer": "Malcolm Arnold", "performers": "Karen Martinsen og Kristina Celik", "contact": ""},
    {"nr": "2", "act": "Alexander + alle", "work": "fellesnummer", "composer": "", "performers": "", "contact": "alexanje"},
    {"nr": "3", "act": "Øyvin spiller", "work": "Menuett fra Suite op. 20", "composer": "Agathe Backer Grøndahl", "performers": "Øyvin Dybsand", "contact": ""},
    {"nr": "4", "act": "Fredrik solo", "work": "Egen/improvisert", "composer": "", "performers": "Fredrik Beckstrøm", "contact": ""},
    {"nr": "5", "act": "Div + allsang", "work": "–", "composer": "–", "performers": "–", "contact": "Koret"},
    {"nr": "6", "act": "Erik Damhus", "work": "Prelude To A Kiss / My Romance", "composer": "Duke Ellington / Richard Rodgers", "performers": "Erik Damhus", "contact": ""},
    {"nr": "7", "act": "Gravel Peak", "work": "Sound of You, What a Shame", "composer": "", "performers": "Jørgen Berg Hansen, Maria Kverno", "contact": ""},
]

PIECES = [
    {"id": "p1", "intro": "Velkommen. Vi skal starte med litt klassisk musikk ved Øyvind Dypsand, vær så god."},
    {"id": "p2", "intro": "Tusen takk, Øyvind. Da skal vi få et innslag med Alexander Retsum. Ja, vi skal få litt jazz ved Fredrik Beekstrøm. Vær så god."},
    {"id": "p3", "intro": "Tusen takk til Fredrik. Nå skal dere få høre årets utgave av koret. Vær så god, Åsle med koret."},
    {"id": "p4", "intro": "Nå skal vi ta en annen sang. Kan ikke du få litt mer lyd?"},
    {"id": "p5", "intro": "Takk, Richard Rogers."},
    {"id": "p6", "intro": "Tusen takk Erik. Dagens siste innslag er et band som heter Gravel Peak."},
]


def test_thanks_are_the_previous_act():
    assert sl.thanked_names("Tusen takk, Øyvind. Da skal vi få høre Fredrik.") == {"øyvind"}
    assert sl.thanked_names("Takk, Richard Rogers.") == set()          # a fragment, not a hand-over


def test_alignment_survives_reordering_and_cancellations():
    r = sl.align_setlist(PIECES, ACTS)
    got = {pid: (None if j is None else ACTS[j]["nr"]) for pid, j in r["assignments"].items()}
    assert got == {"p1": "3", "p2": "4", "p3": "5", "p4": "5", "p5": "6", "p6": "7"}
    assert r["how"]["p4"] == "continues" and r["how"]["p1"] == "name"
    assert [ACTS[j]["nr"] for j in r["not_detected"]] == ["1", "2"]


def test_the_name_said_last_wins_over_an_earlier_mention():
    r = sl.align_setlist([PIECES[1]], ACTS)
    assert ACTS[r["assignments"]["p2"]]["nr"] == "4"


DEFENCE = [
    {"nr": "1", "act": "Trial lecture: Applying Michel Chion’s concept of synchresis in multimedia research", "performers": "Jinyue Guo"},
    {"nr": "2", "act": "Thesis introduction: Machine Synchresis: Immersive and Embodied Audio–Video Information Retrieval", "performers": "Jinyue Guo"},
    {"nr": "3", "act": "First opponent", "performers": "Antonio Camurri, University of Genova"},
    {"nr": "4", "act": "Second opponent", "performers": "Annamaria Mesaros, Tampere University"},
]


def test_a_long_title_matches_as_a_phrase_and_not_word_by_word():
    toks = sl.plan_tokens(DEFENCE)
    assert sl.name_score("we did not include embodied cognition or information about the machine", DEFENCE[1], toks[1])[0] < 0.8
    assert sl.name_score("submitted a thesis entitled Machine Synchresis, Immersive and Embodied", DEFENCE[1], toks[1])[0] >= 0.8


def test_a_name_two_acts_share_names_neither():
    toks = sl.plan_tokens(DEFENCE)
    assert "jinyue" not in toks[0][0] and "guo" not in toks[1][0]
    assert sl.name_score("I now invite the candidate, Jinyue Guo, to the stage", DEFENCE[0], toks[0])[0] < 0.8


def test_the_parts_of_a_defence_are_not_names():
    toks = sl.plan_tokens(DEFENCE)
    assert sl.name_score("so the first aspect here is the second one of the university", DEFENCE[2], toks[2])[0] < 0.8


def test_reading_out_the_committee_names_no_act():
    committee = ("The Faculty appointed the following evaluation committee: Professor Antonio Camurri, University of "
                 "Genova, and Assistant Professor Annamaria Mesaros, Tampere University. The trial lecture on Michel Chion "
                 "was found satisfactory. The candidate will now present the thesis.")
    r = sl.align_setlist([{"id": "p1", "intro": "I now invite the candidate to give the trial lecture on Michel Chion"},
                          {"id": "p2", "intro": committee}], DEFENCE)
    assert r["assignments"] == {"p1": 0, "p2": 1} and r["how"]["p2"] == "order"


def test_a_hand_over_cue_picks_the_act_out_of_a_recap():
    recap = ("Earlier today the candidate gave the trial lecture on Michel Chion and presented the thesis Machine "
             "Synchresis. The second opponent will be Annamaria Mesaros. I now call upon the first opponent, "
             "Professor Antonio Camurri, to come forward.")
    r = sl.align_setlist([{"id": "p3", "intro": recap}], DEFENCE)
    assert r["assignments"] == {"p3": 2} and r["how"]["p3"] == "name"


def test_act_title_cleans_the_work_cell():
    assert sl.act_title(ACTS[6]) == "7. Sound of You, What a Shame – Jørgen Berg Hansen, Maria Kverno"
    assert sl.act_title({"nr": "9", "act": "Band", "work": "Låtene vi skal spille er: / A, / B", "performers": "–"}) == "9. A / B"


def _docx(path, rows):
    cell = lambda t: f"<w:tc><w:p><w:r><w:t>{t}</w:t></w:r></w:p></w:tc>"
    body = "".join("<w:tr>" + "".join(cell(c) for c in r) + "</w:tr>" for r in rows)
    xml = ('<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body><w:tbl>{body}</w:tbl></w:body></w:document>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)
    return path


def test_load_setlist_from_docx_and_json(tmp_path):
    rows = [["Nr.", "Kontaktperson", "Innslag", "Komponist / tekst", "Låt / stykke", "Medvirkende"],
            ["1", "Kari", "Kari solo", "Grieg", "Arietta", "Kari Nordmann"],
            ["2", "Koret", "Div + allsang", "–", "–", "–"]]
    acts = sl.load_setlist(_docx(tmp_path / "plan.docx", rows))
    assert [a["act"] for a in acts] == ["Kari solo", "Div + allsang"]
    assert acts[0]["composer"] == "Grieg" and acts[1]["contact"] == "Koret"
    (tmp_path / "plan.json").write_text(json.dumps(acts))
    assert sl.load_setlist(tmp_path / "plan.json") == acts
    with pytest.raises(ValueError):
        sl.load_setlist(tmp_path / "plan.txt")
