"""Router v4 (prereg SS66): entity match, clause scoping and relation keys."""

import numpy as np

from relic.router import (  # noqa: E402
    RouterV4,
    RouterV5,
    clause_with_span,
    find_subject,
    normalize,
    relation_text,
)


class BagEncoder:
    """Deterministic bag-of-words encoder: cosine = word overlap."""

    def __init__(self):
        self.vocab = {}

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for t in texts:
            v = np.zeros(256)
            for w in normalize(t).split():
                w = w.strip(".,?!;")
                if w:
                    v[self.vocab.setdefault(w, len(self.vocab)) % 256] += 1.0
            n = np.linalg.norm(v)
            rows.append(v / n if n else v)
        return np.array(rows)


FACTS = [
    {
        "subject": "Danielle Darrieux",
        "probe": "The mother tongue of Danielle Darrieux is",
    },
    {
        "subject": "Danielle Darrieux",
        "probe": "Danielle Darrieux was born in the city of",
    },
    {"subject": "Jean Galland", "probe": "The mother tongue of Jean Galland is"},
]


def test_normalize_keeps_positions():
    assert normalize("Jean-Luc_Picard") == "jean luc picard"
    assert len(normalize("İstanbul")) == len("İstanbul")


def test_exact_match_requires_word_boundaries():
    assert find_subject("Rabbits eat grass", "Ra") is None
    assert find_subject(
        "The mother tongue of Danielle Darrieux is", "Danielle Darrieux"
    )


def test_approximate_match_needs_every_subject_token():
    assert find_subject("The mother tongue of Danielle Darieux is", "Danielle Darrieux")
    assert (
        find_subject("The mother tongue of Jean-Baptiste Say is", "Jean Galland")
        is None
    )
    assert find_subject("Darrieux was famous", "Danielle Darrieux") is None


def test_clause_scoping_drops_leading_sentence():
    text = "In 1978 her father died. The mother tongue of Danielle Darrieux was"
    span = find_subject(text, "Danielle Darrieux")
    clause, (s, e) = clause_with_span(text, span)
    assert clause == "The mother tongue of Danielle Darrieux was"
    assert clause[s:e] == "Danielle Darrieux"
    assert relation_text(text, span) == "The mother tongue of X was"


def test_router_selects_sibling_by_relation_and_abstains():
    enc = BagEncoder()
    router = RouterV4(enc, FACTS, tau=0.5)
    j, _ = router.route(
        "In 1978 her father died. The mother tongue of Danielle Darrieux was", enc
    )
    assert j == 0
    j, _ = router.route("Danielle Darrieux was born in the city of", enc)
    assert j == 1
    assert router.route("The mother tongue of Jean-Baptiste Say is", enc)[0] is None
    assert router.route("Who is Muku-75?", enc)[0] is None


def test_other_relation_of_stored_subject_is_declined():
    enc = BagEncoder()
    router = RouterV4(enc, FACTS[2:], tau=0.5)
    j, sim = router.route("Jean Galland plays the instrument", enc)
    assert j is None and sim < 0.5


def test_v5_rejects_relation_stored_for_another_subject():
    enc = BagEncoder()
    facts = [
        {
            "subject": "Jean Galland",
            "probe": "The mother tongue of Jean Galland is the language",
        },
        {"subject": "Paul Klee", "probe": "Paul Klee was born in the city of"},
    ]
    query = "Jean Galland was born in the language city of"
    assert RouterV4(enc, facts, tau=0.3).route(query, enc)[0] == 0
    v5 = RouterV5(enc, facts, tau=0.3, margin=0.0)
    assert v5.route(query, enc)[0] is None
    assert v5.route("The mother tongue of Jean Galland is the language", enc)[0] == 0


class BagScorer:
    """Relation scorer over BagEncoder embeddings (cosine)."""

    def __init__(self):
        self.enc = BagEncoder()

    def sims(self, query, keys):
        e = self.enc.encode([query, *keys])
        return [float(e[0] @ k) for k in e[1:]]


def test_v7_generated_keys_raise_paraphrase_score():
    from relic.router import RouterV7

    facts = [
        {"subject": "Jean Galland", "probe": "The mother tongue of Jean Galland is"}
    ]
    extra = {
        "Jean Galland\tThe mother tongue of Jean Galland is": [
            "Jean Galland natively speaks",
            "A rewrite without the name",
        ]
    }
    plain = RouterV7(facts, BagScorer(), tau=0.9)
    aug = RouterV7(facts, BagScorer(), tau=0.9, extra_keys=extra)
    assert aug.key_sets[0] == ["The mother tongue of X is", "X natively speaks"]
    assert plain.route("Jean Galland natively speaks")[0] is None
    assert aug.route("Jean Galland natively speaks")[0] == 0
    assert aug.route("Who is Muku-75?")[0] is None


def _route_subject(facts, query):
    enc = BagEncoder()
    j, _ = RouterV4(enc, facts, tau=0.5).route(query, enc)
    return None if j is None else facts[j]["subject"]


def test_equal_length_subjects_prefer_the_exact_match_in_any_order():
    facts = [
        {"subject": "Iran", "probe": "The capital city of Iran is"},
        {"subject": "Iraq", "probe": "The capital city of Iraq is"},
    ]
    query = "The capital city of Iraq is"
    assert _route_subject(facts, query) == "Iraq"
    assert _route_subject(facts[::-1], query) == "Iraq"


def test_tied_subjects_are_all_candidates_regardless_of_insertion_order():
    facts = [
        {"subject": "Siena", "probe": "Siena is twinned with"},
        {"subject": "Paris", "probe": "The mayor of Paris is"},
    ]
    query = "Siena is lovely. The mayor of Paris is"
    for order in (facts, facts[::-1]):
        router = RouterV4(BagEncoder(), order, tau=0.0)
        cand, spans = router.candidates(query)
        assert sorted(order[i]["subject"] for i in cand) == ["Paris", "Siena"]
        assert [query[s:e] for s, e in spans] == [order[i]["subject"] for i in cand]
    assert _route_subject(facts, query) == _route_subject(facts[::-1], query) == "Paris"


def test_score_ties_break_by_content_not_insertion_order():
    facts = [
        {"subject": "Paris", "probe": "The mayor of Paris is"},
        {"subject": "Paris", "probe": "The twin of Paris is"},
    ]
    query = "The mayor twin of Paris is"
    picks = set()
    for order in (facts, facts[::-1]):
        enc = BagEncoder()
        router = RouterV4(enc, order, tau=0.0)
        _, sims = router.score(query, enc)
        assert sims[0] == sims[1]
        picks.add(order[router.route(query, enc)[0]]["probe"])
    assert picks == {"The mayor of Paris is"}


def test_exact_match_beats_a_longer_near_namesake():
    facts = [
        {"subject": "Robert Smith", "probe": "Robert Smith works for"},
        {"subject": "Roberta Smith", "probe": "Roberta Smith works for"},
    ]
    query = "Robert Smith works for"
    assert _route_subject(facts, query) == "Robert Smith"
    assert _route_subject(facts[::-1], query) == "Robert Smith"


def test_misspelled_longer_subject_still_beats_its_exact_prefix():
    facts = [
        {"subject": "Paris", "probe": "The mayor of Paris is"},
        {"subject": "Paris Hilton", "probe": "The mother of Paris Hilton is"},
    ]
    query = "The mother of Paris Hiltn is"
    assert _route_subject(facts, query) == "Paris Hilton"


class ToyScorer:
    def __init__(self):
        self.enc = BagEncoder()

    def sims(self, query, keys):
        e = self.enc.encode([query, *keys])
        return [float(e[0] @ k) for k in e[1:]]


def _v7(facts, tau=0.5):
    from relic.router import RouterV7

    return RouterV7(facts, ToyScorer(), tau)


OBAMA = {
    "id": "a",
    "subject": "Barack Obama",
    "probe": "The mother tongue of Barack Obama is",
    "aliases": ["Obama", "obama the president", "BHO"],
}


def test_alias_routes_to_its_entry_and_short_or_lowercase_aliases_do_not():
    router = _v7([OBAMA])
    assert router.route("The mother tongue of Obama is")[0] == 0
    assert router.route("The mother tongue of BHO is")[0] is None
    assert router.route("The mother tongue of obama the president is")[0] == 0
    no_alias = _v7([{k: v for k, v in OBAMA.items() if k != "aliases"}])
    assert no_alias.route("The mother tongue of Obama is")[0] is None


def test_full_subject_outranks_a_shorter_alias_of_another_entry():
    other = {"id": "b", "subject": "Paris Hilton", "probe": "Paris Hilton plays"}
    city = {
        "id": "c",
        "subject": "Lutetia",
        "probe": "Lutetia plays",
        "aliases": ["Paris"],
    }
    router = _v7([other, city])
    assert router.candidates("Paris Hilton plays")[0] == [0]
    assert router.candidates("Paris plays")[0] == [1]


def test_relation_threshold_applies_to_entries_of_that_relation():
    from relic.router import RouterV7

    fact = {**OBAMA, "relation": "P103"}
    q = "The mother tongue of Barack Obama is"
    assert RouterV7([fact], ToyScorer(), 0.5).route(q)[0] == 0
    strict = RouterV7([fact], ToyScorer(), 0.5, relation_tau={"P103": 1.5})
    assert strict.route(q)[0] is None
    other = RouterV7([fact], ToyScorer(), 0.5, relation_tau={"P37": 1.5})
    assert other.route(q)[0] == 0


def test_pronoun_resolves_to_the_last_stored_subject():
    from relic.router import resolve_reference

    router = _v7([OBAMA])
    history = ["Tell me about Barack Obama.", "Go on."]
    assert (
        resolve_reference("The mother tongue of him is", history, router)
        == "The mother tongue of Barack Obama is"
    )
    assert (
        resolve_reference("What is his mother tongue?", history, router)
        == "What is Barack Obama's mother tongue?"
    )
    assert (
        resolve_reference("What is her mother tongue?", history, router)
        == "What is Barack Obama's mother tongue?"
    )


def test_a_near_spelling_in_the_history_is_not_a_mention():
    from relic.router import resolve_reference

    router = _v7([{**OBAMA, "aliases": ["Morven"]}])
    q = "What is his mother tongue?"
    # "more" is within edit distance 2 of the alias, as a real store showed
    assert resolve_reference(q, ["Can you say more?"], router) == q
    assert resolve_reference(q, ["Tell me about Morven."], router) == (
        "What is Barack Obama's mother tongue?"
    )
    assert resolve_reference(
        q, ["Tell me about Barack Obama.", "Can you say more?"], router
    ) == ("What is Barack Obama's mother tongue?")


def test_pronoun_is_left_alone_when_nothing_safe_resolves_it():
    from relic.router import resolve_reference

    router = _v7([OBAMA])
    q = "What is his mother tongue?"
    assert resolve_reference(q, [], router) == q
    assert resolve_reference(q, ["Tell me a joke."], router) == q
    # a more recent turn names someone who is not stored
    named = ["Tell me about Barack Obama.", "What about Angela Merkel?"]
    assert resolve_reference(q, named, router) == q
    # a query that names a stored subject itself is not rewritten
    own = "The mother tongue of Barack Obama and his family is"
    assert resolve_reference(own, ["Tell me about Barack Obama."], router) == own
