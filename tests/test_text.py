from nts_rag.text import normalize_for_match, short_section, tokenize


def test_normalize_strips_accents_and_punctuation():
    assert normalize_for_match("Vacunación contra el VPH (Papiloma)") == (
        "vacunacion contra el vph papiloma"
    )


def test_tokenize_removes_stopwords():
    tokens = tokenize("¿Cuál es la dosis de la vacuna contra el rotavirus?")
    assert "dosi" in tokens
    assert "rotaviru" in tokens
    assert "cual" not in tokens
    assert "de" not in tokens


def test_stem_matches_plural_and_singular():
    assert tokenize("contraindicaciones") == tokenize("contraindicación")
    assert tokenize("gestantes") == tokenize("gestante")
    assert tokenize("vacunas") == tokenize("vacuna")
    assert tokenize("enfermedades") == tokenize("enfermedad")
    assert tokenize("simultáneamente") == tokenize("simultánea")




def test_short_section_keeps_leaf():
    route = "NTS 246 > VI. DISPOSICIONES > 6.1.1.10. VACUNA ANTINEUMOCÓCICA"
    assert short_section(route) == "6.1.1.10. VACUNA ANTINEUMOCÓCICA"
