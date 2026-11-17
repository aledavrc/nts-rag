"""Catálogos cerrados del corpus: campos canónicos, fichas por norma y vacunas.

Son la base determinista del esquema v2 (docs/CONTEXTO_MEJORAS.md, D4, D9 y
Anexo A). Cualquier etiqueta de ficha fuera de `FIELD_VARIANTS` es un error de
`corpus_lint`, no algo que se acepte en silencio.
"""

from __future__ import annotations

import re

from nts_rag.text import normalize_for_match, strip_accents

# D4 — 13 claves canónicas (congeladas).
CANONICAL_FIELDS = (
    "TIPO_DE_INMUNIZACION",
    "DESCRIPCION",
    "COMPOSICION",
    "PRESENTACION",
    "INDICACIONES",
    "ESQUEMA_DE_VACUNACION",
    "DOSIS",
    "CONSERVACION",
    "VIA_Y_SITIO",
    "JERINGAS",
    "EFECTOS_POSTVACUNALES",
    "CONTRAINDICACIONES",
    "USO_SIMULTANEO",
)

# Etiqueta original (normalizada) → claves canónicas.
FIELD_VARIANTS: dict[str, tuple[str, ...]] = {
    "tipo de vacuna": ("TIPO_DE_INMUNIZACION",),
    "tipo de inmunizacion": ("TIPO_DE_INMUNIZACION",),
    "descripcion": ("DESCRIPCION",),
    "composicion": ("COMPOSICION",),
    "presentacion": ("PRESENTACION",),
    "indicacion": ("INDICACIONES",),
    "indicaciones": ("INDICACIONES",),
    "esquema": ("ESQUEMA_DE_VACUNACION",),
    "esquema de vacunacion": ("ESQUEMA_DE_VACUNACION",),
    "esquema de vacunacion y dosis": ("ESQUEMA_DE_VACUNACION", "DOSIS"),
    "dosis": ("DOSIS",),
    "dosis cantidad": ("DOSIS",),
    "conservacion de la vacuna": ("CONSERVACION",),
    "via y sitio de administracion": ("VIA_Y_SITIO",),
    "via y sitio de vacunacion": ("VIA_Y_SITIO",),
    "jeringa": ("JERINGAS",),
    "jeringa a utilizar": ("JERINGAS",),
    "jeringas a utilizar": ("JERINGAS",),
    "posibles efectos post vacunales": ("EFECTOS_POSTVACUNALES",),
    "posibles efectos postvacunales": ("EFECTOS_POSTVACUNALES",),
    "contraindicacion": ("CONTRAINDICACIONES",),
    "contraindicaciones": ("CONTRAINDICACIONES",),
    "uso simultaneo con otras vacunas": ("USO_SIMULTANEO",),
    "uso simultaneo de la vacuna": ("USO_SIMULTANEO",),
}

# Encabezado de las fichas: no es un campo.
FICHA_HEADERS = {"caracteristica"}

# Campos que no existen para una vacuna por su naturaleza ("no aplica", no
# "no consta"): las vacunas orales no usan jeringa.
NO_APLICA: dict[str, set[str]] = {"JERINGAS": {"apo", "rotavirus"}}


def canonical_fields(label: str) -> tuple[str, ...]:
    """Claves canónicas de una etiqueta de ficha; `()` si está fuera del catálogo."""

    return FIELD_VARIANTS.get(normalize_for_match(label), ())


# Anexo A — ficha (numeral) → vacuna_slug por documento. Se alinea por slug,
# nunca por numeral: los numerales cambian entre normas.
FICHAS: dict[str, dict[str, str]] = {
    "nts-246-2026": {
        "6.1.1.1": "bcg",
        "6.1.1.2": "hvb",
        "6.1.1.3": "hexavalente",
        "6.1.1.4": "pentavalente",
        "6.1.1.5": "dt_pediatrico",
        "6.1.1.6": "hib",
        "6.1.1.7": "ipv",
        "6.1.1.8": "apo",
        "6.1.1.9": "rotavirus",
        "6.1.1.10": "neumococo",
        "6.1.1.11": "spr",
        "6.1.1.12": "sr",
        "6.1.1.13": "varicela",
        "6.1.1.14": "ama",
        "6.1.1.15": "dpt",
        "6.1.1.16": "dt_adulto",
        "6.1.1.17": "tdap",
        "6.1.1.18": "hva",
        "6.1.1.19": "vph",
        "6.1.1.20": "influenza",
        "6.1.1.21": "vrs",
        "6.1.1.22": "hexavalente_vih",
        "6.1.1.23": "meningococo",
        "6.1.1.24": "mab_vrs",
    },
    "nts-196-2022": {
        "6.1.1.1": "bcg",
        "6.1.1.2": "hvb",
        "6.1.1.3": "pentavalente",
        "6.1.1.4": "dt_pediatrico",
        "6.1.1.5": "hib",
        "6.1.1.6": "ipv",
        "6.1.1.7": "apo",
        "6.1.1.8": "rotavirus",
        "6.1.1.9": "neumococo",
        "6.1.1.10": "spr",
        "6.1.1.11": "varicela",
        "6.1.1.12": "ama",
        "6.1.1.13": "dpt",
        "6.1.1.14": "dt_adulto",
        "6.1.1.15": "tdap",
        "6.1.1.16": "hva",
        "6.1.1.17": "vph",
        "6.1.1.18": "influenza",
    },
}

VACUNA_SLUGS = tuple(dict.fromkeys(FICHAS["nts-246-2026"].values()))


def ficha_slug(documento_id: str, seccion_numero: str) -> str | None:
    return FICHAS.get(documento_id, {}).get(seccion_numero)


# D2 — tablas de dominio que se parten en filas. Los anexos se reconocen por su
# título porque la numeración cambia entre normas (rescate = Anexo 2 en ambas,
# vías y técnicas = Anexo 6 en 2026 y Anexo 5 en 2022).
DOMAIN_SECTION_PREFIXES = ("6.2", "6.3")
DOMAIN_ANNEX_TITLES = (
    "administracion simultanea",
    "dosis de rescate",
    "frascos abiertos",
    "vias y tecnicas de administracion",
    "esavi severos",
    "errores operativos",
)


def is_domain_section(seccion: str, seccion_numero: str) -> bool:
    number = seccion_numero or ""
    for prefix in DOMAIN_SECTION_PREFIXES:
        if number == prefix or number.startswith(prefix + "."):
            return True
    normalized = normalize_for_match(seccion)
    return "anexo" in normalized and any(title in normalized for title in DOMAIN_ANNEX_TITLES)


# Detección de vacunas en texto libre. Los nombres compuestos van primero y se
# retiran del texto para que "Pentavalente (DPT-HvB-Hib)" no cuente también como
# DPT, HvB e Hib. Las siglas que dependen de mayúsculas (DT pediátrico frente a dT
# adulto, SR, AMA) se buscan sin ignorar mayúsculas.
_I = re.IGNORECASE
_COMPOUNDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hexavalente_vih", re.compile(r"\bDTaP-HvB-Hib-IPV\b")),
    ("hexavalente", re.compile(r"\bDPT-HvB-Hib-IPV\b", _I)),
    ("pentavalente", re.compile(r"\bDPT-HvB-Hib\b", _I)),
    (
        "mab_vrs",
        re.compile(
            r"anticuerpos? monoclonal(?:es)?(?: contra el virus respiratorio sincitial)?"
            r"(?: \(?VRS\)?)?|nirsevimab",
            _I,
        ),
    ),
)
_ALIASES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hexavalente", re.compile(r"\bhexavalente\b", _I)),
    ("pentavalente", re.compile(r"\bpentavalente\b", _I)),
    ("vrs", re.compile(r"\bVRS\b|virus respiratorio sincitial", _I)),
    ("tdap", re.compile(r"\bTdap\b", _I)),
    ("dt_adulto", re.compile(r"\bdT\b|\bDT ADULTO\b")),
    ("dt_pediatrico", re.compile(r"\bDT\b(?! ADULTO)|diftotetano pediatrico|DT PEDIATRICO")),
    ("dpt", re.compile(r"\bDPT\b")),
    ("hvb", re.compile(r"\bHvB\b|\bhepatitis b\b", _I)),
    ("hva", re.compile(r"\bHVA\b|\bHAV\b|\bhepatitis a\b", _I)),
    ("hib", re.compile(r"\bHib\b|haemophilus", _I)),
    ("ipv", re.compile(r"\bIPV\b|antipolio inactivada|\bsalk\b", _I)),
    ("apo", re.compile(r"\bAPO\b|antipolio oral|\bsabin\b", _I)),
    ("rotavirus", re.compile(r"rotavirus", _I)),
    ("neumococo", re.compile(r"neumococ|\bPCV", _I)),
    ("spr", re.compile(r"\bSPR\b|sarampion,? paperas y rubeola", _I)),
    ("sr", re.compile(r"\bSR\b|sarampion y rubeola")),
    ("varicela", re.compile(r"varicela", _I)),
    ("ama", re.compile(r"\bAMA\b|fiebre amarilla|antiamarilica", _I)),
    ("vph", re.compile(r"\bVPH\b|papiloma", _I)),
    ("influenza", re.compile(r"\binfluenza\b", _I)),
    ("meningococo", re.compile(r"meningococ", _I)),
    ("bcg", re.compile(r"\bBCG\b|calmette", _I)),
)


def detect_vaccines(text: str) -> list[str]:
    """Vacunas nombradas en el texto, en orden de primera aparición."""

    remaining = strip_accents(text or "")
    found: dict[str, int] = {}
    for slug, pattern in _COMPOUNDS:
        for match in pattern.finditer(remaining):
            found.setdefault(slug, match.start())
        remaining = pattern.sub(lambda m: " " * len(m.group(0)), remaining)
    for slug, pattern in _ALIASES:
        if slug == "hexavalente" and "hexavalente_vih" in found:
            continue
        match = pattern.search(remaining)
        if match:
            found.setdefault(slug, match.start())
    return [slug for slug, _ in sorted(found.items(), key=lambda item: item[1])]
