from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "a",
    "al",
    "como",
    "con",
    "cual",
    "cuales",
    "cuando",
    "de",
    "del",
    "donde",
    "el",
    "en",
    "es",
    "esta",
    "este",
    "fue",
    "ha",
    "han",
    "la",
    "las",
    "lo",
    "los",
    "mas",
    "para",
    "por",
    "que",
    "se",
    "sea",
    "segun",
    "sin",
    "sobre",
    "son",
    "su",
    "sus",
    "tiene",
    "tienen",
    "un",
    "una",
    "y",
    # Términos del dominio que aparecen en casi todos los fragmentos y no
    # discriminan: actuar como stopwords mejora la señal de los términos raros
    # (regurgita, anafilaxia, dosis concreta…). "año(s)" ya no está aquí: la
    # edad es parte de la pregunta (causa raíz 1) y además se emiten tokens de
    # edad (`age_tokens`).
    "aplicacion",
    "aplicar",
    "grupo",
    "hacer",
    "nina",
    "ninas",
    "nino",
    "ninos",
    "paciente",
    "pacientes",
    "persona",
    "poblacion",
    "vacuna",
    "vacunacion",
    "vacunas",
}


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_for_match(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stem(token: str) -> str:
    """Recorte ligero de plurales y vocales finales para emparejar léxicamente.

    No pretende ser un stemmer completo: resuelve los casos frecuentes del
    corpus (contraindicaciones/contraindicación, vacunas/vacuna, gestantes/
    gestante, enfermedades/enfermedad) sin tocar siglas ni dosis.
    """

    if len(token) <= 3:
        return token
    if token.endswith("mente") and len(token) > 8:
        token = token[:-5]
    if token.endswith("es"):
        token = token[:-2]
    elif token.endswith("s"):
        token = token[:-1]
    if len(token) > 5 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    normalized = normalize_for_match(text)
    tokens = normalized.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return [stem(token) for token in tokens] + age_tokens(normalized)


# --- Edades -----------------------------------------------------------------

_YEARS = r"anos?"
_MONTHS = r"meses|mes"
_AGE_RANGE_RE = re.compile(rf"\b(\d{{1,2}})\s*(?:a|y|-)\s*(\d{{1,2}})\s*({_YEARS}|{_MONTHS})\b")
_AGE_SINGLE_RE = re.compile(rf"\b(\d{{1,2}})\s*({_YEARS}|{_MONTHS})\b")
_AGE_TOKEN_SPAN = 24


def _is_years(unit: str) -> bool:
    return unit.startswith("ano")


def age_tokens(normalized: str) -> list[str]:
    """Tokens sintéticos de edad: "5 años" → `edad5a`; "5 a 11 años" → edad5a…edad11a.

    Así una pregunta por "5 años" casa léxicamente con la fila "5 a 11 años" del
    calendario, cosa que el número suelto no consigue. Espera texto ya pasado por
    `normalize_for_match`.
    """

    tokens: list[str] = []
    covered: list[tuple[int, int]] = []
    for match in _AGE_RANGE_RE.finditer(normalized):
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end and end - start <= _AGE_TOKEN_SPAN:
            suffix = "a" if _is_years(match.group(3)) else "m"
            tokens += [f"edad{n}{suffix}" for n in range(start, end + 1)]
            covered.append(match.span())
    for match in _AGE_SINGLE_RE.finditer(normalized):
        if any(a <= match.start() < b for a, b in covered):
            continue
        suffix = "a" if _is_years(match.group(2)) else "m"
        tokens.append(f"edad{match.group(1)}{suffix}")
    return tokens


_HASTA_RE = re.compile(
    r"hasta (?:los |las |el |la )?((?:\d{1,2}\s*(?:anos?|meses|mes|dias?)\s*(?:y\s*)?)+)"
)
_COMPONENT_RE = re.compile(r"(\d{1,2})\s*(anos?|meses|mes|dias?)")


def _components_to_months(text: str) -> int:
    months = 0
    for amount, unit in _COMPONENT_RE.findall(text):
        if unit.startswith("ano"):
            months += 12 * int(amount)
        elif unit.startswith("mes"):
            months += int(amount)
    return months


def age_limit_months(text: str) -> int | None:
    """Mayor límite "hasta los …" del texto, en meses cumplidos.

    "hasta los 4 años 11 meses 29 días" → 59 (los días no completan un mes).
    """

    normalized = normalize_for_match(text)
    limits = [_components_to_months(m.group(1)) for m in _HASTA_RE.finditer(normalized)]
    return max(limits) if limits else None


def age_range_months(text: str) -> tuple[int | None, int | None] | None:
    """Rango de edad, en meses inclusivos, de una celda tipo "EDAD" del calendario.

    "Recién nacido" → (0, 0); "2 meses" → (2, 2); "4 años" → (48, 59);
    "5 a 11 años" → (60, 143); "menor de 2 años" → (0, 23); "mayores de 60 años"
    → (720, None). Edades gestacionales (semanas) no son edad del vacunado: None.
    """

    t = normalize_for_match(text)
    if "recien nacido" in t or "neonat" in t:
        return (0, 0)
    m = _AGE_RANGE_RE.search(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if _is_years(m.group(3)):
            return (12 * a, 12 * b + 11)
        return (a, b)
    m = re.search(rf"menor(?:es)? de (\d{{1,2}})\s*({_YEARS}|{_MONTHS})\b", t)
    if m:
        n = int(m.group(1))
        return (0, 12 * n - 1 if _is_years(m.group(2)) else n - 1)
    m = re.search(rf"mayor(?:es)? de (\d{{1,2}})\s*({_YEARS}|{_MONTHS})\b", t)
    if m:
        n = int(m.group(1))
        return (12 * n if _is_years(m.group(2)) else n, None)
    m = _AGE_SINGLE_RE.search(t)
    if m:
        n = int(m.group(1))
        return (12 * n, 12 * n + 11) if _is_years(m.group(2)) else (n, n)
    return None


# --- Versión de la norma (D7) -------------------------------------------------

_HISTORIC_RE = re.compile(
    r"\b2022\b|\b196\b|\bderogada\b|"
    r"\b(?:norma|nts|version|esquema)s? (?:anterior|antigua|previa|pasada)\b"
)
_CURRENT_RE = re.compile(r"\b2026\b|\b246\b|\bvigente\b|\b(?:norma|nts|version) actual\b")
# "antes" y "previa" sueltos no cuentan: "vacunas previas" o "antes de vacunar"
# no hablan de otra norma (§12, trampas).
_COMPARE_RE = re.compile(
    r"\bcompar|\bcambi(?:o|aron|os|ado)\b|\bdiferencia|"
    r"\bvacunas? nuevas?\b|\bnuevas? vacunas?\b|\bincorpora|\bversus\b|\bvs\b"
)


def detect_vigencias(query: str) -> tuple[str, ...]:
    """Normas que debe consultar una pregunta: por defecto solo la vigente.

    La histórica entra solo con señal explícita (2022, 196, "norma anterior");
    una comparación o la mención de ambas consulta las dos.
    """

    t = normalize_for_match(query)
    historic = bool(_HISTORIC_RE.search(t))
    current = bool(_CURRENT_RE.search(t))
    compare = bool(_COMPARE_RE.search(t))
    if (historic and (current or compare)) or (compare and current):
        return ("vigente", "derogada")
    if historic:
        return ("derogada",)
    return ("vigente",)


# Expansión mínima para términos coloquiales del dominio. No es un diccionario
# clínico: solo acerca la pregunta al vocabulario que usa la norma.
QUERY_SYNONYMS = {
    "gestante": ("embarazo", "gestacion", "prenatal"),
    "embarazo": ("gestante", "gestacion", "prenatal"),
    "bebe": ("recien nacido", "neonato"),
    "nino": ("nina", "infantil", "pediatrico"),
    "tarda": ("tiempo", "aparicion"),
    "tardar": ("tiempo", "aparicion"),
    "demora": ("tiempo", "aparicion"),
    "demorar": ("tiempo", "aparicion"),
    "aparecer": ("aparicion", "tiempo"),
    # Sinónimos frecuentes en los nombres de vacunas.
    "antipoliomielitica": ("antipolio", "poliomielitis"),
    "antipolio": ("antipoliomielitica", "poliomielitis"),
    "poliomielitis": ("antipolio", "antipoliomielitica"),
    "neumococo": ("antineumococica", "neumococica"),
    "antineumococica": ("neumococo",),
    "neumococica": ("neumococo",),
    "papiloma": ("vph",),
    "vph": ("papiloma",),
    "antiamarilica": ("fiebre amarilla",),
    "fiebre amarilla": ("antiamarilica",),
    "influenza": ("gripe",),
    "gripe": ("influenza",),
    "antigripal": ("influenza", "gripe"),
    # Dosis de rescate (Anexo 2): la norma dice "rescate"; la pregunta, "recuperar",
    # "completar" o "no ha recibido".
    "recuperar": ("rescate",),
    "recuperable": ("rescate",),
    "completar": ("rescate", "incompleto"),
    "incompleto": ("rescate", "completar"),
    "atrasado": ("rescate",),
    "recibido": ("rescate",),
    "interrumpido": ("rescate", "reiniciar"),
    # La norma dice "serotipos" donde la pregunta dice "cepas".
    "cepa": ("serotipo", "serogrupo", "subgrupo"),
    "serotipo": ("cepa", "serogrupo"),
}


def expand_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for key, values in QUERY_SYNONYMS.items():
        key_tokens = set(tokenize(key))
        if key_tokens and key_tokens <= expanded:
            for value in values:
                expanded.update(tokenize(value))
    return expanded


def short_section(section: str) -> str:
    """Devuelve la última parte de una ruta de sección."""

    return section.split(" > ")[-1].strip() if section else ""
