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
    # (regurgita, anafilaxia, dosis concreta…).
    "ano",
    "anos",
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
    tokens = normalize_for_match(text).split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return [stem(token) for token in tokens]


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
