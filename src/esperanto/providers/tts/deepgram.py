from typing import Dict, List, Tuple

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice


class DeepGramTextToSpeechModel(TextToSpeechModel):
    """DeepGram Text-To-Speech provider implementation.

    Supports multiple models including:
    - Aura: English-only model
    - Aura 2: Multilingual model
    """

    DEFAULT_MODEL = "aura-2"
    DEFAULT_VOICE: Dict[str, Dict[str, str]] = {
        "aura-2": {
            "en": "thalia",
            "es": "celeste",
            "nl": "rhea",
            "fr": "agathe",
            "de": "julius",
            "it": "livia",
            "ja": "fujin"
        },
        "aura": {
            "en": "asteria"
        }
    }
    PROVIDER = "deepgram"
    MODEL_LAN_VOICE_MAP: Dict[str, Dict[str, Tuple[str, ...]]] = {
        "aura-2": {
            "en": (
                "amalthea", "andromeda", "apollo", "arcas", "aries",
                "asteria", "athena", "atlas", "aurora", "callista",
                "cora", "cordelia", "delia", "draco", "electra",
                "harmonia", "helena", "hera", "hermes", "hyperion",
                "iris", "janus", "juno", "jupiter", "luna", "mars",
                "minerva", "neptune", "odysseus", "ophelia", "orion",
                "orpheus", "pandora", "phoebe", "pluto", "saturn",
                "selene", "thalia", "theia", "vesta", "zeus"
            ),
            "es": (  # Spanish
                "sirio", "nestor", "carina", "celeste", "alvaro", "diana",
                "aquila", "selena", "estrella", "javier", "agustina", "antonia",
                "gloria", "luciano", "olivia", "silvia", "valerio"
            ),
            "nl": (  # Dutch
                "beatrix", "daphne", "cornelia", "sander", "hestia", "lars",
                "roman", "rhea", "leda"
            ),
            "fr": (
                "agathe", "hector"
            ),
            "de": (
                "elara", "aurelia", "lara", "julius", "fabian", "kara",
                "viktoria"
            ),
            "it": (
                "melia", "elio", "flavio", "maia", "cinzia", "cesare", "livia",
                "perseo", "dionisio", "demetra"
            ),
            "ja": (
                "uzume", "ebisu", "fujin", "izanami", "ama"
            )
        },
        "aura": {
            "en": (
                "asteria", "luna", "stella", "athena", "hera", "orion", "arcas",
                "perseus", "angus", "orpheus", "helios", "zeus"
            )
        }
    }

