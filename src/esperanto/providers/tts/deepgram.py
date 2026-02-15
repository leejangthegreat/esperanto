from sklearn.ensemble import VotingClassifier
import langchain_anthropic
from tornado.test.web_test import DecoratedStreamingRequestFlowControlTest
from sklearn.externals.array_api_compat.torch import AcceleratorError
from pexpect.ANSI import DoEnableScroll
from fsspec.conftest import m
from sympy.categories import preview_diagram
from sympy import preview
from pyasn1_modules.rfc2985 import gender
from triton.tools.compile import desc
from triton import language
from accelerate.commands.config.config import description
from typing import Dict, Tuple, Optional, List
import os

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

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **config
    ):
        """Initialize DeepGram TTS provider.

        Args:
            model_name: Name of the model to use
            api_key: DeepGram API key. Try to get from env:DEEPGRAM_API_KEY
                if not provided.
            base_url: Optional base URL for API
            **config: Additional configuration options including voice settings

        Raises:
            ValueError: API key not found in arg or env
        """
        api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepGram API key not provided. "
                "Set DEEPGRAM_API_KEY env or pass api_key parameter."
            )

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url or "https://api.deepgram.com/v1/speak",
            config=config
        )

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers fpr DeepGram API requests.

        Returns:
            Dict as post header
        """
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses.

        Raises:
            RuntimeError: Any DeepGram API error
        """
        if response.status_code >= 400:
            try:
                error_data = response.json()
                err_msg = (
                    error_data.get("details")
                    or error_data.get("message")
                    or f"HTTP {response.status_code}"
                )
            except Exception as _:
                # Parsing response failed
                err_msg = f"HTTP {response.status_code}: {response.text}"

            raise RuntimeError(
                f"DeepGram API error: {err_msg}"
            )

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="aura-2",
                owned_by="deepgram",
                context_window=None
            ),
            Model(
                id="aura",
                owned_by="deepgram",
                context_window=None
            )
        ]

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "deepgram"

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices."""
        # DeepGram TTS has predefined voices
        voices = {
            "amalthea": Voice(
                name="amalthea",
                id="amalthea",
                gender="FEMALE",
                description="Engaging, Natural, Cheerful",
                language_code="en-ph",
                accent="Filipino",
                age="young",
                use_case="Casual chat",
                preview_url="https://static.deepgram.com/examples/Aura-2-amalthea.wav"
            ),
            "andromeda": Voice(
                name="andromeda",
                id="andromeda",
                gender="FEMALE",
                description="Casual, Expressive, Comfortable",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service, IVR",
                preview_url="https://static.deepgram.com/examples/Aura-2-andromeda.wav"
            ),
            "apollo": Voice(
                name="apollo",
                id="apollo",
                gender="MALE",
                description="Confident, Comfortable, Casual",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Casual chat",
                preview_url="https://static.deepgram.com/examples/Aura-2-apollo.wav"
            ),
            "arcas": Voice(
                name="arcas",
                id="arcas",
                gender="MALE",
                description="Natural, Smooth, Clear, Comfortable",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service, casual chat",
                preview_url="https://static.deepgram.com/examples/Aura-2-arcas.wav"
            ),
            "aries": Voice(
                name="aries",
                id="aries",
                gender="MALE",
                description="Warm, Energetic, Caring",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Casual chat",
                preview_url="https://static.deepgram.com/examples/Aura-2-aries.wav"
            ),
            "asteria": Voice(
                name="asteria",
                id="asteria",
                gender="FEMALE",
                description="Clear, Confident, Knowledgeable, Energetic",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Advertising",
                preview_url="https://static.deepgram.com/examples/Aura-2-asteria.wav"
            ),
            "athena": Voice(
                name="athena",
                id="athena",
                gender="FEMALE",
                description="Calm, Smooth, Professional",
                language_code="en-us",
                accent="American",
                age="mature",
                use_case="Storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-athena.wav"
            ),
            "atlas": Voice(
                name="atlas",
                id="atlas",
                gender="MALE",
                description="Enthusiastic, Confident, Approachable, Friendly",
                language_code="en-us",
                accent="American",
                age="mature",
                use_case="Advertising",
                preview_url="https://static.deepgram.com/examples/Aura-2-atlas.wav"
            ),
            "aurora": Voice(
                name="aurora",
                id="aurora",
                gender="FEMALE",
                description="Cheerful, Expressive, Energetic",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Interview",
                preview_url="https://static.deepgram.com/examples/Aura-2-aurora.wav"
            ),
            "callista": Voice(
                name="callista",
                id="callista",
                gender="FEMALE",
                description="Clear, Energetic, Professional, Smooth",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="IVR",
                preview_url="https://static.deepgram.com/examples/Aura-2-callista.wav"
            ),
            "cora": Voice(
                name="cora",
                id="cora",
                gender="FEMALE",
                description="Smooth, Melodic, Caring",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-cora.wav"
            ),
            "cordelia": Voice(
                name="cordelia",
                id="cordelia",
                gender="FEMALE",
                description="Approachable, Warm, Polite",
                language_code="en-us",
                accent="American",
                age="young",
                use_case="Storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-cordelia.wav"
            ),
            "delia": Voice(
                name="delia",
                id="delia",
                gender="FEMALE",
                description="Casual, Friendly, Cheerful, Breathy",
                language_code="en-us",
                accent="American",
                age="young",
                use_case="Interview",
                preview_url="https://static.deepgram.com/examples/Aura-2-delia.wav"
            ),
            "draco": Voice(
                name="draco",
                id="draco",
                gender="MALE",
                description="Warm, Approachable, Trustworthy, Baritone",
                language_code="en-gb",
                accent="British",
                age="adult",
                use_case="Storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-draco.wav"
            ),
            "electra": Voice(
                name="electra",
                id="electra",
                gender="FEMALE",
                description="Professional, Engaging, Knowledgeable",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="IVR, advertising, customer service",
                preview_url="https://static.deepgram.com/examples/Aura-2-electra.wav"
            ),
            "harmonia": Voice(
                name="harmonia",
                id="harmonia",
                gender="FEMALE",
                description="Empathetic, Clear, Calm, Confident",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service",
                preview_url="https://static.deepgram.com/examples/Aura-2-harmonia.wav"
            ),
            "helena": Voice(
                name="helena",
                id="helena",
                gender="FEMALE",
                description="Caring, Natural, Positive, Friendly, Raspy",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="IVR, casual chat",
                preview_url="https://static.deepgram.com/examples/Aura-2-helena.wav"
            ),
            "hera": Voice(
                name="hera",
                id="hera",
                gender="FEMALE",
                description="Smooth, Warm, Professional",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Informative",
                preview_url="https://static.deepgram.com/examples/Aura-2-hera.wav"
            ),
            "hermes": Voice(
                name="hermes",
                id="hermes",
                gender="MALE",
                description="Expressive, Engaging, Professional",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Informative",
                preview_url="https://static.deepgram.com/examples/Aura-2-hermes.wav"
            ),
            "hyperion": Voice(
                name="hyperion",
                id="hyperion",
                gender="MALE",
                description="Caring, Warm, Empathetic",
                language_code="en-au",
                accent="Australian",
                age="adult",
                use_case="Interview",
                preview_url="https://static.deepgram.com/examples/Aura-2-hyperion.wav"
            ),
            "iris": Voice(
                name="iris",
                id="iris",
                gender="FEMALE",
                description="Cheerful, Positive, Approachable",
                language_code="en-us",
                accent="American",
                age="young",
                use_case="IVR, advertising, customer service",
                preview_url="https://static.deepgram.com/examples/Aura-2-iris.wav"
            ),
            "janus": Voice(
                name="janus",
                id="janus",
                gender="FEMALE",
                description="Southern, Smooth, Trustworthy",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-janus.wav"
            ),
            "juno": Voice(
                name="juno",
                id="juno",
                gender="FEMALE",
                description="Natural, Engaging, Melodic, Breathy",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Interview",
                preview_url="https://static.deepgram.com/examples/Aura-2-juno.wav"
            ),
            "jupiter": Voice(
                name="jupiter",
                id="jupiter",
                gender="MALE",
                description="Expressive, Knowledgeable, Baritone",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Informative",
                preview_url="https://static.deepgram.com/examples/Aura-2-jupiter.wav"
            ),
            "luna": Voice(
                name="luna",
                id="luna",
                gender="FEMALE",
                description="Friendly, Natural, Engaging",
                language_code="en-us",
                accent="American",
                age="young",
                use_case="IVR",
                preview_url="https://static.deepgram.com/examples/Aura-2-luna.wav"
            ),
            "mars": Voice(
                name="mars",
                id="mars",
                gender="MALE",
                description="Smooth, Patient, Trustworthy, Baritone",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service",
                preview_url="https://static.deepgram.com/examples/Aura-2-mars.wav"
            ),
            "minerva": Voice(
                name="minerva",
                id="minerva",
                gender="FEMALE",
                description="Positive, Friendly, Natural",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-minerva.wav"
            ),
            "neptune": Voice(
                name="neptune",
                id="neptune",
                gender="MALE",
                description="Professional, Patient, Polite",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service",
                preview_url="https://static.deepgram.com/examples/Aura-2-neptune.wav"
            ),
            "odysseus": Voice(
                name="odysseus",
                id="odysseus",
                gender="MALE",
                description="Calm, Smooth, Comfortable, Professional",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Advertising",
                preview_url="https://static.deepgram.com/examples/Aura-2-odysseus.wav"
            ),
            "ophelia": Voice(
                name="ophelia",
                id="ophelia",
                gender="FEMALE",
                description="Expressive, Enthusiastic, Cheerful",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Interview",
                preview_url="https://static.deepgram.com/examples/Aura-2-ophelia.wav"
            ),
            "orion": Voice(
                name="orion",
                id="orion",
                gender="MALE",
                description="Approachable, Comfortable, Calm, Polite",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Informative",
                preview_url="https://static.deepgram.com/examples/Aura-2-orion.wav"
            ),
            "orpheus": Voice(
                name="orpheus",
                id="orpheus",
                gender="MALE",
                description="Professional, Clear, Confident, Trustworthy",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service, storytelling",
                preview_url="Customer service, storytelling"
            ),
            "pandora": Voice(
                name="pandora",
                id="pandora",
                gender="FEMALE",
                description="Smooth, Calm, Melodic, Breathy",
                language_code="en-gb",
                accent="British",
                age="adult",
                use_case="IVR, informative",
                preview_url="https://static.deepgram.com/examples/Aura-2-pandora.wav"
            ),
            "phoebe": Voice(
                name="phoebe",
                id="phoebe",
                gender="FEMALE",
                description="Energetic, Warm, Casual",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service",
                preview_url="https://static.deepgram.com/examples/Aura-2-phoebe.wav"
            ),
            "pluto": Voice(
                name="pluto",
                id="pluto",
                gender="MALE",
                description="Smooth, Calm, Empathetic, Baritone",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Interview, storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-pluto.wav"
            ),
            "saturn": Voice(
                name="saturn",
                id="saturn",
                gender="MALE",
                description="Knowledgeable, Confident, Baritone",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service",
                preview_url="https://static.deepgram.com/examples/Aura-2-saturn.wav"
            ),
            "selene": Voice(
                name="selene",
                id="selene",
                gender="FEMALE",
                description="Expressive, Engaging, Energetic",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Informative",
                preview_url="https://static.deepgram.com/examples/Aura-2-selene.wav"
            ),
            "thalia": Voice(
                name="thalia",
                id="thalia",
                gender="FEMALE",
                description="Clear, Confident, Energetic, Enthusiastic",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Casual chat, customer service, IVR",
                preview_url="https://static.deepgram.com/examples/Aura-2-thalia.wav"
            ),
            "theia": Voice(
                name="theia",
                id="theia",
                gender="FEMALE",
                description="Expressive, Polite, Sincere",
                language_code="en-au",
                accent="Australian",
                age="adult",
                use_case="Informative",
                preview_url="https://static.deepgram.com/examples/Aura-2-theia.wav"
            ),
            "vesta": Voice(
                name="vesta",
                id="vesta",
                gender="FEMALE",
                description="Natural, Expressive, Patient, Empathetic",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="Customer service, interview, storytelling",
                preview_url="https://static.deepgram.com/examples/Aura-2-vesta.wav"
            ),
            "zeus": Voice(
                name="zeus",
                id="zeus",
                gender="MALE",
                description="Deep, Trustworthy, Smooth",
                language_code="en-us",
                accent="American",
                age="adult",
                use_case="IVR",
                preview_url="https://static.deepgram.com/examples/Aura-2-zeus.wav"
            )
        }

        return voices
