"""Support for the Silero speech service."""
import logging

import os
import tempfile
import torch
import voluptuous as vol

from homeassistant.components.tts import CONF_LANG, PLATFORM_SCHEMA, Provider

_LOGGER = logging.getLogger(__name__)

SUPPORT_LANGUAGES = ["en"]

DEFAULT_LANG = "en"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {vol.Optional(CONF_LANG, default=DEFAULT_LANG): vol.In(SUPPORT_LANGUAGES)}
)


async def async_get_engine(
    hass, config, discovery_info=None
):  # pylint: disable=unused-argument
    """Set up Silero speech component."""
    return SileroProvider(hass, config[CONF_LANG])


class SileroProvider(Provider):
    """The Silero speech provider."""

    def __init__(self, hass, lang):
        """Init Silero service."""
        self.hass = hass
        self._lang = lang
        self.name = "Silero"

        # Load Silero model
        local_file = os.path.join(os.path.dirname(__file__), "models", "v3_en.pt")
        self.model = torch.package.PackageImporter(local_file).load_pickle(
            "tts_models", "model"
        )
        device = torch.device("cpu")  # pylint: disable=no-member
        self.model.to(device)

    @property
    def default_language(self):
        """Return the default language."""
        return self._lang

    @property
    def supported_languages(self):
        """Return list of supported languages."""
        return SUPPORT_LANGUAGES

    def get_tts_audio(self, message, language, options=None):
        """Generate WAV from Silero."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            fname = tmpf.name

        self.model.save_wav(
            message, speaker="en_0", sample_rate=48000, audio_path=fname
        )
        data = None
        try:
            with open(fname, "rb") as voice:
                data = voice.read()
        except OSError:
            _LOGGER.error("Error trying to read %s", fname)
            return (None, None)
        finally:
            os.remove(fname)

        if data:
            return ("wav", data)
        return (None, None)
