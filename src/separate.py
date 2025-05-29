import os
from audio_separator.separator import Separator as _AudioSeparator
from .audio import Audio
import tempfile


class Separator:
    """Wraps the audio separator to provide a simpler interface."""

    def __init__(self, model_name: str = ""):
        self.separator = _AudioSeparator(log_level=40)  # Set log level to ERROR to suppress debug messages
        if model_name:
            self.separator.load_model(model_name)
        else:
            # Default model
            self.separator.load_model()

    def separate(self, audio: Audio) -> list[Audio]:
        """Separate the audio into vocals and accompaniment."""
        custom_names = {
            "vocals": "vocals",
            "drums": "drums",
            "bass": "bass",
            "other": "other",
            "instrumental": "instrumental",
            "accompaniment": "accompaniment",
            "piano": "piano",
            "guitar": "guitar",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            # Has to be adjusted on the fly
            audio_file = os.path.join(temp_dir, "input.wav")
            audio.save(audio_file)
            self.separator.model_instance.output_dir = temp_dir  # type: ignore
            output_paths = self.separator.separate(audio_file, custom_names)
            output_paths = sorted(output_paths)
            return [Audio.load(os.path.join(temp_dir, filename)) for filename in output_paths]
