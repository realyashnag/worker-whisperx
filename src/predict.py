"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict
from runpod.serverless.utils import rp_cuda
from whisperx.asr import FasterWhisperPipeline
from whisperx import (
    load_model,
    load_audio,
    load_align_model,
    align
)
from rp_schema import *


class Predictor:
    """
    A Predictor class for the WhisperX
    """

    models: Dict[str, FasterWhisperPipeline] = {}
    model_dir: str = "/tmp"

    @property
    def device(self):
        return "cuda" if rp_cuda.is_available() else "cpu"

    @property
    def compute_type(self):
        return "float16" if rp_cuda.is_available() else "int8"

    def load_model(self, model_name) -> Tuple[str, FasterWhisperPipeline]:
        """
        Load the model from the weights folder
        """
    
        loaded_model = load_model(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.model_dir,
            asr_options = {
                "max_new_tokens": None,
                "clip_timestamps": None,
                "hallucination_silence_threshold": None,
            }
        )

        return model_name, loaded_model

    def setup(self):
        """
        Load the model into memory to make running multiple predictions efficient
        """

        model_names = ["tiny", "large-v2"]   #  ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(self.load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model

    def __call__(
        self,
        audio: str,
        model_name: str = "large-v2",
        language: Optional[str] = None,
        batch_size: int = 16,
    ) -> TranscriberOutput:
        """
        Run a single prediction on the model
        """

        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        audio = load_audio(audio)
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
            print_progress=True
        )
        model_a, metadata = load_align_model(
            language_code=result["language"],
            device=self.device
        )
        result = align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
            print_progress=True
        )
        return TranscriberOutput(**result)
