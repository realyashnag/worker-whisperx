from typing import Optional, List
from pydantic import BaseModel


class WordSegment(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None

class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: List[WordSegment] = []

class TranscriberOutput(BaseModel):
    segments: List[Segment] = []
    word_segments: List[WordSegment] = []



INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': True,
        'default': ''
    },
    'model_name': {
        'type': str,
        'required': False,
        'default': "large-v2"
    },
    'language': {
        'type': str,
        'required': False,
        'default': None
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 16
    }
}
