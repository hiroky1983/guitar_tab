from dataclasses import dataclass

@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: float
