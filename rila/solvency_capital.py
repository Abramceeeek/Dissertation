import numpy as np
from rila.hedging import conditional_tail_expectation

def calculate_scr(hedged_pnl: np.ndarray, confidence: float = 0.995) -> float:
    return conditional_tail_expectation(hedged_pnl, 1 - confidence) 