from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional



@dataclass
class Extracted_Parameters:
    """
    Container for all of our extracted parameters
    Will use to help predict template and populate chosen template
    """

    min_val: Optional[int] = None
    max_val: Optional[int] = None
    games: List[str] = None
    matchups: List[str] = None
    byes: List[str] = None
    rounds: List[str] = None
    venues: List[str] = None
    networks: List[str] = None
    teams: List[str] = None
    round1: Optional[str] = None
    round2: Optional[str] = None
    k: Optional[int] = None
    m: Optional[int] = None
    home_away_bye_active: Optional[str] = None
    each_or_all: Optional[str] = None
