"""
WorldJEPA — Open, trustworthy world models.
CertesLabs · 2026 · MIT License
"""

__version__ = "0.1.0"
__author__ = "CertesLabs"
__license__ = "MIT"

from worldjepa.model import WorldJEPA, WorldJEPAEncoder, WorldJEPAPredictor
from worldjepa.sigreg import SIGReg, SIGRegFast

__all__ = [
    "WorldJEPA",
    "WorldJEPAEncoder",
    "WorldJEPAPredictor",
    "SIGReg",
    "SIGRegFast",
]
