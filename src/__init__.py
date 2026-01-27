"""
PROVE: Probabilistic Reasoning Over Visual Evidence

Unified pipeline that runs both probabilistic and deterministic modes
with shared evidence collection to isolate the effect of perception uncertainty.
"""

from .prove import PROVE
from .core.types import UnifiedResult, SharedEvidence, ModeResult

__all__ = ['PROVE', 'UnifiedResult', 'SharedEvidence', 'ModeResult']
__version__ = '0.2.0'
