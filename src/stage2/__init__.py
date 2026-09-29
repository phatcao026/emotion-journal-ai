"""
src/stage2/
-----------
Dedicated Stage 2 Multimodal Emotion Recognition (MER) Pipeline.
Isolated from the legacy Stage 1 / Stage 1.5 pipeline to ensure baseline preservation
and reproducible side-by-side benchmarking.
"""

from src.stage2.melody_extractor import ExplicitMelodyExtractor
from src.stage2.gated_fusion_stage2 import Stage2GatedFusion
from src.stage2.hierarchical_mer_stage2 import Stage2HierarchicalMERModel, Stage2ModelOutput
from src.stage2.ldam_loss import LDAMLoss
from src.stage2.uncertainty_loss import MultiTaskUncertaintyLoss

__all__ = [
    "ExplicitMelodyExtractor",
    "Stage2GatedFusion",
    "Stage2HierarchicalMERModel",
    "Stage2ModelOutput",
    "LDAMLoss",
    "MultiTaskUncertaintyLoss",
]
