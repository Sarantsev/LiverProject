"""Liver-tumor CDSS built on top of SegVol.

The package adds task-specific heads on top of SegVol (a 3D foundation model):
- multiphase: fusion of 4-phase CT
- classifier: tumor-type classification head (multi-class differential diagnosis)
- (further) data / train / radiomics
"""

__version__ = "0.1.0"
