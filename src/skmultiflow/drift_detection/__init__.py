"""
The :mod:`skmultiflow.drift_detection` module includes methods for Concept Drift Detection.
"""

from .adwin import ADWIN
from .ddm import DDM
from .eddm import EDDM
from .page_hinkley import PageHinkley
from .hddm_a import HDDM_A
from .hddm_w import HDDM_W
from .kswin import KSWIN
from .drift_detection_method_classifier import DriftDetectionMethodClassifier
from .adwin_change_detector import ADWINChangeDetector
from .ewma import EWMA

__all__ = ["ADWIN", "DDM", "EDDM", "PageHinkley", "HDDM_A", "HDDM_W", "KSWIN", "DriftDetectionMethodClassifier", "ADWINChangeDetector", "EWMA"]
