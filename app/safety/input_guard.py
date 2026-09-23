import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.safety_model import SafetyModel


class InputGuard:

    def __init__(self, safety_model=None):
        self.safety_model = safety_model or SafetyModel()

    def analyze(self, text):
        result = self.safety_model.analyze(text)

        return {
            "allowed": result["decision"] == "ALLOW",
            "decision": result["decision"],
            "risk_level": result["risk_level"],
            "harmful_probability": result["harmful_probability"],
            "unharmful_probability": result["unharmful_probability"],
            "threshold": result["threshold"],
            "feature_850_activation": result["feature_850_activation"],
            "device": result["device"]
        }