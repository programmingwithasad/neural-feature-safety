import os

import torch
import joblib

from transformers import AutoTokenizer, AutoModel

from .autoencoder import SparseAutoencoder


class SafetyModel:

    def __init__(
        self,
        threshold=0.50
    ):

        root_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.distilbert = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        ).to(self.device)

        self.distilbert.eval()

        self.sae = SparseAutoencoder(
            input_dim=768,
            hidden_dim=2048
        ).to(self.device)

        sae_path = os.path.join(
            root_dir,
            "models",
            "sparse_autoencoder_full.pt"
        )

        self.sae.load_state_dict(
            torch.load(
                sae_path,
                map_location=self.device,
                weights_only=False
            )
        )

        self.sae.eval()

        classifier_path = os.path.join(
            root_dir,
            "models",
            "final_safety_classifier.pkl"
        )

        self.classifier = joblib.load(
            classifier_path
        )

    def _extract_sparse_features(self, text):

        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.distilbert(
                **inputs
            )

            hidden_states = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(
                -1
            ).expand(
                hidden_states.size()
            ).float()

            masked_hidden_states = (
                hidden_states * mask
            )

            summed = masked_hidden_states.sum(
                dim=1
            )

            counts = mask.sum(
                dim=1
            )

            pooled = summed / counts

            _, sparse_features = self.sae(
                pooled
            )

        return sparse_features

    def _get_prediction(self, sparse_features):

        sparse_features_np = (
            sparse_features.detach().cpu().numpy()
        )

        probabilities = (
            self.classifier.predict_proba(
                sparse_features_np
            )[0]
        )

        unharmful_probability = probabilities[0]
        harmful_probability = probabilities[1]

        if harmful_probability >= self.threshold:

            risk_level = "HIGH RISK"
            decision = "BLOCK"

        else:

            risk_level = "LOW RISK"
            decision = "ALLOW"

        return (
            float(harmful_probability),
            float(unharmful_probability),
            risk_level,
            decision
        )

    def analyze(self, text):

        sparse_features = self._extract_sparse_features(
            text
        )

        feature_activation = sparse_features[
            0,
            850
        ].item()

        (
            harmful_probability,
            unharmful_probability,
            risk_level,
            decision
        ) = self._get_prediction(
            sparse_features
        )

        return {
            "harmful_probability":
                harmful_probability,
            "unharmful_probability":
                unharmful_probability,
            "risk_level":
                risk_level,
            "decision":
                decision,
            "threshold":
                self.threshold,
            "feature_850_activation":
                float(feature_activation),
            "device":
                str(self.device)
        }

    def explain(
        self,
        text,
        top_k=10
    ):

        sparse_features = self._extract_sparse_features(
            text
        )

        feature_values = (
            sparse_features[0]
            .detach()
            .cpu()
            .numpy()
        )

        (
            harmful_probability,
            unharmful_probability,
            risk_level,
            decision
        ) = self._get_prediction(
            sparse_features
        )

        scaler = self.classifier.named_steps["scaler"]

        classifier = self.classifier.named_steps["classifier"]

        scaled_features = (
            feature_values - scaler.mean_
        ) / scaler.scale_

        coefficients = classifier.coef_[0]

        contributions = (
            scaled_features * coefficients
        )

        feature_indices = list(
            range(len(feature_values))
        )

        top_activated_indices = sorted(
            feature_indices,
            key=lambda index: feature_values[index],
            reverse=True
        )[:top_k]

        top_positive_indices = sorted(
            feature_indices,
            key=lambda index: contributions[index],
            reverse=True
        )[:top_k]

        top_negative_indices = sorted(
            feature_indices,
            key=lambda index: contributions[index]
        )[:top_k]

        top_activated_features = []

        for index in top_activated_indices:

            top_activated_features.append(
                {
                    "feature": int(index),
                    "activation": float(
                        feature_values[index]
                    ),
                    "classifier_weight": float(
                        coefficients[index]
                    ),
                    "contribution": float(
                        contributions[index]
                    )
                }
            )

        top_positive_contributions = []

        for index in top_positive_indices:

            top_positive_contributions.append(
                {
                    "feature": int(index),
                    "activation": float(
                        feature_values[index]
                    ),
                    "classifier_weight": float(
                        coefficients[index]
                    ),
                    "contribution": float(
                        contributions[index]
                    )
                }
            )

        top_negative_contributions = []

        for index in top_negative_indices:

            top_negative_contributions.append(
                {
                    "feature": int(index),
                    "activation": float(
                        feature_values[index]
                    ),
                    "classifier_weight": float(
                        coefficients[index]
                    ),
                    "contribution": float(
                        contributions[index]
                    )
                }
            )

        return {
            "harmful_probability":
                harmful_probability,
            "unharmful_probability":
                unharmful_probability,
            "risk_level":
                risk_level,
            "decision":
                decision,
            "threshold":
                float(self.threshold),
            "feature_850_activation":
                float(feature_values[850]),
            "device":
                str(self.device),
            "feature_space":
                {
                    "input_dimensions": 768,
                    "latent_dimensions": 2048
                },
            "classifier":
                {
                    "type": "LogisticRegression",
                    "class_weight": "balanced",
                    "intercept": float(
                        classifier.intercept_[0]
                    )
                },
            "top_activated_features":
                top_activated_features,
            "top_positive_contributions":
                top_positive_contributions,
            "top_negative_contributions":
                top_negative_contributions
        }