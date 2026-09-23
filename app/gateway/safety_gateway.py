import json

from app.llm.llama_client import LlamaClient
from app.safety.input_guard import InputGuard
from app.safety.output_guard import OutputGuard


class SafetyGateway:
    def __init__(self):
        print("Initializing Neural Feature Safety Gateway...")

        self.input_guard = InputGuard()

        self.output_guard = OutputGuard(
            safety_model=self.input_guard.safety_model
        )

        self.llama = LlamaClient()

        self.max_recovery_attempts = 2

        print("Neural Feature Safety Gateway ready.")

    def _generate_safe_alternative(self, original_prompt):
        recovery_prompt = (
            "Provide a safe and helpful response to the user's request. "
            "Do not provide instructions that could enable harm, illegal "
            "activity, violence, exploitation, or unauthorized access. "
            "If the request is unsafe, briefly explain that you cannot "
            "provide the requested harmful instructions and redirect the "
            "user toward a legitimate and safe alternative.\n\n"
            f"User request:\n{original_prompt}"
        )

        return self.llama.generate(
            recovery_prompt,
            max_new_tokens=200,
            temperature=0.2
        )

    def _generate_normal_response(self, prompt):
        return self.llama.generate(
            prompt,
            max_new_tokens=200,
            temperature=0.2
        )

    def _validate_output(self, response):
        return self.output_guard.analyze(response)

    def _build_fallback(self):
        fallback_candidates = [
            "I can't provide that request. I can help with a safe and lawful alternative.",
            "I can't provide instructions that could cause harm. I can help with a safe alternative.",
            "I can't assist with that request. I can help with information for a legitimate and safe purpose."
        ]

        for response in fallback_candidates:
            output_result = self._validate_output(response)

            if output_result["allowed"]:
                return response, output_result

        return None, output_result

    def _result(
        self,
        response,
        input_result,
        output_result,
        path,
        recovery_attempts
    ):
        return {
            "response": response,
            "input_safety": input_result,
            "output_safety": output_result,
            "path": path,
            "recovery_attempts": recovery_attempts
        }

    def explain_with_llama(self, prompt, top_k=8):
        prompt = prompt.strip()

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        evidence = self.input_guard.safety_model.explain(
            prompt,
            top_k=top_k
        )

        explanation_evidence = {
            "harmful_probability": evidence["harmful_probability"],
            "unharmful_probability": evidence["unharmful_probability"],
            "risk_level": evidence["risk_level"],
            "decision": evidence["decision"],
            "threshold": evidence["threshold"],
            "feature_850_activation": evidence["feature_850_activation"],
            "top_positive_contributions": evidence[
                "top_positive_contributions"
            ][:top_k],
            "top_negative_contributions": evidence[
                "top_negative_contributions"
            ][:top_k]
        }

        evidence_text = json.dumps(
            explanation_evidence,
            indent=2
        )

        explanation_prompt = f"""
You are the explanation writer for an AI safety analysis system.

The safety classifier has ALREADY made the safety decision.
You must NOT make a new safety decision.

Your job is only to explain the classifier evidence in clear,
professional language.

Rules:

1. Do not override the supplied decision.
2. Do not invent meanings for SAE feature IDs.
3. SAE feature IDs are internal model features unless their semantic
   meaning has been experimentally demonstrated.
4. Treat feature contributions as classifier evidence, not proof of
   causation.
5. Positive contribution means that the feature pushed the classifier
   toward the harmful class.
6. Negative contribution means that the feature pushed the classifier
   away from the harmful class.
7. Feature 850 is observational only. Do not claim that it causes
   the decision.
8. Explain the probability relative to the configured threshold.
9. Do not repeat or expand potentially harmful instructions from the
   original prompt.
10. Keep the explanation concise and suitable for a professional
    AI safety dashboard.
11. Do not say that Llama detected the harmful content.
12. The Neural Feature Safety classifier performed the safety analysis.

Original user prompt:
{prompt}

Classifier evidence:
{evidence_text}

Write a concise explanation containing:

- The final safety decision.
- The harmful probability and threshold.
- Whether the classifier evidence pushed toward or away from the
  harmful class.
- A short explanation of the strongest positive and negative
  classifier contributions.
- A short caveat that SAE feature IDs do not automatically have
  human-interpretable meanings.
"""

        llama_result = self.llama.generate(
            explanation_prompt,
            max_new_tokens=180,
            temperature=0.1
        )

        explanation = llama_result["response"].strip()

        output_result = self._validate_output(explanation)

        if output_result["allowed"]:
            return {
                "evidence": evidence,
                "llama_explanation": explanation,
                "explanation_source": "llama",
                "generation_time": llama_result["generation_time"],
                "explanation_output_safety": output_result
            }

        fallback = self._build_explanation_fallback(evidence)

        fallback_output = self._validate_output(fallback)

        if fallback_output["allowed"]:
            return {
                "evidence": evidence,
                "llama_explanation": fallback,
                "explanation_source": "deterministic_fallback",
                "generation_time": llama_result["generation_time"],
                "explanation_output_safety": fallback_output
            }

        raise RuntimeError(
            "No safe explanation passed the output safety check."
        )

    def _build_explanation_fallback(self, evidence):
        probability = evidence["harmful_probability"]
        threshold = evidence["threshold"]
        decision = evidence["decision"]

        if decision == "BLOCK":
            return (
                f"The Neural Feature Safety classifier assigned a harmful "
                f"probability of {probability:.3f}, which is above the "
                f"configured blocking threshold of {threshold:.2f}. "
                f"The classifier therefore returned BLOCK. The displayed "
                f"SAE features represent model evidence used by the "
                f"classifier; their numerical IDs should not be interpreted "
                f"as human-readable concepts without separate validation."
            )

        return (
            f"The Neural Feature Safety classifier assigned a harmful "
            f"probability of {probability:.3f}, which is below the "
            f"configured blocking threshold of {threshold:.2f}. "
            f"The classifier therefore returned ALLOW. The displayed SAE "
            f"features represent model evidence used by the classifier; "
            f"their numerical IDs should not be interpreted as "
            f"human-readable concepts without separate validation."
        )

        def explain_with_llama(self, prompt, top_k=8):
            prompt = prompt.strip()

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        evidence = self.input_guard.safety_model.explain(
            prompt,
            top_k=top_k
        )

        explanation_evidence = {
            "harmful_probability": evidence["harmful_probability"],
            "unharmful_probability": evidence["unharmful_probability"],
            "risk_level": evidence["risk_level"],
            "decision": evidence["decision"],
            "threshold": evidence["threshold"],
            "feature_850_activation": evidence["feature_850_activation"],
            "top_positive_contributions": evidence[
                "top_positive_contributions"
            ][:top_k],
            "top_negative_contributions": evidence[
                "top_negative_contributions"
            ][:top_k]
        }

        import json

        evidence_text = json.dumps(
            explanation_evidence,
            indent=2
        )

        explanation_prompt = f"""
You are the explanation writer for an AI safety analysis system.

The Neural Feature Safety classifier has ALREADY made the safety
decision.

You must NOT make a new safety decision.

Your job is ONLY to explain the supplied classifier evidence.

Rules:

1. Never override the supplied safety decision.
2. Never invent meanings for SAE feature IDs.
3. SAE feature IDs are internal model representations unless their
semantic meaning has been experimentally demonstrated.
4. Treat feature contributions as classifier evidence, not proof
of causation.
5. A positive contribution means the feature pushed the classifier
toward the harmful class.
6. A negative contribution means the feature pushed the classifier
away from the harmful class.
7. Feature 850 is observational only.
8. Do not claim Feature 850 causes the decision.
9. Explain the harmful probability relative to the configured threshold.
10. Do not repeat or expand harmful instructions from the original prompt.
11. Do not say that Llama detected the harmful content.
12. The Neural Feature Safety classifier performed the safety analysis.
13. Keep the explanation concise and professional.

Original user prompt:
{prompt}

Classifier evidence:
{evidence_text}

Write a concise explanation that includes:

- The final safety decision.
- The harmful probability and configured threshold.
- Whether the classifier evidence pushed toward or away from the
harmful class.
- The strongest positive classifier contributions.
- The strongest negative classifier contributions.
- A caveat that SAE feature IDs do not automatically represent
human-interpretable concepts.
"""

        llama_result = self.llama.generate(
            explanation_prompt,
            max_new_tokens=180,
            temperature=0.1
        )

        explanation = llama_result["response"].strip()

        output_result = self._validate_output(explanation)

        if output_result["allowed"]:
            return {
                "evidence": evidence,
                "llama_explanation": explanation,
                "explanation_source": "llama",
                "generation_time": llama_result["generation_time"],
                "explanation_output_safety": output_result
            }

        fallback = self._build_explanation_fallback(evidence)

        fallback_output = self._validate_output(fallback)

        if fallback_output["allowed"]:
            return {
                "evidence": evidence,
                "llama_explanation": fallback,
                "explanation_source": "deterministic_fallback",
                "generation_time": llama_result["generation_time"],
                "explanation_output_safety": fallback_output
            }

        raise RuntimeError(
            "No safe explanation passed the output safety check."
        )

    def _build_explanation_fallback(self, evidence):
        probability = evidence["harmful_probability"]
        threshold = evidence["threshold"]
        decision = evidence["decision"]

        if decision == "BLOCK":
            return (
                f"The Neural Feature Safety classifier assigned a harmful "
                f"probability of {probability:.3f}, which is above the "
                f"configured blocking threshold of {threshold:.2f}. "
                f"The classifier therefore returned BLOCK. The displayed "
                f"SAE features represent model evidence used by the "
                f"classifier; their numerical IDs should not be interpreted "
                f"as human-readable concepts without separate validation."
            )

        return (
            f"The Neural Feature Safety classifier assigned a harmful "
            f"probability of {probability:.3f}, which is below the "
            f"configured blocking threshold of {threshold:.2f}. "
            f"The classifier therefore returned ALLOW. The displayed SAE "
            f"features represent model evidence used by the classifier; "
            f"their numerical IDs should not be interpreted as "
            f"human-readable concepts without separate validation."
        )

    
    def chat(self, prompt):
        prompt = prompt.strip()

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        input_result = self.input_guard.analyze(prompt)

        if input_result["decision"] == "BLOCK":
            response_result = self._generate_safe_alternative(prompt)
            response = response_result["response"]
            output_result = self._validate_output(response)

            if output_result["allowed"]:
                return self._result(
                    response,
                    input_result,
                    output_result,
                    "blocked_input_safe_alternative",
                    0
                )

            for attempt in range(1, self.max_recovery_attempts + 1):
                recovery_result = self._generate_safe_alternative(prompt)
                response = recovery_result["response"]
                output_result = self._validate_output(response)

                if output_result["allowed"]:
                    return self._result(
                        response,
                        input_result,
                        output_result,
                        "blocked_input_recovery",
                        attempt
                    )

            fallback_response, fallback_result = self._build_fallback()

            if fallback_response is not None:
                return self._result(
                    fallback_response,
                    input_result,
                    fallback_result,
                    "deterministic_fallback",
                    self.max_recovery_attempts
                )

            raise RuntimeError(
                "No safe response passed the output safety check."
            )

        generation_result = self._generate_normal_response(prompt)
        response = generation_result["response"]
        output_result = self._validate_output(response)

        if output_result["allowed"]:
            return self._result(
                response,
                input_result,
                output_result,
                "normal_generation",
                0
            )

        for attempt in range(1, self.max_recovery_attempts + 1):
            recovery_result = self._generate_safe_alternative(prompt)
            response = recovery_result["response"]
            output_result = self._validate_output(response)

            if output_result["allowed"]:
                return self._result(
                    response,
                    input_result,
                    output_result,
                    "output_recovery",
                    attempt
                )

        fallback_response, fallback_result = self._build_fallback()

        if fallback_response is not None:
            return self._result(
                fallback_response,
                input_result,
                fallback_result,
                "deterministic_fallback",
                self.max_recovery_attempts
            )

        raise RuntimeError(
            "No safe response passed the output safety check."
        )