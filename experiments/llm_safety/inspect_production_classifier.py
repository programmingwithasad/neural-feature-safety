import os
import joblib


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

model_path = os.path.join(
    project_root,
    "models",
    "final_safety_classifier.pkl"
)


classifier = joblib.load(model_path)


print("=" * 70)
print("PRODUCTION CLASSIFIER INSPECTION")
print("=" * 70)

print(f"Path: {model_path}")
print(f"Type: {type(classifier)}")

print("\nModel representation:")
print(classifier)

print("\nAttributes:")

for attribute in [
    "classes_",
    "coef_",
    "intercept_",
    "n_features_in_",
    "feature_names_in_"
]:

    if hasattr(classifier, attribute):

        value = getattr(classifier, attribute)

        if attribute == "coef_":
            print(
                f"{attribute}: shape={value.shape}"
            )

        elif attribute == "intercept_":
            print(
                f"{attribute}: shape={value.shape}"
            )

        elif attribute == "feature_names_in_":
            print(
                f"{attribute}: present"
            )

        else:
            print(
                f"{attribute}: {value}"
            )


if hasattr(classifier, "named_steps"):

    print("\nPipeline steps:")

    for name, step in classifier.named_steps.items():

        print(
            f"{name}: {type(step)}"
        )

        if hasattr(step, "n_features_in_"):
            print(
                f"  n_features_in_: "
                f"{step.n_features_in_}"
            )

        if hasattr(step, "classes_"):
            print(
                f"  classes_: "
                f"{step.classes_}"
            )

        if hasattr(step, "class_weight"):
            print(
                f"  class_weight: "
                f"{step.class_weight}"
            )

        if hasattr(step, "C"):
            print(
                f"  C: "
                f"{step.C}"
            )

        if hasattr(step, "max_iter"):
            print(
                f"  max_iter: "
                f"{step.max_iter}"
            )

        if hasattr(step, "random_state"):
            print(
                f"  random_state: "
                f"{step.random_state}"
            )

print("\nInspection completed.")