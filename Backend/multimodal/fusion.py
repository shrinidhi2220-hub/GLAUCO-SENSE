"""Simple late-fusion utility for the shared integration stage.

Keep each modality model independent and fuse calibrated probabilities only after
both standalone models have been validated.
"""


def weighted_average(image_probability: float, clinical_probability: float, image_weight: float = 0.7) -> float:
    if not 0 <= image_weight <= 1:
        raise ValueError("image_weight must be between 0 and 1")
    return image_weight * image_probability + (1 - image_weight) * clinical_probability
