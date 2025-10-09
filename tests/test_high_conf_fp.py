#!/usr/bin/env python3
"""
Test background error with high-confidence false positive.
"""

import pytest
import tidecv


@pytest.mark.parametrize(
    "scenario_name,predictions_data,ground_truths_data,pos_threshold,expected_conditions",
    [
        (
            "high_conf_fp_vs_low_conf_tp",
            {
                "name": "high_conf_fp",
                "detections": [
                    # FP (high conf)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [100, 100, 30, 30],
                    },
                    # TP (low conf)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [10, 10, 50, 50],
                    },
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "high_conf_gt",
                "ground_truths": [
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1"},
            },
            0.5,
            {
                "has_main_errors": True,
                "has_special_errors": True,
            },
        ),
        (
            "different_threshold",
            {
                "name": "high_conf_fp",
                "detections": [
                    # FP (high conf)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [100, 100, 30, 30],
                    },
                    # TP (low conf)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [10, 10, 50, 50],
                    },
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "high_conf_gt",
                "ground_truths": [
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1"},
            },
            0.3,
            {
                "has_main_errors": True,
                "has_special_errors": True,
            },
        ),
    ],
)
def test_high_conf_fp(
    scenario_name,
    predictions_data,
    ground_truths_data,
    pos_threshold,
    expected_conditions,
):
    """Test with false positive having higher confidence than true positive."""

    print(f"Testing {scenario_name} scenario...")

    # Create TIDE evaluator and data objects
    tide_evaluator = tidecv.TIDE()
    predictions = tidecv.Data(predictions_data["name"], max_dets=100)
    ground_truths = tidecv.Data(ground_truths_data["name"], max_dets=100)

    # Add predictions
    for detection in predictions_data["detections"]:
        predictions.add_detection(
            image_id=detection["image_id"],
            class_id=detection["class_id"],
            score=detection["score"],
            box=detection["box"],
        )

    # Add ground truths
    for gt in ground_truths_data["ground_truths"]:
        ground_truths.add_ground_truth(
            image_id=gt["image_id"], class_id=gt["class_id"], box=gt["box"]
        )

    # Add classes
    for class_id, class_name in predictions_data["classes"].items():
        predictions.add_class(class_id, class_name)
    for class_id, class_name in ground_truths_data["classes"].items():
        ground_truths.add_class(class_id, class_name)

    # Evaluate
    run = tide_evaluator.evaluate(
        ground_truths, predictions, pos_threshold=pos_threshold, mode=tidecv.TIDE.BOX
    )

    print(f"Original AP: {run.ap:.2f}")

    # Show main and special errors
    main_errors = run.fix_main_errors()
    special_errors = run.fix_special_errors()

    # Verify expected conditions
    has_main_errors = any(value > 0 for value in main_errors.values())
    has_special_errors = any(value > 0 for value in special_errors.values())

    assert (
        has_main_errors == expected_conditions["has_main_errors"]
    ), f"Expected main errors: {expected_conditions['has_main_errors']}, got: {has_main_errors}"
    assert (
        has_special_errors == expected_conditions["has_special_errors"]
    ), f"Expected special errors: {expected_conditions['has_special_errors']}, got: {has_special_errors}"

    print("\nMain errors:")
    for error_type, value in main_errors.items():
        if value > 0:
            print(f"  {error_type.__name__}: {value:.2f}")

    print("\nSpecial errors:")
    for error_type, value in special_errors.items():
        if value > 0:
            print(f"  {error_type.__name__}: {value:.2f}")

    # Full summary
    print("\nFull TIDE summary:")
    tide_evaluator.summarize()


if __name__ == "__main__":
    test_high_conf_fp()
