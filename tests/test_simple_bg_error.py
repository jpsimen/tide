#!/usr/bin/env python3
"""
Simple test to understand how background errors are calculated.
"""

import pytest
import tidecv


@pytest.mark.parametrize(
    "scenario_name,predictions_data,ground_truths_data,pos_threshold,expected_error_count,expected_improvements",
    [
        (
            "one_tp_one_fp",
            {
                "name": "simple_predictions",
                "detections": [
                    # TP - exactly matches ground truth
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    # FP - low IoU with ground truth
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "simple_gt",
                "ground_truths": [
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1"},
            },
            0.5,
            1,  # expected_error_count
            {"fp_fix_improves_ap": True},
        ),
        (
            "different_threshold",
            {
                "name": "simple_predictions",
                "detections": [
                    # TP - exactly matches ground truth
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    # FP - low IoU with ground truth
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "simple_gt",
                "ground_truths": [
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1"},
            },
            0.3,
            1,  # expected_error_count
            {"fp_fix_improves_ap": True},
        ),
        (
            "multiple_fps",
            {
                "name": "simple_predictions",
                "detections": [
                    # TP - exactly matches ground truth
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    # FP 1 - low IoU with ground truth
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },
                    # FP 2 - another false positive
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.7,
                        "box": [200, 200, 25, 25],
                    },
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "simple_gt",
                "ground_truths": [
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1"},
            },
            0.5,
            2,  # expected_error_count
            {"fp_fix_improves_ap": True},
        ),
    ],
)
def test_simple_background_error(
    scenario_name,
    predictions_data,
    ground_truths_data,
    pos_threshold,
    expected_error_count,
    expected_improvements,
):
    """Test with false positives to understand background error calculation."""

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

    print(f"Base AP: {run.ap:.2f}")
    print(f"Number of errors: {len(run.errors)}")

    # Assert expected error count
    assert (
        len(run.errors) == expected_error_count
    ), f"Expected {expected_error_count} errors, got {len(run.errors)}"

    # Show error details
    for i, error in enumerate(run.errors):
        error_name = type(error).__name__
        pred_id = getattr(error, "pred", {}).get("_id", "N/A")
        pred_score = getattr(error, "pred", {}).get("score", "N/A")
        print(f"  Error {i+1}: {error_name}, pred_id={pred_id}, score={pred_score}")

    # Debug main and special errors
    main_errors = run.fix_main_errors()
    special_errors = run.fix_special_errors()

    print("\nMain errors:")
    for error_type, value in main_errors.items():
        if value > 0:
            print(f"  {error_type.__name__}: {value:.2f}")

    print("\nSpecial errors:")
    for error_type, value in special_errors.items():
        if value > 0:
            print(f"  {error_type.__name__}: {value:.2f}")

    # Test false positive fix improvement
    if expected_improvements.get("fp_fix_improves_ap"):
        print("\nTesting false positive fix:")

        # Get original AP
        original_ap = run.ap_data.get_mAP()
        print(f"Original AP: {original_ap:.2f}")

        # Fix false positives by setting their score to 0
        fixed_fp_ap_data = run.fix_errors(
            transform=tidecv.errors.main_errors.FalsePositiveError.fix
        )
        fixed_ap = fixed_fp_ap_data.get_mAP()
        print(f"AP after fixing FPs: {fixed_ap:.2f}")

        improvement = fixed_ap - original_ap
        print(f"Improvement: {improvement:.2f}")

        # Assert that fixing false positives improves or maintains AP
        assert (
            improvement >= 0
        ), f"Expected non-negative improvement, got {improvement:.2f}"

    tide_evaluator.summarize()


if __name__ == "__main__":
    test_simple_background_error()
