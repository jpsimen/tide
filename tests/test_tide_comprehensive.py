#!/usr/bin/env python3
"""
Comprehensive test script to verify TIDE fix for images without ground truth.
"""

import sys
import pytest
import tidecv


@pytest.mark.parametrize(
    "scenario_name,predictions_data,ground_truths_data,expected_bg_errors,expected_missed_errors,expected_box_errors,expected_total_errors",
    [
        (
            "only_predictions_no_gt",
            {
                "name": "only_predictions",
                "detections": [
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    {
                        "image_id": 1,
                        "class_id": 2,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            {
                "name": "empty_gt",
                "ground_truths": [],
                "classes": {1: "class1", 2: "class2"},
            },
            2,  # expected_bg_errors
            0,  # expected_missed_errors
            0,  # expected_box_errors
            2,  # expected_total_errors
        ),
        (
            "mixed_scenario",
            {
                "name": "mixed_predictions",
                "detections": [
                    # Image 0: predictions only ==> should count as background errors
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.99,
                        "box": [10, 10, 50, 50],
                    },
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.98,
                        "box": [20, 20, 40, 40],
                    },
                    # Image 1: true positive prediction and GT
                    {
                        "image_id": 1,
                        "class_id": 1,
                        "score": 0.95,
                        "box": [11, 10, 49, 50],
                    },
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "mixed_gt",
                "ground_truths": [
                    # Image 1: GT for true positive
                    {"image_id": 1, "class_id": 1, "box": [10, 10, 50, 50]},
                    # Image 2: GT only (no predictions) ==> should count as a missed detection
                    {"image_id": 2, "class_id": 1, "box": [30, 30, 60, 60]},
                ],
                "classes": {1: "class1"},
            },
            2,  # expected_bg_errors
            1,  # expected_missed_errors
            0,  # expected_box_errors
            3,  # expected_total_errors
        ),
    ],
)
def test_comprehensive_tide_scenarios(
    scenario_name,
    predictions_data,
    ground_truths_data,
    expected_bg_errors,
    expected_missed_errors,
    expected_box_errors,
    expected_total_errors,
):
    """Test multiple scenarios with TIDE using parameterized data."""

    print(f"\n=== Testing {scenario_name} ===")

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
        ground_truths, predictions, pos_threshold=0.5, mode=tidecv.TIDE.BOX
    )

    # Count different error types
    bg_errors = [
        e
        for e in run.errors
        if isinstance(e, tidecv.errors.main_errors.BackgroundError)
    ]
    missed_errors = [
        e for e in run.errors if isinstance(e, tidecv.errors.main_errors.MissedError)
    ]
    box_errors = [
        e for e in run.errors if isinstance(e, tidecv.errors.main_errors.BoxError)
    ]

    # Assertions
    assert (
        len(bg_errors) == expected_bg_errors
    ), f"Expected {expected_bg_errors} background errors, got {len(bg_errors)}"
    assert (
        len(missed_errors) == expected_missed_errors
    ), f"Expected {expected_missed_errors} missed errors, got {len(missed_errors)}"
    assert (
        len(box_errors) == expected_box_errors
    ), f"Expected {expected_box_errors} box errors, got {len(box_errors)}"
    assert (
        len(run.errors) == expected_total_errors
    ), f"Expected {expected_total_errors} total errors, got {len(run.errors)}"

    # Additional checks for mixed scenario
    if scenario_name == "mixed_scenario":
        main_errors = tide_evaluator.get_main_errors()
        assert main_errors["mixed_predictions"]["Bkg"] > 0.0
        special_errors = tide_evaluator.get_special_errors()
        assert special_errors["mixed_predictions"]["FalsePos"] > 0.0


if __name__ == "__main__":
    success = test_comprehensive_tide_scenarios()
    sys.exit(0 if success else 1)
