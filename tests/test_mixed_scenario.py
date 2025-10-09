#!/usr/bin/env python3
"""
Test special errors with a mix of scenarios to understand the expected behavior.
"""

import pytest
import tidecv


@pytest.mark.parametrize(
    "scenario_name,predictions_data,ground_truths_data,pos_threshold,expected_error_types,expected_total_errors",
    [
        (
            "standard_mixed_scenario",
            {
                "name": "mixed_predictions",
                "detections": [
                    # Image 0: One correct detection, one false positive
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },  # Should be TP
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },  # Should be FP
                    # Image 1: False positive only (no ground truth)
                    {
                        "image_id": 1,
                        "class_id": 1,
                        "score": 0.7,
                        "box": [200, 200, 40, 40],
                    },  # Should be FP
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "mixed_gt",
                "ground_truths": [
                    # Image 0: Ground truth for correct detection
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                    # Image 2: Missed detection (ground truth only)
                    {"image_id": 2, "class_id": 1, "box": [300, 300, 60, 60]},
                ],
                "classes": {1: "class1"},
            },
            0.5,
            {"BackgroundError": 2, "MissedError": 1},  # expected_error_types
            3,  # expected_total_errors
        ),
        (
            "higher_threshold_scenario",
            {
                "name": "mixed_predictions",
                "detections": [
                    # Image 0: One correct detection, one false positive
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },  # Should be TP
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },  # Should be FP
                    # Image 1: False positive only (no ground truth)
                    {
                        "image_id": 1,
                        "class_id": 1,
                        "score": 0.7,
                        "box": [200, 200, 40, 40],
                    },  # Should be FP
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "mixed_gt",
                "ground_truths": [
                    # Image 0: Ground truth for correct detection
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                    # Image 2: Missed detection (ground truth only)
                    {"image_id": 2, "class_id": 1, "box": [300, 300, 60, 60]},
                ],
                "classes": {1: "class1"},
            },
            0.7,
            {"BackgroundError": 2, "MissedError": 1},  # expected_error_types
            3,  # expected_total_errors
        ),
        (
            "all_false_positives",
            {
                "name": "mixed_predictions",
                "detections": [
                    # All false positives - no matching ground truths
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [100, 100, 30, 30],
                    },
                    {
                        "image_id": 1,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [200, 200, 40, 40],
                    },
                    {
                        "image_id": 2,
                        "class_id": 1,
                        "score": 0.7,
                        "box": [300, 300, 50, 50],
                    },
                ],
                "classes": {1: "class1"},
            },
            {
                "name": "mixed_gt",
                "ground_truths": [
                    # Ground truths that don't match any predictions
                    {"image_id": 0, "class_id": 1, "box": [10, 10, 50, 50]},
                    {"image_id": 1, "class_id": 1, "box": [20, 20, 50, 50]},
                ],
                "classes": {1: "class1"},
            },
            0.5,
            {"BackgroundError": 3, "MissedError": 2},  # expected_error_types
            5,  # expected_total_errors
        ),
    ],
)
def test_mixed_scenario(
    scenario_name,
    predictions_data,
    ground_truths_data,
    pos_threshold,
    expected_error_types,
    expected_total_errors,
):
    """Test with a scenario that should show both main and special errors."""

    print(f"Testing {scenario_name}...")

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

    # Assert total number of errors
    assert (
        len(run.errors) == expected_total_errors
    ), f"Expected {expected_total_errors} total errors, got {len(run.errors)}"

    # Count error types
    error_type_counts = {}
    for error in run.errors:
        error_type_name = type(error).__name__
        error_type_counts[error_type_name] = (
            error_type_counts.get(error_type_name, 0) + 1
        )
        print(
            f"  {error_type_name}: prediction {getattr(error, 'pred', {}).get('_id', 'N/A')}"
        )

    # Assert expected error type counts
    for error_type, expected_count in expected_error_types.items():
        actual_count = error_type_counts.get(error_type, 0)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} {error_type} errors, got {actual_count}"

    # Debug main errors
    main_errors = run.fix_main_errors()
    print("\nMain errors:")
    for error_type, value in main_errors.items():
        print(f"  {error_type.__name__}: {value:.2f}")

    # Debug special errors
    special_errors = run.fix_special_errors()
    print("\nSpecial errors:")
    for error_type, value in special_errors.items():
        print(f"  {error_type.__name__}: {value:.2f}")

    # Show full summary
    print("\nFull summary:")
    tide_evaluator.summarize()


if __name__ == "__main__":
    test_mixed_scenario()
