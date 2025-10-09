#!/usr/bin/env python3
"""
Test script to verify that TIDE properly handles images without ground truth.
"""

import pytest
import tidecv


@pytest.mark.parametrize(
    "scenario_name,predictions_data,ground_truths_data,pos_threshold,expected_total_errors,expected_bg_errors,should_complete_successfully",
    [
        (
            "mixed_images_with_different_classes",
            {
                "name": "test_predictions",
                "detections": [
                    # Image 0: predictions only (no ground truth for this image)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    # Image 2: prediction with matching ground truth
                    {
                        "image_id": 2,
                        "class_id": 2,
                        "score": 0.8,
                        "box": [11, 11, 49, 49],
                    },
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            {
                "name": "test_ground_truths",
                "ground_truths": [
                    # Image 1: ground truth only
                    {"image_id": 1, "class_id": 1, "box": [10, 10, 50, 50]},
                    # Image 2: ground truth with matching prediction
                    {"image_id": 2, "class_id": 2, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            0.5,
            2,  # expected_total_errors
            1,  # expected_bg_errors
            True,  # should_complete_successfully
        ),
        (
            "mixed_images_with_same_classes",
            {
                "name": "test_predictions",
                "detections": [
                    # Image 0: predictions only (no ground truth for this image)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    # Image 2: prediction with matching ground truth
                    {
                        "image_id": 2,
                        "class_id": 1,
                        "score": 0.8,
                        "box": [11, 11, 49, 49],
                    },
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            {
                "name": "test_ground_truths",
                "ground_truths": [
                    # Image 1: ground truth only
                    {"image_id": 1, "class_id": 1, "box": [10, 10, 50, 50]},
                    # Image 2: ground truth with matching prediction
                    {"image_id": 2, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            0.5,
            2,  # expected_total_errors
            1,  # expected_bg_errors
            True,  # should_complete_successfully
        ),
        (
            "mixed_images_with_and_without_gt_2",
            {
                "name": "test_predictions",
                "detections": [
                    # Image 0: predictions only (no ground truth for this image)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    {
                        "image_id": 0,
                        "class_id": 2,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },
                    # Image 1: prediction with matching ground truth
                    {
                        "image_id": 1,
                        "class_id": 1,
                        "score": 0.95,
                        "box": [11, 11, 49, 49],
                    },
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            {
                "name": "test_ground_truths",
                "ground_truths": [
                    # Image 1: ground truth for matching prediction
                    {"image_id": 1, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            0.5,
            2,  # expected_total_errors (2 background errors from image 0)
            2,  # expected_bg_errors
            True,  # should_complete_successfully
        ),
        (
            "only_predictions_no_gt_at_all",
            {
                "name": "test_predictions",
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
                    {
                        "image_id": 2,
                        "class_id": 1,
                        "score": 0.7,
                        "box": [200, 200, 40, 40],
                    },
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            {
                "name": "test_ground_truths",
                "ground_truths": [],
                "classes": {1: "class1", 2: "class2"},
            },
            0.5,
            3,  # expected_total_errors (all predictions become background errors)
            3,  # expected_bg_errors
            True,  # should_complete_successfully
        ),
        (
            "different_threshold",
            {
                "name": "test_predictions",
                "detections": [
                    # Image 0: predictions only (no ground truth for this image)
                    {
                        "image_id": 0,
                        "class_id": 1,
                        "score": 0.9,
                        "box": [10, 10, 50, 50],
                    },
                    {
                        "image_id": 0,
                        "class_id": 2,
                        "score": 0.8,
                        "box": [100, 100, 30, 30],
                    },
                    # Image 1: prediction with matching ground truth
                    {
                        "image_id": 1,
                        "class_id": 1,
                        "score": 0.95,
                        "box": [15, 15, 40, 40],
                    },
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            {
                "name": "test_ground_truths",
                "ground_truths": [
                    # Image 1: ground truth for matching prediction
                    {"image_id": 1, "class_id": 1, "box": [10, 10, 50, 50]},
                ],
                "classes": {1: "class1", 2: "class2"},
            },
            0.3,
            2,  # expected_total_errors (2 background errors from image 0)
            2,  # expected_bg_errors
            True,  # should_complete_successfully
        ),
    ],
)
def test_tide_no_ground_truth(
    scenario_name,
    predictions_data,
    ground_truths_data,
    pos_threshold,
    expected_total_errors,
    expected_bg_errors,
    should_complete_successfully,
):
    """Test that TIDE processes predictions correctly when there's no ground truth."""

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
    if should_complete_successfully:
        run = tide_evaluator.evaluate(
            ground_truths,
            predictions,
            pos_threshold=pos_threshold,
            mode=tidecv.TIDE.BOX,
        )

        print("✓ TIDE evaluation completed successfully!")
        print(f"  - Total predictions processed: {len(predictions_data['detections'])}")
        print(f"  - Total ground truths: {len(ground_truths_data['ground_truths'])}")
        print(f"  - Number of errors detected: {len(run.errors)}")

        # Assert expected total errors
        assert (
            len(run.errors) == expected_total_errors
        ), f"Expected {expected_total_errors} total errors, got {len(run.errors)}"

        # Check for background errors
        bg_errors = [
            e
            for e in run.errors
            if isinstance(e, tidecv.errors.main_errors.BackgroundError)
        ]

        print(f"  - Background errors: {len(bg_errors)}")
        assert (
            len(bg_errors) == expected_bg_errors
        ), f"Expected {expected_bg_errors} background errors, got {len(bg_errors)}"

        # Print each error for debugging
        for i, error in enumerate(run.errors):
            error_name = type(error).__name__
            pred_id = getattr(error, "pred", {}).get("_id", "N/A")
            print(f"  - Error {i+1}: {error_name} for prediction {pred_id}")

        print(f"  - Final mAP: {run.ap:.4f}")

        # Debug: show prediction details
        print(f"  - Predictions in data object: {len(predictions_data['detections'])}")
        for i, detection in enumerate(predictions_data["detections"]):
            print(
                f"    Pred {i}: class={detection['class_id']}, score={detection['score']}, image={detection['image_id']}"
            )

        # Summarize
        tide_evaluator.summarize()

    else:
        # If we expect it to fail, we should catch the exception
        with pytest.raises(Exception):
            run = tide_evaluator.evaluate(
                ground_truths,
                predictions,
                pos_threshold=pos_threshold,
                mode=tidecv.TIDE.BOX,
            )


if __name__ == "__main__":
    test_tide_no_ground_truth()
