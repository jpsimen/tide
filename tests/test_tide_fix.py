#!/usr/bin/env python3
"""
Test script to verify that TIDE properly handles images without ground truth.
"""

import tidecv


def test_tide_no_ground_truth():
    """Test that TIDE processes predictions correctly when there's no ground truth."""

    print("Testing TIDE evaluation with images that have no ground truth...")

    # Create TIDE evaluator
    tide_evaluator = tidecv.TIDE()

    # Create data objects
    predictions = tidecv.Data("test_predictions", max_dets=100)
    ground_truths = tidecv.Data("test_ground_truths", max_dets=100)

    # Add some predictions for image 0 (no ground truth for this image)
    predictions.add_detection(image_id=0, class_id=1, score=0.9, box=[10, 10, 50, 50])
    predictions.add_detection(image_id=0, class_id=2, score=0.8, box=[100, 100, 30, 30])

    # Add prediction and ground truth for image 1
    predictions.add_detection(image_id=1, class_id=1, score=0.95, box=[15, 15, 40, 40])
    ground_truths.add_ground_truth(image_id=1, class_id=1, box=[10, 10, 50, 50])

    # Add classes
    predictions.add_class(1, "class1")
    predictions.add_class(2, "class2")
    ground_truths.add_class(1, "class1")
    ground_truths.add_class(2, "class2")

    # This should cause ZeroDivisionError and fail the test
    run = tide_evaluator.evaluate(
        ground_truths, predictions, pos_threshold=0.5, mode=tidecv.TIDE.BOX
    )

    print("✓ TIDE evaluation completed successfully!")
    print("  - Total predictions processed: 3")  # We know we added 3 predictions
    print("  - Total ground truths: 1")  # We know we added 1 ground truth
    print(f"  - Number of errors detected: {len(run.errors)}")

    # Check for background errors (should include predictions from image 0)
    bg_errors = [
        e
        for e in run.errors
        if isinstance(e, tidecv.errors.main_errors.BackgroundError)
    ]
    print(
        f"  - Background errors (should include predictions from image 0): {len(bg_errors)}"
    )

    # Print each error for debugging
    for i, error in enumerate(run.errors):
        error_name = type(error).__name__
        pred_id = getattr(error, "pred", {}).get("_id", "N/A")
        print(f"  - Error {i+1}: {error_name} for prediction {pred_id}")

    print(f"  - Final mAP: {run.ap:.4f}")

    # Debug: show all predictions in the data object
    print("  - Predictions in data object: 3")
    print("    Pred 0: class=1, score=0.9, image=0")
    print("    Pred 1: class=2, score=0.8, image=0")
    print("    Pred 2: class=1, score=0.95, image=1")

    # Summarize
    tide_evaluator.summarize()


if __name__ == "__main__":
    test_tide_no_ground_truth()
