#!/usr/bin/env python3
"""
Test special errors with a mix of scenarios to understand the expected behavior.
"""

import tidecv


def test_mixed_scenario():
    """Test with a scenario that should show both main and special errors."""

    print("Testing mixed scenario for special errors...")

    # Create a scenario with both correct and incorrect predictions
    tide_evaluator = tidecv.TIDE()
    predictions = tidecv.Data("mixed_predictions", max_dets=100)
    ground_truths = tidecv.Data("mixed_gt", max_dets=100)

    # Image 0: One correct detection, one false positive
    predictions.add_detection(
        image_id=0, class_id=1, score=0.9, box=[10, 10, 50, 50]
    )  # Should be TP
    predictions.add_detection(
        image_id=0, class_id=1, score=0.8, box=[100, 100, 30, 30]
    )  # Should be FP
    ground_truths.add_ground_truth(image_id=0, class_id=1, box=[10, 10, 50, 50])

    # Image 1: False positive only (no ground truth)
    predictions.add_detection(
        image_id=1, class_id=1, score=0.7, box=[200, 200, 40, 40]
    )  # Should be FP

    # Image 2: Missed detection (ground truth only)
    ground_truths.add_ground_truth(image_id=2, class_id=1, box=[300, 300, 60, 60])

    predictions.add_class(1, "class1")
    ground_truths.add_class(1, "class1")

    # Evaluate
    run = tide_evaluator.evaluate(
        ground_truths, predictions, pos_threshold=0.5, mode=tidecv.TIDE.BOX
    )

    print(f"Base AP: {run.ap:.2f}")
    print(f"Number of errors: {len(run.errors)}")

    # Print each error type
    for error in run.errors:
        print(
            f"  {type(error).__name__}: prediction {getattr(error, 'pred', {}).get('_id', 'N/A')}"
        )

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
