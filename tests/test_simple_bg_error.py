#!/usr/bin/env python3
"""
Simple test to understand how background errors are calculated.
"""

import tidecv


def test_simple_background_error():
    """Test with just false positives to understand background error calculation."""

    print("Testing simple background error scenario...")

    tide_evaluator = tidecv.TIDE()
    predictions = tidecv.Data("simple_predictions", max_dets=100)
    ground_truths = tidecv.Data("simple_gt", max_dets=100)

    # Add one ground truth
    ground_truths.add_ground_truth(image_id=0, class_id=1, box=[10, 10, 50, 50])

    # Add one true positive and one false positive
    predictions.add_detection(
        image_id=0, class_id=1, score=0.9, box=[10, 10, 50, 50]
    )  # TP
    predictions.add_detection(
        image_id=0, class_id=1, score=0.8, box=[100, 100, 30, 30]
    )  # FP (low IoU)

    predictions.add_class(1, "class1")
    ground_truths.add_class(1, "class1")

    # Evaluate
    run = tide_evaluator.evaluate(
        ground_truths, predictions, pos_threshold=0.5, mode=tidecv.TIDE.BOX
    )

    print(f"Base AP: {run.ap:.2f}")
    print(f"Number of errors: {len(run.errors)}")

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

    # Also test what happens if we manually fix false positives
    print("\nManual false positive fix test:")

    # Get original AP
    original_ap = run.ap_data.get_mAP()
    print(f"Original AP: {original_ap:.2f}")

    # Fix false positives by setting their score to 0
    fixed_fp_ap_data = run.fix_errors(
        transform=tidecv.errors.main_errors.FalsePositiveError.fix
    )
    fixed_ap = fixed_fp_ap_data.get_mAP()
    print(f"AP after fixing FPs: {fixed_ap:.2f}")
    print(f"Improvement: {fixed_ap - original_ap:.2f}")

    tide_evaluator.summarize()


if __name__ == "__main__":
    test_simple_background_error()
