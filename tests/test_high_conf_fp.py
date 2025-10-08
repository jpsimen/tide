#!/usr/bin/env python3
"""
Test background error with high-confidence false positive.
"""

import tidecv


def test_high_conf_fp():
    """Test with false positive having higher confidence than true positive."""

    print("Testing high-confidence false positive scenario...")

    tide_evaluator = tidecv.TIDE()
    predictions = tidecv.Data("high_conf_fp", max_dets=100)
    ground_truths = tidecv.Data("high_conf_gt", max_dets=100)

    # Scenario: FP has higher confidence than TP
    ground_truths.add_ground_truth(image_id=0, class_id=1, box=[10, 10, 50, 50])
    predictions.add_detection(
        image_id=0, class_id=1, score=0.9, box=[100, 100, 30, 30]
    )  # FP (high conf)
    predictions.add_detection(
        image_id=0, class_id=1, score=0.8, box=[10, 10, 50, 50]
    )  # TP (low conf)

    predictions.add_class(1, "class1")
    ground_truths.add_class(1, "class1")

    # Evaluate
    run = tide_evaluator.evaluate(
        ground_truths, predictions, pos_threshold=0.5, mode=tidecv.TIDE.BOX
    )

    print(f"Original AP: {run.ap:.2f}")

    # Show main and special errors
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

    # Full summary
    print("\nFull TIDE summary:")
    tide_evaluator.summarize()


if __name__ == "__main__":
    test_high_conf_fp()
