#!/usr/bin/env python3
"""
Comprehensive test script to verify TIDE fix for images without ground truth.
"""

import sys

import tidecv


def test_comprehensive_tide_scenarios():
    """Test multiple scenarios with TIDE."""

    print("Running comprehensive TIDE tests...")

    # Test 1: Only predictions, no ground truth at all
    print("\n=== Test 1: Only predictions, no ground truth ===")
    tide_evaluator = tidecv.TIDE()
    predictions = tidecv.Data("only_predictions", max_dets=100)
    ground_truths = tidecv.Data("empty_gt", max_dets=100)

    predictions.add_detection(image_id=0, class_id=1, score=0.9, box=[10, 10, 50, 50])
    predictions.add_detection(image_id=1, class_id=2, score=0.8, box=[100, 100, 30, 30])

    predictions.add_class(1, "class1")
    predictions.add_class(2, "class2")
    ground_truths.add_class(1, "class1")
    ground_truths.add_class(2, "class2")

    # This should fail with ZeroDivisionError, causing the test to fail
    run = tide_evaluator.evaluate(
        ground_truths, predictions, pos_threshold=0.5, mode=tidecv.TIDE.BOX
    )
    bg_errors = [
        e
        for e in run.errors
        if isinstance(e, tidecv.errors.main_errors.BackgroundError)
    ]
    assert len(bg_errors) == 2

    # Test 2: Mixed scenario - some images with GT, some without
    print("\n=== Test 2: Mixed scenario ===")
    tide_evaluator2 = tidecv.TIDE()
    predictions2 = tidecv.Data("mixed_predictions", max_dets=100)
    ground_truths2 = tidecv.Data("mixed_gt", max_dets=100)

    # Image 0: predictions only ==> should count as background errors
    predictions2.add_detection(image_id=0, class_id=1, score=0.99, box=[10, 10, 50, 50])
    predictions2.add_detection(image_id=0, class_id=1, score=0.98, box=[20, 20, 40, 40])

    # Image 1: true positive prediction and GT
    predictions2.add_detection(image_id=1, class_id=1, score=0.95, box=[11, 10, 49, 50])
    ground_truths2.add_ground_truth(image_id=1, class_id=1, box=[10, 10, 50, 50])

    # Image 2: GT only (no predictions) ==> should count as a missed detection
    ground_truths2.add_ground_truth(image_id=2, class_id=1, box=[30, 30, 60, 60])

    predictions2.add_class(1, "class1")
    ground_truths2.add_class(1, "class1")

    run2 = tide_evaluator2.evaluate(
        ground_truths2, predictions2, pos_threshold=0.5, mode=tidecv.TIDE.BOX
    )
    # Verify that we processed all images
    expected_errors = 2 + 0 + 1  # 2 from image 0, 0 from image 1, 1 from image 2
    assert len(run2.errors) == expected_errors

    bg_errors = [
        e
        for e in run2.errors
        if isinstance(e, tidecv.errors.main_errors.BackgroundError)
    ]
    missed_errors = [
        e for e in run2.errors if isinstance(e, tidecv.errors.main_errors.MissedError)
    ]
    box_errors = [
        e for e in run2.errors if isinstance(e, tidecv.errors.main_errors.BoxError)
    ]

    assert len(bg_errors) == 2
    assert len(missed_errors) == 1
    assert len(box_errors) == 0

    # Assert that the false positive predictions from image 0 are counted as background errors
    main_errors = tide_evaluator2.get_main_errors()
    assert main_errors["mixed_predictions"]["Bkg"] > 0.0
    special_errors = tide_evaluator2.get_special_errors()
    assert special_errors["mixed_predictions"]["FalsePos"] > 0.0


if __name__ == "__main__":
    success = test_comprehensive_tide_scenarios()
    sys.exit(0 if success else 1)
