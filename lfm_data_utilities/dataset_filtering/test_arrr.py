#! /usr/bin/env python3

import unittest

import torch

import arrr


class TestPerImgReduction(unittest.TestCase):
    def setUp(self):
        self.test_mini_prediction = torch.cat(
            (torch.zeros(3, 5), torch.tensor([[0.6, 0.2], [0.2, 0.8], [0.1, 0.9]])),
            dim=1,
        )
        self.test_img_prediction = torch.zeros(5, 12)
        self.test_img_prediction[:, 5:10] = torch.eye(5)

    def test_predicted_confidence_base(self):
        gt = torch.zeros(5, 7)
        gt[:, :5] = torch.eye(5)
        self.assertEqual(
            arrr.PerImgReduction.predicted_confidence(
                self.test_img_prediction
            ).tolist(),
            gt.tolist(),
        )

    def test_predicted_confidence_mini(self):
        test_mini_gt = torch.tensor([[0.6, 0.0], [0.0, 0.8], [0.0, 0.9]])
        self.assertEqual(
            arrr.PerImgReduction.predicted_confidence(
                self.test_mini_prediction
            ).tolist(),
            test_mini_gt.tolist(),
        )

    def test_predicted_confidence_ignore_smaller(self):
        gt = torch.zeros(5, 7)
        gt[:, :5] = torch.eye(5)

        # shouldn't do anything, since 1 > 0.5, the 0.5 will be masked out
        self.test_img_prediction[:, -1] = 0.5
        self.assertEqual(
            arrr.PerImgReduction.predicted_confidence(
                self.test_img_prediction
            ).tolist(),
            gt.tolist(),
        )

    def test_mean_predicted_confidence(self):
        gt = torch.zeros(7)
        gt[:5] = 1
        self.assertEqual(
            arrr.PerImgReduction.mean_predicted_confidence(
                self.test_img_prediction
            ).tolist(),
            gt.tolist(),
        )

    def test_mean_predicted_confidence_mini(self):
        test_mini_gt = torch.tensor([0.6, 0.85])
        self.assertTrue(
            torch.allclose(
                arrr.PerImgReduction.mean_predicted_confidence(
                    self.test_mini_prediction
                ),
                test_mini_gt,
            )
        )

    def test_class_count_mini(self):
        test_mini_gt = [1, 2]
        self.assertEqual(
            arrr.PerImgReduction.count_class(self.test_mini_prediction).tolist(),
            test_mini_gt,
        )


if __name__ == "__main__":
    unittest.main()
