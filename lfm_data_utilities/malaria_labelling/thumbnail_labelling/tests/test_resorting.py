#! /usr/bin/env python3


import shutil
import random
import zipfile
import unittest

from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union, Tuple, Generator

from yogo.data import YOGO_CLASS_ORDERING

from lfm_data_utilities.malaria_labelling.thumbnail_labelling.sort_thumbnails import (
    sort_thumbnails,
)


class TestResortingThumbnails(unittest.TestCase):
    test_labels_dir = Path(__file__) / "test_data" / "small_set_of_labels"
    test_thumbnails_dir = Path(__file__) / "test_data" / "test_thumbnails_dir"

    test_labels_zip = Path(__file__) / "test_data" / "labels_save.zip"
    test_thumbnails_zip = Path(__file__) / "test_data" / "thumbnails_save.zip"

    def label_path_to_labels(self, label_path: Path) -> List[List[Union[str, float]]]:
        with open(label_path, "r") as f:
            return [
                [YOGO_CLASS_ORDERING.index(line[0]), *[float(n) for n in line[1:]]]
                for line in f.readlines()
            ]

    def random_sort_thumbnails(self) -> Dict[str, int]:
        initial_total_count = 0

        final_class_counts: Dict[str, int] = defaultdict(int)
        moved_thumbnail_counts: Dict[str, int] = defaultdict(int)

        for class_ in YOGO_CLASS_ORDERING:
            class_dir = self.test_thumbnails_dir / class_

            original_thumbnails = list(class_dir.iterdir())

            initial_total_count += len(original_thumbnails)

            k = random.randint(0, len(original_thumbnails) - 1)

            final_class_counts[class_] = len(original_thumbnails) - k

            thumbnails_to_move = random.sample(original_thumbnails, k)
            target_classes = random.choices(YOGO_CLASS_ORDERING, k=k)

            for thumbnail, target_class in zip(thumbnails_to_move, target_classes):
                shutil.move(
                    thumbnail, self.test_thumbnails_dir / f"corrected_{target_class}"
                )
                moved_thumbnail_counts[target_class] += 1

        for class_, moved_count in moved_thumbnail_counts.items():
            final_class_counts[class_] += moved_count

        assert sum(final_class_counts.values()) == initial_total_count, f"Expected {initial_total_count} thumbnails, but found {sum(final_class_counts.values())}"
        return dict(final_class_counts)

    def count_num_classes_in_label_dir(self, label_dir: Path) -> Dict[str, int]:
        num_classes_by_label: Dict[str, int] = defaultdict(int)
        for label_path in label_dir.iterdir():
            labels = self.label_path_to_labels(label_path)
            for label in labels:
                num_classes_by_label[YOGO_CLASS_ORDERING[int(label[0])]] += 1

        return num_classes_by_label

    def reset_dirs(self):
        shutil.rmtree(self.test_labels_dir)
        shutil.rmtree(self.test_thumbnails_dir)

        with zipfile.ZipFile(self.test_thumbnails_zip, "r") as zip_ref:
            zip_ref.extractall(self.test_thumbnails_dir)

    def setUp(self):
        self.reset_dirs()

    def tearDown(self):
        self.reset_dirs()

    def test_resorting_thumbnails(self):
        expected_class_counts = self.random_sort_thumbnails()

        sort_thumbnails(self.test_thumbnails_dir, _backup=False)

        number_of_corrected_labels = self.count_num_classes_in_label_dir(self.test_labels_dir)

        print(f"{expected_class_counts=}")
        print(f"{number_of_corrected_labels=}")

        self.assertEqual(expected_class_counts, number_of_corrected_labels)
