import json
import shutil
import unittest
import tempfile

from pathlib import Path

from lfm_data_utilities.malaria_labelling import generate_tasks_from_list as gtl


class TestLabelMovement(unittest.TestCase):
    """
    In `generate_tasks_from_list.py`, we copy files with different folders into one for label studio,
    and then copy files back to their original folders. We want to ensure that there are no errors here.

    We can do this by copying to one, and from one to each, which should be identical to the originals.

    original label -> "central dir" -> "actual"

    original label should be actual

    original_label_dir/
        notes.json  <-- original notes.json file, can be different due to label studio weirdness
        labels/
            label1.txt
            label2.txt
            ...

    make "central dir" which acts as copy for many different original label dirs
    central_dir/
        notes.json  <-- master notes dot json file, usually will be from label studio
        labels/
            label1.txt
            label2.txt
            ...

    make "actual_dir" which should equal original label dir
    actual_dir/
        notes.json  <-- should already exist and be same as original_label_dir
        labels/
            label1.txt
            ...

    cmp original_label_dir w/ actual_dir
    """

    def setUp(self):
        self.central_dir = tempfile.TemporaryDirectory()
        self.central_dir_path = Path(self.central_dir.name)
        self.central_labels_dir = self.central_dir_path / "labels"
        self.central_labels_dir.mkdir()

        self.actual_dir = tempfile.TemporaryDirectory()
        self.actual_dir_path = Path(self.actual_dir.name)
        self.actual_labels_dir = self.actual_dir_path / "labels"
        self.actual_labels_dir.mkdir()

    def tearDown(self):
        self.central_dir.cleanup()
        self.actual_dir.cleanup()

    def test_copy_in_copy_out_is_id(self):
        path_to_labels_dir_a = Path(__file__).parent / Path("test_data/test_labels_a")
        path_to_labels_dir_b = Path(__file__).parent / Path("test_data/test_labels_b")
        path_to_label_dirs = [path_to_labels_dir_a, path_to_labels_dir_b]

        for label_dir_path in path_to_label_dirs:
            for label_file in (label_dir_path / "labels").iterdir():
                central_file_path = self.central_labels_dir / "central_labels.txt"
                with open(self.central_dir_path / "notes.json", "w") as f:
                    json.dump(gtl.MASTER_NOTES_DOT_JSON, f)

                actual_file_name = self.actual_dir_path / "labels" / "actual.txt"
                shutil.copyfile(
                    label_file.parent.parent / "notes.json",
                    self.actual_dir_path / "notes.json",
                )

                gtl.copy_label_to_central_dir(label_file, central_file_path)
                gtl.copy_label_to_original_dir(central_file_path, actual_file_name)

                labelfile = open(label_file, "r")
                actualfile = open(actual_file_name, "r")

                labelfile_data = labelfile.read().strip()
                actualfile_data = actualfile.read().strip()

                labelfile.close()
                actualfile.close()

                # compare both files; ignore trailing whitespace as that wouldn't be transfered
                self.assertEqual(labelfile_data, actualfile_data)


if __name__ == "__main__":
    unittest.main()
