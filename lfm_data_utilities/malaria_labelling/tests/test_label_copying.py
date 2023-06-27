import shutil
import filecmp
import unittest
import tempfile

from pathlib import Path

from lfm_data_utilities.malaria_labelling import generate_tasks_from_list as gtl


class TestLabelMovement(unittest.TestCase):
    """
    In `generate_tasks_from_list.py`, we copy files with different folders into one for label studio,
    and then copy files back to their original folders. We want to ensure that there are no errors here.

    We can do this by copying to one, and from one to each, which should be identical to the originals.
    """

    def setUp(self):
        # this dir should mock the original source label dir; we don't want to destroy any test data, so
        # we copy it here
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.temp_labels_dir = self.temp_dir_path / "labels"
        self.temp_labels_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_copy_in_copy_out_is_id(self):
        path_to_labels_dir_a = Path(__file__).parent / Path("test_data/test_labels_a")
        path_to_labels_dir_b = Path(__file__).parent / Path("test_data/test_labels_b")
        path_to_label_dirs = [path_to_labels_dir_a, path_to_labels_dir_b]

        for label_dir_path in path_to_label_dirs:
            for label_file in (label_dir_path / "labels").iterdir():
                central_labels = tempfile.NamedTemporaryFile()
                central_file_path = Path(central_labels.name)

                actual_file_name = self.temp_labels_dir / central_file_path.name

                # copy notes.json into the tempdir and make a labels dir in it too
                shutil.copyfile(
                    label_file.parent.parent / "notes.json",
                    self.temp_dir_path / "notes.json",
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
