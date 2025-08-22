from pathlib import Path


def get_verified_class_from_thumbnail_path(thumbail_path: Path):
    """
    Based on the folder path of the thumbail,
    we know what its original class was (based on its filename),
    as well as what its verified class is (based on the folder it is currently in).

    For example, if the thumbnail name is "ring_blahblah.png" and it is in the
    "corrected_healthy" folder, we know that it was machine labelled as ring
    but is actually a healthy cell based on human verification.

    Similarly, if the thumbnail's name is "ring_blahblah.png" and it remains in
    the "ring/*-completed-*" folder, then it was correctly classified as a ring
    by the model and has been human verified.
    """

    if "complete" in thumbail_path.parent.stem:
        return thumbail_path.parent.parent.stem
    elif "corrected" in thumbail_path.parent.stem:
        return thumbail_path.parent.stem
    else:
        raise ValueError(f"Thumbnail path {thumbail_path} is not in a valid folder.")
