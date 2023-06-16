# What is here

## YOGO Output Analyzer

Lets you inspect the output layers of the network, useful for debugging what the network is predicting given an input

## `loss-ranking.ipynb` or `rank_yogo_loss.py`

Run YOGO over all labelled data! Useful to show where YOGO is struggling most, or where labels are incorrect.
`loss-ranking.ipynb` is an interactive tool, where `rank_yogo_loss.py` will generate a `csv` appropriate for import into Excel.

## `make\_bbox\_video.sh`

Creates a video w/ bboxes overlayed

## `calculate\_titration.py`

Calculates the titration curve of a titration dataset. From the spreadsheet,

> This is a serial dilution of cultured parasites into fresh (sampled at start of the experiment) healthy, whole fingertip blood. The starting culture will be unsynced parasites at 15-20% parasitemia. All points will be imaged at 5% hematocrit in diluent. The serial dilution can be created at the start of the experiment, with all the tubes kept in the incubator until right before imaging. Full-length oracle runs will be collected for each dilution point (20,000 frames). For ground truth, either a high-accuracy Giemsa (lots of FOVs counted) smear can be performed of the starting point, or flow cytometry can be performed to determine the underlying parasitemia.
