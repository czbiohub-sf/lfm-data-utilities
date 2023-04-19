# Strategy

As of the 17 April 2023, we have 266,939 images. Labelling each image would take prohibitively long. This document will describe the stratey we are employing in order to get to a high quality model.

An important note: the strategy is in flux, and this should be considered maleable. We are continuously open to improvements, open an issue or send us a Slack message!

## Strategy: Bottom-up

```
                    ┌────────────┐
                    │ Raw Images │
                    └─────┬──────┘
                          │
                          │
                      Cellpose
                      bounding box
                      labels
                          │
                          │
        All Labels        │
       ┌──────────────────▼──────────────────┐
       │                                     │
  ┌───►│ Machine labels   │    Human labels  │◄───────────────┐
  │    │                                     │                │
  │    └──────────────────┬──────────────────┘                │
  │                       │                                   │
  │                       │                                   │
  │                       ▼                                   │
  │                       │                                   │
┌─┴──────────┐            │                  ┌────────────────┴───┐
│Retrain YOGO│            │                  │Human Classification│
│and Relabel |◄───────────┴─────────────────►|and correction      │
└────────────┘                     ▲         └────────────────────┘
                                   │
                                   │
                                   │
                               Low-confidence
                               filtering
```

Initially, we just have images with no label data whatsoever. We use [`cellpose`](http://www.cellpose.org/) to initially provide us with bounding box labels. We classify everything as "healthy" since cellpose can't classify our cells. We keep track of what we've labelled and where labels are via [this spreadsheet](https://docs.google.com/spreadsheets/d/1PwMpBin-klGy4dKTF3670KhGDrrqVC0-AYRoPRTE0DE/).

From there, we enter the iteration loop. We take existing labels, classify and refine them using LabelStudio, then put them back into the pool of labels. Ocassionally, we can retrain YOGO and relabel the data for higher-quality machine labels, but note: there will always be a separation from human-generated labels and machine-generated labels, the first being more highly valued than the second.

We can also introduce low-confidence filtering, where we only send low-confidence YOGO predictions to humans for fixing. This will reduce the amount of data that humans have to process, while increasing the model.

Now, we shall discuss specifics.

## Specifics

See [scripts.md](https://github.com/czbiohub/lfm-data-utilities/blob/main/lfm_data_utilities/malaria-labelling/scripts.md) for information on the organization of data into folders. This section will discuss some specifics about how those folders should change over time.

### Machine labels vs. Human labels

Generated machine labels should be "beside" the images, as discussed in `scripts.md`. For now, human generated labels are in `/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels`. The reason is that we won't always label every image in a run, since variation across runs is more important. Several fragmented folders of labels for the same set of images is better suited in one location, and we can keep track of the label paths via the spreadsheet and by the names of the label directories. For training, we use [dataset description files](https://github.com/czbiohub/yogo/blob/main/docs/dataset_description.md) to tell YOGO where labels and corresponding images are[^1].


### Footnotes

[^1]: YOGO finds images to train on via the *labels*. So if you tell it that a folder of labels matches a folder of images, it will find the corresponding image for each label, instead of the corresponding label for each image. This lets you have multiple folders of labels corresponding to the same folder of images, useful for several fragmented folders of labels.
