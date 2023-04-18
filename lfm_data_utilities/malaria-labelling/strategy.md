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

Initially, we just have images with no label data whatsoever. We use [`cellpose`](http://www.cellpose.org/) to initially provide us with bounding box labels. We classify everything as "healthy" since cellpose can't classify our cells.

From there, we enter the iteration loop. We take existing labels, classify and refine them using LabelStudio, then put them back into the pool of labels. Ocassionally, we can retrain YOGO and relabel the data for higher-quality machine labels, but note: there will always be a separation from human-generated labels and machine-generated labels, the first being more highly valued than the second.

We can also introduce low-confidence filtering, where we only send low-confidence YOGO predictions to humans for fixing. This will reduce the amount of data that humans have to process, while increasing the model.

Now, we shall discuss specifics.

## Specifics

See [scripts.md](https://github.com/czbiohub/lfm-data-utilities/blob/main/lfm_data_utilities/malaria-labelling/scripts.md) for information on the organization of data into folders. This section will discuss some specifics about how those folders should change over time.
