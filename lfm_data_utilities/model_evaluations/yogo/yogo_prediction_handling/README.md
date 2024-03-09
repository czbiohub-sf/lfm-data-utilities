# YOGO predictions - are we making the most of our predictions?

Due to YOGO's large grid size, each RBC prediction has many individual grid cells making positive predictions. Via bbox NMS (using objectness only as the score), we make a prediction on the cell's class. By integrating class confidence, we can probably improve on this metric.

## Prediction Selection Methods

A. NMS w/ score being `(class_confidence | objectness_score > 0.5)`

B. NMS w/ score being `(class_confidence * objectness_score)`

C. Some sort of voting? e.g. in a nxn (maybe n=5) grid centered on the "winning" grid cell, winning class is the $argmax(\sum_{i} max(p(c_i)))$

## E2E testing method

**just guess and check**

1. load a good model (eg fw1931) and run it through initial testing
    - do e.g. two identical runs to confirm that we get consistent results
2. modify formatting code so the confusion matrix uses the given prediction selection method A,B,C
3. run the model on the test set for each method and compare

## Open Questions

- What analysis can we perform on predictions to analytically determine the best method?
- Does training-time prediction selection methods change how the network learns?
