# Thumbnail Labelling

## Labelling Guide

Given datasets (i.e. folder of images + folder of labels), grab a thumbnail of each cell and put it in folders, sorted by class. A user can then sort those thumbnails into the correct folder. You can then use those corrected thumbnanils to update the original labels.

Knowing the folder structure for thumbnails and how this tool uses it is very important for understanding how you can label.

When creating thumbnails (with `thumbnail_sort_labelling.py create-thumbnails`), a directory with the following structure is created:

```console
$ tree -I *.png
.
├── corrected_gametocyte
├── corrected_healthy
├── corrected_misc
├── corrected_ring
├── corrected_schizont
├── corrected_trophozoite
├── corrected_wbc
├── gametocyte
│   └── 0                 // these numbered folders contain thumbnails, but I won't list any of these
│       ├── gametocyte_f53l5vgr22_0.png
│       └── gametocyte_gmaa35f509_0.png
├── healthy
│   ├── 0
│   ├── 1
│   └── 2
├── wbc
│   └── 0
├── misc
│   └── 0
├── ring
│   └── 0
├── trophozoite
│   └── 0
├── schizont
├── tasks
│   └── thumbnail_correction_task_0.json
├── see_in_context.py
└── id_to_task_path.json
```

Namely,

- there will be the class folders (e.g. `healthy`, `ring`) that have thumbnails from all of the labels / predictions
- there will the `corrected_*` folders, which are empty. When you find a cell in one of the class folders that belongs to a different class (e.g. a ring in the `healthy` folder), sort it into `corrected_ring`.
- there will be a folder `tasks` with one or more Label Studio `task.json` files. This folder is very important, as it contains the files that are used to update the original labels with corrections. Do *NOT* touch anything in the `tasks` folder.
- `id_to_task_path.json` is used to map which dataset each cell is from. It should also not be touched.
- `see_in_context.py` can be used to see the original image for a given thumbnail. For example, `see_in_context.py ring_2a4bee1ecf_0.png` will show you the image to which that thumbnail belongs.

We also have two labelling guides to help with classifications:

- [main labelling guide](https://docs.google.com/document/d/1SIrPd26qItAEqbjrFD6go6M3KcSVFFXLJkKim5K4tH0/)
- [ring -> troph -> schizont transition](https://docs.google.com/document/d/1cH8Bprr64GjiaRhwKqBGlBN8xjNwSHJUYw-AGYtVg6A/)


# Creating and sorting thumbnails (i.e. using `thumbnail_sort_labelling.py`)

This section will discuss how to actually *create* and *sort* thumbnails. First,

```console
thumbnail_labelling (main) | ./thumbnail_sort_labelling.py --help
usage: thumbnail_sort_labelling.py [-h] {create,sort} ...

positional arguments:
  {create,sort}

optional arguments:
  -h, --help     show this help message and exit
```

## `thumbnail_sort_labelling.py create`

Creating thumbnails is a natural place to start. The general idea is that you have a set of images, and one of the three hold

- `labels`: You have labels for the images, and you want to update / correct the labels
- `yogo-incorrect`: You have labels and you want to update / correct the labels based on YOGO's predictions
- `yogo-confidence`: You don't *necessarily* have labels, and want to export thumbnails based solely on YOGOs predictions. Labels are not used for this method.

`thumbnail_sort_labelling.py create` can create the thumbnails in each case:

```console
thumbnail_labelling (main) | ./thumbnail_sort_labelling.py create --help
usage: thumbnail_sort_labelling.py create [-h] [--path-to-labelled-data-ddf PATH_TO_LABELLED_DATA_DDF | --path-to-run PATH_TO_RUN] [--overwrite-previous-thumbnails] [--ignore-class IGNORE_CLASS]
                                          [--thumbnail-type {labels,yogo-confidence,yogo-incorrect}] [--path-to-pth PATH_TO_PTH] [--max-confidence MAX_CONFIDENCE] [--min-confidence MIN_CONFIDENCE]
                                          [--obj-thresh OBJ_THRESH] [--iou-thresh IOU_THRESH] [--image-server-relative-parent-override IMAGE_SERVER_RELATIVE_PARENT_OVERRIDE]
                                          path_to_output_dir

positional arguments:
  path_to_output_dir

optional arguments:
  -h, --help            show this help message and exit
  --path-to-labelled-data-ddf PATH_TO_LABELLED_DATA_DDF
                        path to dataset descriptor file for labelled data (default {default_ddf})
  --path-to-run PATH_TO_RUN
                        path to dataset descriptor file run
  --overwrite-previous-thumbnails
                        if set, will overwrite previous thumbnails
  --ignore-class IGNORE_CLASS
                        if set, will ignore this class when creating thumbnails - e.g. `--ignore-class healthy` you can provide this argument multiple times to ignore multiple classes - e.g. `--ignore-class healthy
                        --ignore-class misc` suggested: `--ignore-class healthy`
  --thumbnail-type {labels,yogo-confidence,yogo-incorrect}
                        which type of thumbnail to create - labels: thumbnails with the labels as provided by the human labelers yogo-confidence: thumbnails predicted by YOGO, with high class confidence scores
                        filtered by `--max-confidence` yogo-incorrect: thumbnails where the yogo model was incorrect
  --path-to-pth PATH_TO_PTH
                        if `--thumbnail-type yogo-confidence` or `--thumbnail-type yogo-incorrect` is provided, this is the path to the .pth file containing the model weights
  --max-confidence MAX_CONFIDENCE
                        if `--thumbnail-type yogo-confidence` is provided, this is the maximum confidence score to include in the thumbnail (default 1)
  --min-confidence MIN_CONFIDENCE
                        if `--thumbnail-type yogo-confidence` is provided, this is the minimum confidence score to include in the thumbnail (default 0)
  --obj-thresh OBJ_THRESH, --obj-threshold OBJ_THRESH, --objectness-threshold OBJ_THRESH
                        objectness threshold for YOGO predictions
  --iou-thresh IOU_THRESH, --iou-threshold IOU_THRESH, --iou-threshold IOU_THRESH
                        iou threshold for YOGO predictions
  --image-server-relative-parent-override IMAGE_SERVER_RELATIVE_PARENT_OVERRIDE
                        override the image server relative parent for generating tasks; if you want the root of the image serverto be different from LFM_Scope, you can provide that here - but don't touch this if
                        that doesn't make sense
```

You must provide an output directory (we have been using `.../LFM_scope/thumbnail_corrections/<group of corrections, such as 'Uganda Subsets'>). Other parameters are dependant on what you want to export

- `--thumbnail-type` is the switch for the different export types
- `--ignore-class` is a flag to ignore a certain class, for example `healthy`. There are *many many* more healthy cells than parasitic cells, and if you are exporting thumbnails to correct parasite labels, you probably don't want to export healthy cells (I suggest exporting healthy cells with `--max-confidence 0.8` or below). This can be chained: eg `--ignore-class healthy --ignore-class misc --ignore-class wbc`
- `--path-to-run` and `--path-to-labelled-data-ddf` are mutually exclusive. If you are exporting thumbnails from only one run, you can simply use `--path-to-run`. If you are exporting thumbnails for many runs, it can be nice to define which runs you want to export via a [dataset definition file](https://github.com/czbiohub-sf/yogo/blob/main/docs/dataset-definition.md).
- `--path-to-pth` is the path to the YOGO model that you want to use if your thumbnail type is `yogo-incorrect` or `yogo-confidence`. Often the best models are in `.../LFM_scope/yogo_models`

`--min/max-confidence` are the minimum and maximum *class* confidence scores required for a thumbnail to be exported. This is very good for finding instances where YOGO makes mistakes, and for cutting down the number of thumbnails that you have to observe[^1].
`--obj-threshold` and `--iou-threshold` are the objectness and [intersection over union](https://en.wikipedia.org/wiki/Jaccard_index) (for [non-maximum supression](https://en.wikipedia.org/wiki/Edge_detection#Canny) thresholds, used to filter YOGO predictions. Quite standard, read more about them [here](https://github.com/czbiohub-sf/yogo/blob/main/docs/yogo-high-level.md).

After that point, you should be good to go start [labelling](https://github.com/czbiohub-sf/lfm-data-utilities/blob/main/lfm_data_utilities/thumbnail_labelling/README.md#labelling-guide).

## `thumbnail_sort_labelling.py sort`

Alright, you have sorted a set of thumbnails. Now we want to update our labels for training.

> [!WARNING]\
> If you are correcting labels, heed this: first, and most importantly, our labels live in a git repo ([here](https://github.com/czbiohub-sf/lfm-human-labels)) at the path `.../LFM_scope/biohub-labels/vetted`. **Make sure that there are no changes in the labels before sorting thumbnails**. The repo on Github is considered correct, and after updating labels, you can verify the changes with `git diff`.

Here is the sorting tool:

```console
thumbnail_labelling (main) | ./thumbnail_sort_labelling.py sort --help
usage: thumbnail_sort_labelling.py sort [-h] [--commit | --no-commit] [--ok-if-class-mismatch | --no-ok-if-class-mismatch] path_to_thumbnails

positional arguments:
  path_to_thumbnails

optional arguments:
  -h, --help            show this help message and exit
  --commit, --no-commit
                        actually modify files on the file system instead of reporting what would be changed (default: False)
  --ok-if-class-mismatch, --no-ok-if-class-mismatch
                        if set, will not raise an error if the class of a thumbnail does not match the folder it is in (default: False)
```

Much simpler than the tool for creation, you simply give it the path to the thumbnail directory. By default, no changes will be made, and you will have to append `--commit` to your command in order to make changes. Also, if you are importing the labels in multiple steps (eg you are correcting a huge number of thumbnails and you wanted to import corrections after fixing the ring predictions), you have to append `--ok-if-class-mismatch`. This is a check that verifies that a corrected thumbnail had the expected original class.

After running it without `--commit`, you'll see something like this:

```console
(eeeeeee) [axel.jacobsen@login-02 thumbnail_labelling]$ ./thumbnail_sort_labelling.py sort <path to thumbnails>
--commit not provided, so this will be a dry run - no files will be modified
would have written corrected tasks.json file to <path to thumbnails>/tasks/thumbnail_correction_task_0.json
not corrected: 0, was corrected: 7
would have overwritten YOGO labels at <path to original labels>
```

If you see something like this,

```console
(eeeeeee) [axel.jacobsen@login-02 thumbnail_labelling]$ ./thumbnail_sort_labelling.py sort <path to thumbnails>
--commit not provided, so this will be a dry run - no files will be modified
could not find cell_id d77a321e95 in task {'label_path': '<path to original labels>', 'task_name': 'thumbnail_correction_task_0.json', 'task_num': 0}                                          
not corrected: 1, was corrected: 138
would have overwritten YOGO labels at <path to original labels>
```

that means that the tool could not find the thumbnail with id `d77a321e95` in it's `tasks.json` file. This is not good, as that means that we can't the source for that thumbnail, and therefore we can't correct its label. **However, the other thumbnails that were found can be corrected**. When this occurs, please ping Axel, [open an issue](https://github.com/czbiohub-sf/lfm-data-utilities/issues/new), or carefully try to debug the issue[^2].

## How we correct labels with thumbnails (under the hood)

When we want to re-sort our labels, we follow this algorithm:
- iterate through `corrected_*` folders and create a map of each thumbnail to the corrected class (e.g. if `ring_2a4bee1ecf_0.png` is in `corrected_healthy`, it is actually a healthy cell, so remember that)
- for each cell in that map of corrections, find the the cell in it's `tasks.json` file and change it's class to the correct class
  - in the name `ring_2a4bee1ecf_0.png`, `ring` is the originally predicted class, `2a4bee1ecf` is the id of that thumbnail in it's `task.json` file, and `0` is the `tasks.json` file that was used, as defined by `id_to_task_path.json`
- export each `tasks.json` file to the location of the source

Note that we only look at `corrected_*` for corrections; we do not look at the class folders at all, so you can change those in any way that makes labelling easier for you.

[^1]: My hypothesis is that you can start with a low maximum confidence for healthy cells, correct the parasites that were classified as healthy, retrain, and re-export thumbnails at a similar confidence to find a new batch of parasites in the healthy classification.

[^2]: The cell IDs are 10 hexidecimal digits, so there is a (1 in 64^10) ~= 8.67 * 10^-19 percent chance of ID collision. So, if we can find the id in it's source `tasks.json` file, we will be able to move it into the correct spot. While writing this document, I actually did this for `d77a321e95`. The thumbnail was in `...LFM_scope/thumbnail-corrections/Uganda-subsets/2023-02-15-042659_/corrected_misc`, so I ran `cd LFM_scope/thumbnail-corrections/Uganda-subsets` and then `grep -rl d77a321e95 */tasks/*`. This immediately gave me `2023-04-27-052714_/tasks/thumbnail_correction_task_0.json` - which is not the same run of the thumbnail's current location! So, just moving the thumbnail from `2023-02-15-042659_/corrected_misc` to `2023-04-27-052714_/corrected_misc` fixed the problem. Be careful to put thumbnails into the correct spot! But more importantly, be able to debug these sorts of issues.
