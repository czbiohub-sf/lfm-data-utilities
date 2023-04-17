# Tools

Having a clear, reproducible, and understandable data pipeline is absolutely necessary if we would like to maintain our sanity + scientific rigour. Here is the framework that I am using for our specific scenario (lots of "little" datasets, many various run conditions, e.t.c.).

## Initial Data Processing

Data from the scope is collected to `/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/scope-parasite-data/run-sets/`

- Runs are grouped together in folders, e.g. `.../run-sets/2022-12-14-111221-Aditi-parasites`
- These runs are in folders specifying the chip, e.g. `.../run-sets/2022-12-14-111221-Aditi-parasites/2022-12-13-122015__chip0562-A4_F`
- These runs have a [Zarrfile](https://zarr.readthedocs.io/en/stable/) storing the images (as a single multidimensional array), along with metadata, and subsample of images (as .pngs) from the run. Example

``` console
.../run-sets/
  2022-12-14-111221-Aditi-parasites/

    2022-12-13-122015__chip0562-A4_F/          # Run Folder
      2022-12-13-122015__chip0562-A4_F.zip    # Zarrfile
      metadata.csv
      sub\_sample\_images/
        0.png
        ...

    2022-12-13-122015__chip0562-A4_m/          # Run Folder
      2022-12-13-122015__chip0562-A4_m.zip    # Zarrfile
      metadata.csv
      sub\_sample\_images/
        0.png
        ...

    ...

  2022-12-14-154742-Aditi-parasites/
    ... etc  etc ...
```

YOGO and annotation requires image files, so we must convert the zarrfiles to folders of images.

Use `python3 process_zarr_to_images.py <path to run-sets>` - this will look for all Zarrfiles, and make a folder of images beside it. Each run folder would become

```console
2022-12-13-122015__chip0562-A4_F          # Run Folder
  2022-12-13-122015__chip0562-A4_F.zip    # Zarrfile
  images/                                 # folder of images from Zarrfile
    img_0000.png
    img_0001.png
    ...
  metadata.csv
  sub\_sample\_images
    0.png
    ...
  ... # some other experiment metadata
```

## Creating Cellpose or YOGO labels

To create bounding box labels for images, we use [Cellpose](https://www.google.com/search?client=firefox-b-d&q=Cellpose). We can also create labels from a trained YOGO model (or untrained, but they would be very bad labels).

We use the tool `generate_labels.py`. If you run `generate_labels.py --help`, you get

```console
usage: label a set of run folders [-h] [--existing-label-action {skip,overwrite}] [--model {cellpose,yogo}] [--path-to-yogo-pth PATH_TO_YOGO_PTH] [--label-dir-name LABEL_DIR_NAME] [--tasks-file-name TASKS_FILE_NAME] path_to_runset

positional arguments:
  path_to_runset        path to run folders

optional arguments:
  -h, --help            show this help message and exit
  --existing-label-action {skip,overwrite}, -e {skip,overwrite}
                        skip or overwrite existing labels when encountered, defaults to 'skip'
  --model {cellpose,yogo}
                        choose cellpose or yogo to label the data
  --path-to-yogo-pth PATH_TO_YOGO_PTH
                        path to pth file for the yogo model; required if yogo is the selected model
  --label-dir-name LABEL_DIR_NAME
                        name for label dir for each runset - defaults to 'labels'
  --tasks-file-name TASKS_FILE_NAME
                        name for label studio tasks file - defaults to tasks.json
```

You need a GPU to run this script in any reasonable amount of time. You can allocate an interactive GPU node with `salloc --partition=gpu --gres=gpu:1 -c 16` if you are doing some smaller jobs, but for any big jobs (more than a couple run sets), use `sbatch submit_cmd.sh python3 generate_labels.py <your args>`.

The simplest invocation is

```
sbatch submit_cmd.sh python3 generate_labels.py /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/scope-parasite-data/run-sets/
```

This will find all run sets in `.../scope-parasite-data/run-sets/`, create a `labels` directory beside each `images` directory, label all the images with Cellpose (the default model), and then create a `tasks.json` file for further Labelstudio labelling. Note that by default, it will also skip over existing labels. You can overwrite existing labels with `--existing-label-action overwrite`. This is useful when you make errors, or maybe you've tweaked the label format that YOGO expects.

```console
2022-12-13-122015__chip0562-A4_F          # Run Folder
  2022-12-13-122015__chip0562-A4_F.zip    # Zarrfile
  images/                                 # folder of images from Zarrfile
    img_0000.png
    img_0001.png
    ...
  labels/                                 # labels for images/
    img_0000.txt
    img_0001.txt
  tasks.json
  ... # the other experiment metadata, subsample images, etc
```

At this point, `labels` should have good bounding boxes for `images`. You can verify with

`./visualize_boxes.py <path to images folder> <path to labels folder>`

(make sure you've `ssh`'d into Bruno with XTerm - e.g. `ssh account@login01.czbiohub.org -Y`)

As well as the files created per run folder, a `dataset_description.yml` file will be created in the directory where you executed `sbatch submit_cmd.sh python3 generate_labels.py ...`. This file includes paths to all of the images and new label directories, and it is what is used (see [the docs for Dataset Description files](https://github.com/czbiohub/yogo/blob/main/docs/dataset_description.md)).

All cells will be classified as healthy, though. If you have a YOGO model trained with classification, you can run

``` console
sbatch submit_cmd.sh python3 generate_labels.py --model yogo --path-to-yogo-pth <path to pretrained yogo .pth file> --label-dir-name yogo_labels --tasks-file-name yogo_labelled_tasks.json
```

Note that with the `--label-dir-name yogo_labels` option, instead of `labels/`, the folder `yogo_labels/` will be created. You should separate YOGO labels from Cellpose, and in general, important YOGO versions. E.G., you could use the pretrained model file in the name: `--label-dir-name yogo_run-name_labels`. Similarily, you should specify a tasks file name that is different from the previous task file names.

### Generating tertiary files only

You can generate only the tertiary files (`tasks.json` and `dataset_defn.yml`) with `generate_dataset_def.py` or `generate_labelstudio_tasks.json`, and they accept similar arguments to `generate_labels.py`.

## Checking Sanity

Sometimes things go sideways. Scripts like these are necessarily somewhat fragile to file structure, which people may change over time. The script `dataset_label_sanity_checker.sh` runs a very basic check over the datasets: does the number of images match the number of labels. You can run it with

```console
./dataset_label_sanity_checker.sh /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/scope-parasite-data/run-sets/ labels   # or yogo_labels, or whatever else
```
