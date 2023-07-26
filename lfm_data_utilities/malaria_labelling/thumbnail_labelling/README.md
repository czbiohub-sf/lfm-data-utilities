# Thumbnail Labelling

## Main Idea

Given datasets (i.e. folder of images + folder of labels), grab a thumbnail of each cell and put it in folders, sorted by class. A user can then sort those thumbnails into the correct folder. You can then use those corrected thumbnanils to update the original labels.

## Folder Structure

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
│   └── 0
├── healthy
│   ├── 0
│   ├── 1
│   └── 3
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
