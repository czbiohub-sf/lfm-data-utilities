# Labelling: An adventure in book keeping

Labelling data at a high quality without inducing tremendous costs (of work-hours or otherwise) is challenging, but we are attempting to make it as easy and accurate as possible. There are several pages of documentation for the data pipeline alone (and this is one of them!). Here is a short map of the documentation:

- [this document](https://github.com/czbiohub/lfm-data-utilities/blob/main/lfm_data_utilities/malaria-labelling/README.md#human-annotation) includes instruction for setting up and using LabelStudio
- [strategy.md](https://github.com/czbiohub/lfm-data-utilities/blob/main/lfm_data_utilities/malaria-labelling/scripts/strategy.md) includes the high-level strategy for training YOGO and improving data labels.
- [scripts.md](https://github.com/czbiohub/lfm-data-utilities/blob/main/lfm_data_utilities/malaria-labelling/scripts/scripts.md) includes instructions for running scripts over our data (primarily labelling)

Where you should start depends entirely on what you need to do. If you are just labelling data, you probably just need this document. If you are handling data or performing training, you should look at the other documents above. Now, let's get into it!

# Human Annotation

We will use [Label Studio](https://labelstud.io/) for human annotation.

## Preparation

### Python

Use `python3.9`. You can check your specific Python version with

```console
python3 --version
```

If you get `Python 3.9.*`, everything is good! Move on to **Installation for Annotations**.

You can install Python3.9 with your favourite package manager (like Homebrew on Mac), or from [Python's website](https://www.python.org/downloads/release/python-3913/).  You could also use [`pyenv`](https://github.com/pyenv/pyenv), but be warned that `pyenv` can be quite finicky.

You must either make sure that your `python3` executable is `Python 3.9`, or you must **remeber to always invoke the scripts with `python3.9` instead of `python3`**.

### Installation for Annotations

To perform annotations, you will need [Label Studio](https://labelstud.io/). I suggest using a [virtual environment](https://docs.python.org/3/library/venv.html). You can install Label Studio with

```console
python3 -m pip install -r requirements.txt
```

If you are on an M1 Mac, you then need to run

```console
brew install heartexlabs/tap/label-studio
```

else, if you can run

```console
python3 -m pip install label-studio
```

## Annotation

1. Start Label Studio by running: `python3 run_label_studio.py`. This assumes running on `OnDemand`. To run locally, mount `flexo` to your computer and run `python3 run_label_studio.py <path to LFM_scope folder>`
    - The `LFM_scope` folder has path (relative to `flexo`) `flexo/MicroscopyData/Bioengineering/LFM_scope/`. So if I've mounted `flexo` on my Mac, `<path to LFM_scope folder>` should be `/Volumes/flexo/MicroscopyData/Bioengineering/LFM_scope/`
2. In LabelStudio, click `Create Project`
  - Name your project the name of the run folder, or else
  - Go to "Labelling Setup" and click "Custom Template" on the left. Under the "Code" section, paste in the following XML and save
```xml
<View>
    <Image name="image" value="$image" zoom="true" zoomControl="true"/>
    <Header value="RectangleLabels"/>
    <RectangleLabels name="label" toName="image" canRotate="false" strokeWidth="3" opacity=".0">
        <Label value="healthy" background="#27b94c" category="0"/>
        <Label value="ring" background="rgba(250, 100, 150, 1)" category="1"/>
        <Label value="trophozoite" background="#eebd68" category="2"/>
        <Label value="schizont" background="rgba(100, 180, 255, 1)" category="3"/>
        <Label value="gametocyte" background="rgba(255, 200, 255, 1)" category="4"/>
        <Label value="wbc" background="#9cf2ec" category="5"/>
        <Label value="misc" background="rgba(100, 100, 100, 1)" category="6"/>
    </RectangleLabels>
</View>
```
  - Go to the "Data Import" tab, click "Upload Files", and import the `yogo_labelled_tasks.json` in the run folder that you are annotating. It will be somewhere in `.../scope-parasite-data/run-sets`. **Note**: If this fails (e.g "too many SQL variables") - try creating the project _without_ doing this "Data Import" step. Then, once the project has been created, upload the data after.
  - Click "Save"

and you are ready to annotate!

## On `tasks.json` files

As YOGO improves, we can use it to label our data. YOGO labels will have `yogo_labelled_tasks.json` as it's LabelStudio tasks file. Cellpose labels still have `tasks.json` as it's tasks file.

## Exporting from Label Studio

After annotating your images, it is time to export. If you use the "Export" button on the UI, LabelStudio will also export your unchanged images. We do not want that - we want just the labels. Therefore, we will use their API endpoint.

Click on the symbol for your account on the upper-right of the screen (for me, it is a circle with "AJ" in the center), and go to "Account & Settings". There, copy your "Authorization Token".

Note the project ID from the URL of the project. Navigate to the project from which you are exporting labels. The URL should look something like:

```
http://localhost:8080/projects/13/data?tab=9&task=2

                               ^^ "13" is the project id
```

Now, depending on the export format, run the following in a *new terminal*, substituting in the project ID and the auth token.

### Exporting labels for YOGO

When exporting for training (i.e. you've completed labelling the entire batch of images), use the following command:

```console
curl -X GET "http://localhost:8080/api/projects/<project id>/export?exportType=YOLO&download_resources=false" -H "Authorization: Token <paste the Auth. token here>" --output annotations.zip
```

Send that folder to Axel. Thank you!

### Exporting `tasks.json` for further labelling or review

```console
curl -X GET "http://localhost:8080/api/projects/<project id>/export?exportType=JSON&download_resources=false" -H "Authorization: Token <paste the Auth. token here>" --output tasks.json
```

## Troubleshooting

### "Package Not Found" during installation

If your `pip` version is really low (e.g. version 9), try `python3 -m pip install --upgrade pip`. This could also be a sign of your python version being quite low (which occurs, e.g., with a base `conda` environment). Double check with `python3 --version`. It should be at least python 3.7.
