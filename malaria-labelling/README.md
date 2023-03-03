# Human Annotation

We will use [Label Studio](https://labelstud.io/) for human annotation.

## Preparation

### Python

Use `python3.9`. You can check your specific Python version with

```console
python3 --version
```

You can install Python3.9 with your favourite package manager (like Homebrew on Mac), or from [Python's website](https://www.python.org/downloads/release/python-3913/). 

**Remeber to always invoke the scripts with `python3.9` instead of `python3`**.

You could also use [`pyenv`](https://github.com/pyenv/pyenv), but be warned that `pyenv` can be quite finicky.

### Installation for Annotations

To perform annotations, you will need [Label Studio](https://labelstud.io/). I suggest using a [virtual environment](https://docs.python.org/3/library/venv.html). You can install it with

```console
python3 -m pip install -r requirements.txt
```

If you are on an M1 Mac, you then need to run

```
brew install heartexlabs/tap/label-studio
```

else, if you can run

```console
python3 -m pip install label-studio
```

## Annotating

1. Start Label Studio by running: `python3 run_label_studio.py`. This assumes running on `OnDemand`. To run locally, mount `flexo` to your computer and run `python3 run_label_studio.py <path to run-set folder>`
2. In LabelStudio, click `Create Project`
  - Name your project the name of the run folder, or else
  - Go to "Labelling Setup" and click "Custom Template" on the left. Under the "Code" section, paste in the following XML and save
```xml
<View>
    <Image name="image" value="$image" zoom="true" zoomControl="true" />
    <Header value="RectangleLabels" />
    <RectangleLabels
        name="label"
        toName="image"
        canRotate="false"
        strokeWidth="3"
        opacity=".0"
    >
        <Label value="healthy" background="rgba(200, 255, 200, 1)" />
        <Label value="ring" background="rgba(250, 100, 150, 1)" />
        <Label value="trophozoite" background="rgba(255, 220, 200, 1)" />
        <Label value="schizont" background="rgba(100, 180, 255, 1)" />
        <Label value="gametocyte" background="rgba(255, 200, 255, 1)" />
        <Label value="wbc" background="rgba(200, 250, 255, 1)" />
        <Label value="misc" background="rgba(100, 100, 100, 1)" />
    </RectangleLabels>
</View>
```
  - Go to the "Data Import" tab, click "Upload Files", and import the `tasks.json` in the run folder that you are annotating. It will be somewhere in `.../scope-parasite-data/run-sets`. **Note**: If this fails (e.g "too many SQL variables") - try creating the project _without_ doing this "Data Import" step. Then, once the project has been created, upload the data after.
  - Click "Save"

and you are ready to annotate!

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

### "error: unrecognized arguments: Scope/scope-parasite-data/run-sets"

You need to either quote the entire path, or escape the space in "LFM Scope". For example, try

`".../LFM Scope/scope-parasite-data"` or `.../LFM\ Scope/scope-parasite-data`

They're equivalent, pick your poison.
