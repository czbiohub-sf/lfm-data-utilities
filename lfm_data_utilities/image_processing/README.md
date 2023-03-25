# Image processing utilities

### `make_video.py`
- Generate videos from Zarr files
- Usage
> `python3 make_video.py <directory of experiments> <directory to save videos>`

Example:
> `python3 make_video.py "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/Uganda_full/curiosity" "/hpc/mydata/ilakkiyan.jeyakumar/curiosity_vids"`

Note:
1. The script will automatically create the directory where the videos will be stored if it doesn't exist already.
2. **Run this on Bruno!** - the time it takes to traverse the file tree is painfully slow when running locally and trying to access `flexo`
3. If running on Bruno, you have 1TB of storage on `/hpc/projects/first.last` - set the save directory to there, or to a location on `flexo` directly.
4. The progress bar will flicker since the videos are being generated in parallel processes but the "average" position of the flickering progress bar is still a good indication of progress. For ~100 20k image videos, I found that it takes ~5-10mins.