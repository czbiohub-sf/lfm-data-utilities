# Metadata analysis tools

### plot_metadata.py
Tool to plot parameters stored in each experiment's `per_image_metadata.csv` file. All experiments run on a single instance of Oracle are stored in the same folder.
- You can either pass in a high-level directory containing multiple oracle runs (e.g `SamsungSSD/`) 
- or a specific run's folder (e.g `SamsungSSD/2023-03-04-041223`), 
- or a specific experiment (e.g `SamsungSSD/2023-03-04-041223/2023-03-04-041224_`)

The following parameters will be plotted:
- Flowrate
- Focus error
- Syringe position
- Motor position
- Total number of images in dataset
- Cell counts (note that this was only added in later, so some of the earlier datasets may leave this plot blank)

#### Example Usage
Plotting in bulk, by day:
> Enter 1 for aggregated by day, 2 for single folder/experiment: 1
> Enter top level directory: {yadayada...}/Uganda Data/curiosity
> date 1 (YYYY-MM-DD): {optional, you can leave this blank}
> date 2 (YYYY-MM-DD): {optional, you can leave this blank}
> Enter the scope name: Curiosity
> {plotting will begin now, you can press enter (in the terminal) to advance to the next plot}

Plotting a single data:
> Enter 1 for aggregated by day, 2 for single folder/experiment: 2
> Enter folder path: {yadayada...}/curiosity/2023-03-04-042543/2023-03-04-045258_
> Enter the scope name: Curiosity
> {plotting now}