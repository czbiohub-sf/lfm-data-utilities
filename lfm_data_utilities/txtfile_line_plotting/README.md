# .txt file line plotter

### line_plotter.py
Tool to get data from a folde of .txt files and plot overlapping line plots

Inputs:
- Path to folder containing datasets: eg. /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/SSAF_Uganda_full/curiosity
- (Optional) custom ylabel: eg. SSAF error
- (Optional) custom title: eg. Curiosity SSAF

If less than 10 datasets are included in the folder, the plot will display a legend for each dataset (with RMS value). If there are more than 10 datasets, no legend will be displayed.