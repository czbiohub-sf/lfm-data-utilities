# Description
Allan deviation is a type of metric that characterizes the time domain stability of a signal. We use it to understand the correlated and uncorrelated noise sources in various aspects of the remoscope, and to optimize things like filter values when filtering raw data signals. Allan deviation shows the timescale at which the overall noise is a minimum; for example, an aggressive filter reduces stochastic noise while lowering the bandwidth. But if the filter is too aggressive, the signal becomes dominated by low frequency drift (1/f noise). 

# Usage
