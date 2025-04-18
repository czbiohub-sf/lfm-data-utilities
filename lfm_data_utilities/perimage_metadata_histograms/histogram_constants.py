# -------------------- General ---------------------
IMCOUNT_TARGET = 20000 - 1

# -------------------- Focus targets ---------------------
FOCUS_TARGET = 0
FOCUS_ABS_MARGIN = 2

MIN_FOCUS_TARGET = FOCUS_TARGET - FOCUS_ABS_MARGIN
MAX_FOCUS_TARGET = FOCUS_TARGET + FOCUS_ABS_MARGIN

# -------------------- Flowrate targets ---------------------
FLOWRATE_TARGET = 7.58
FLOWRATE_PERC_MARGIN = 0.2

MIN_FLOWRATE_TARGET = FLOWRATE_TARGET * (1 - FLOWRATE_PERC_MARGIN)
MAX_FLOWRATE_TARGET = FLOWRATE_TARGET * (1 + FLOWRATE_PERC_MARGIN)
