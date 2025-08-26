import matplotlib.pyplot as plt
import numpy as np

parasitemias = [
    200000,
    100000,
    50000,
    25000,
    12500,
    6250,
    3125,
    0,
]

run1 = [
    17489,
    17345,
    20994,
    10863,
    3840,
    1280,
    162,
    0,
]

run2 = [
    np.nan,
    np.nan,
    8522,
    7047,
    2509,
    2043,
    400,
    0,
]

yticks = [
    40000, 20000, 10000, 5000, 2500, 1250, 625, 312, 156,
]
xticks = [
    200000,
    100000,
    50000,
    25000,
    12500,
    6250,
    3125,
    1562,
    781,
]

other = np.array([
    [25000, 3369],
    [25000, 5796],
])

plt.yscale('log', base=2)
plt.xscale('log', base=2)
plt.xticks(xticks, xticks)
plt.yticks(yticks, yticks)

plt.title("2025-08-21 Titration")
plt.xlim([1000, 250000])
plt.ylim([100, 50000])

plt.plot(parasitemias, run1, label="Run 1", marker='x')
plt.plot(parasitemias, run2, label="Run 2", marker='x')
plt.plot(parasitemias, parasitemias, linestyle='--', c='gray')
plt.scatter(other[:, 0], other[:, 1], label="Repeatability", marker='x', c='r')

plt.legend()
plt.xlabel('Expected parasitemia (parasites/uL)')
plt.ylabel('Remoscope parasitemia (parasites/uL)')

plt.savefig('figs/2025-08-21-hochuen-flowcells.png')

plt.show()
