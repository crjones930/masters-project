import numpy as np
import matplotlib.pyplot as plt
from time import strftime, gmtime

dump_dir = ""

time, log_ke = np.genfromtxt(dump_dir + "time_series.dat", skip_header=1, usecols=[0,1], unpack=True)

checkpoints = np.arange(len(time))

fig, ax1 = plt.subplots()
ax1.semilogy(time, 10**log_ke, c="tab:red") # , ls="", marker=".", ms="1")
ax1.set_xlabel("Time")
ax1.set_ylabel("total ke")
ax1.set_ylim(1e41, 4e44)
ax1.grid(True, alpha=.3)
ax2 = ax1.twiny()

ax2.set_xlabel("model")

ax1.set_xlim(min(time), max(time))
ax2.set_xlim(min(checkpoints), max(checkpoints))

timestr = strftime("%d-%b-%Y_%H-%M-%S%p", gmtime())

fig.savefig(f"total_ke_until_dump_{max(checkpoints)}_{timestr}.png")