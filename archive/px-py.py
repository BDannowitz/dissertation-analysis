import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import MySQLdb as Mdb
from matplotlib._cm import cubehelix
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace

import sys
sys.path.appand('modules')
from servers import server_dict

plt.ion()

# Define a function to make the ellipses
def ellipse(ra, rb, ang, x0, y0, Nb=100):
    xpos, ypos = x0, y0
    radm, radn = ra, rb
    an = ang
    co, si = np.cos(an), np.sin(an)
    the = linspace(0, 2 * np.pi, Nb)
    X = radm * np.cos(the) * co - si * radn * np.sin(the) + xpos
    Y = radm * np.cos(the) * si + co * radn * np.sin(the) + ypos
    return X, Y

# Define the x and y data
# For example just using random numbers
# x = np.random.randn(10000)
# y = np.random.randn(10000)

db = Mdb.connect(read_default_group='guest', db='merged_roadset57_R004',
                     host='e906-db3.fnal.gov', port=server_dict['e906-db3.fnal.gov']['port'])
cur = db.cursor()

query = """SELECT dpx, dpy
           FROM jDimuon
           WHERE dpx BETWEEN -2.5 AND 2.5
           AND dpy BETWEEN -2.5 AND 2.5
           LIMIT 1000"""

cur.execute(query)
items = cur.fetchall()
x = []
y = []
for item in items:
    x.append(item[0])
    y.append(item[1])

if db:
    db.close()

# Set up default x and y limits
xlims = [min(x), max(x)]
ylims = [min(y), max(y)]

# Set up your x and y labels
xlabel = '$\mathrm{p_x}$'
ylabel = '$\mathrm{p_y}$'

# Define the locations for the axes
# left, width = 0.12, 0.55
# bottom, height = 0.12, 0.55
# bottom_h = left_h = left + width + 0.02

# gs = gridspec.GridSpec(3,3)
# rect_histx = plt.subplot(gs[0,:-1])
# rect_histy = plt.subplot(gs[1:,-1])
# rect_temperature = plt.subplot(gs[1:,:-1])

# Set up the size of the figure
fig = plt.figure(1, figsize=(9.5, 9))

gs = gridspec.GridSpec(3,3)
rect_temperature = fig.add_subplot(gs[1:,:-1], sharex=rect_histx, sharey=rect_histy)
rect_histx = fig.add_subplot(gs[0,:-1], sharex=rect_temperature)
rect_histy = fig.add_subplot(gs[1:,-1], sharey=rect_temperature)


# Set up the geometry of the three plots
# rect_temperature = [left, bottom, width, height]  # dimensions of temp plot
# rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
# rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram


# Make the three plots
axTemperature = plt.axes(rect_temperature)  # temperature plot
axHistx = plt.axes(rect_histx)  # x histogram
axHisty = plt.axes(rect_histy)  # y histogram

# Remove the inner axes numbers of the histograms
nullfmt = NullFormatter()
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# Find the min/max of the data
xmin = min(xlims)
xmax = max(xlims)
ymin = min(ylims)
ymax = max(y)

# Make the 'main' temperature plot
# Define the number of bins
nxbins = 50
nybins = 50
nbins = 100

xbins = linspace(start=xmin, stop=xmax, num=nxbins)
ybins = linspace(start=ymin, stop=ymax, num=nybins)
xcenter = (xbins[0:-1] + xbins[1:]) / 2.0
ycenter = (ybins[0:-1] + ybins[1:]) / 2.0
aspectratio = 1.0 * (xmax - 0) / (1.0 * ymax - 0)

H, xedges, yedges = np.histogram2d(y, x, bins=(ybins, xbins))

X = xcenter
Y = ycenter
Z = H

# Plot the temperature data
cax = (axTemperature.imshow(H, extent=[xmin, xmax, ymin, ymax],
                            interpolation='nearest', origin='lower', aspect=aspectratio, cmap='GnBu'))

# Spectral, summer, coolwarm, Wistia_r, pink_r, Set1, Set2, Set3, brg_r, Dark2, prism, PuOr_r, afmhot_r,
# terrain_r, PuBuGn_r, RdPu, gist_ncar_r, gist_yarg_r, Dark2_r, YlGnBu, RdYlBu, hot_r, gist_rainbow_r, gist_stern,
# PuBu_r, cool_r, cool, gray, copper_r, Greens_r, GnBu, gist_ncar, spring_r, gist_rainbow, gist_heat_r, Wistia, OrRd_r,
# CMRmap, bone, gist_stern_r, RdYlGn, Pastel2_r, spring, terrain, YlOrRd_r, Set2_r, winter_r, PuBu, RdGy_r, spectral,
# rainbow, flag_r, jet_r, RdPu_r, gist_yarg, BuGn, Paired_r, hsv_r, bwr, cubehelix, Greens, PRGn, gist_heat, spectral_r,
# Paired, hsv, Oranges_r, prism_r, Pastel2, Pastel1_r, Pastel1, gray_r, jet, Spectral_r, gnuplot2_r, gist_earth,
# YlGnBu_r, copper, gist_earth_r, Set3_r, OrRd, gnuplot_r, ocean_r, brg, gnuplot2, PuRd_r, bone_r, BuPu, Oranges,
# RdYlGn_r, PiYG, CMRmap_r, YlGn, binary_r, gist_gray_r, Accent, BuPu_r, gist_gray, flag, bwr_r, RdBu_r, BrBG, Reds,
# Set1_r, summer_r, GnBu_r, BrBG_r, Reds_r, RdGy, PuRd, Accent_r, Blues, autumn_r, autumn, cubehelix_r,
# nipy_spectral_r, ocean, PRGn_r, Greys_r, pink, binary, winter, gnuplot, RdYlBu_r, hot, YlOrBr, coolwarm_r,
# rainbow_r, Purples_r, PiYG_r, YlGn_r, Blues_r, YlOrBr_r, seismic, Purples, seismic_r, RdBu, Greys, BuGn_r,
# YlOrRd, PuOr, PuBuGn, nipy_spectral, afmhot

# Plot the temperature plot contours
contourcolor = 'white'
xcenter = np.mean(x)
ycenter = np.mean(y)
ra = np.std(x)
rb = np.std(y)
ang = 0

X, Y = ellipse(ra, rb, ang, xcenter, ycenter)
axTemperature.plot(X, Y, "k:", ms=1, linewidth=2.0)
axTemperature.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data', xytext=(10, 10),
                       textcoords='offset points', horizontalalignment='right',
                       verticalalignment='bottom', fontsize=25)

X, Y = ellipse(2 * ra, 2 * rb, ang, xcenter, ycenter)
axTemperature.plot(X, Y, "k:", ms=1, linewidth=2.0)
# axTemperature.plot(X, Y, "k:", color=contourcolor, ms=1, linewidth=2.0)
axTemperature.annotate('$2\\sigma$', xy=(X[15], Y[15]), xycoords='data', xytext=(10, 10),
                       textcoords='offset points', horizontalalignment='right',
#                       verticalalignment='bottom', fontsize=25, color=contourcolor)
                        verticalalignment='bottom', fontsize=25)

X, Y = ellipse(3 * ra, 3 * rb, ang, xcenter, ycenter)
# axTemperature.plot(X, Y, "k:", color=contourcolor, ms=1, linewidth=2.0)
axTemperature.plot(X, Y, "k:", ms=1, linewidth=2.0)
axTemperature.annotate('$3\\sigma$', xy=(X[15], Y[15]), xycoords='data', xytext=(10, 10),
                       textcoords='offset points', horizontalalignment='right',
#                       verticalalignment='bottom', fontsize=25, color=contourcolor)
                        verticalalignment='bottom', fontsize=25)

# Plot the axes labels
axTemperature.set_xlabel(xlabel, fontsize=25)
axTemperature.set_ylabel(ylabel, fontsize=25)

#Make the tickmarks pretty
ticklabels = axTemperature.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('serif')

ticklabels = axTemperature.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('serif')

# Set up the plot limits
axTemperature.set_xlim(xlims)
axTemperature.set_ylim(ylims)

# Set up the histogram bins
xbins = np.arange(xmin, xmax, (xmax - xmin) / nbins)
ybins = np.arange(ymin, ymax, (ymax - ymin) / nbins)

# Plot the histograms
axHistx.hist(x, bins=xbins, color='blue')
axHisty.hist(y, bins=ybins, orientation='horizontal', color='blue')

# Set up the histogram limits
axHistx.set_xlim(min(x), max(x))
axHisty.set_ylim(min(y), max(y))

#Make the tickmarks pretty
ticklabels = axHistx.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(12)
    label.set_family('serif')

#Make the tickmarks pretty
ticklabels = axHisty.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(12)
    label.set_family('serif')

#Cool trick that changes the number of tickmarks for the histogram axes
axHisty.xaxis.set_major_locator(MaxNLocator(4))
axHistx.yaxis.set_major_locator(MaxNLocator(4))

plt.setp(rect_temperature.get_xticklabels(), visible=True)
plt.setp(rect_temperature.get_yticklabels(), visible=True)

plt.tight_layout()

#Show the plot
plt.draw()
plt.show()

# Save to a File
filename = 'myplot'
plt.savefig(filename + '.pdf', format='pdf', transparent=True)

