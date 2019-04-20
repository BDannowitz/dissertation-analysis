import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in SeaQuest data
e906 = pd.read_csv('e906.dat')

# "Read in" NA51 data
na51 = pd.DataFrame([[0.145,0.212,0.18,1.961,0.2462,0.2462,0.0]], columns=e906.columns)

# Read in E-866 data
e866 = pd.read_csv('e866.dat')

# Read in BS15 data and calculate upper and lower dbar/ubar bounds
bs15 = pd.read_csv('BS15nlo.dat')
bs15['hi_dbar_ubar'] = bs15.apply(lambda x: max(x.dbar1/x.ubar1, x.dbar2/x.ubar2), axis=1)
bs15['low_dbar_ubar'] = bs15.apply(lambda x: min(x.dbar1/x.ubar1, x.dbar2/x.ubar2), axis=1)
# Cut off values here for aesthetics
bs15 = bs15.query('x<0.6025')

# Read in CT14 data and calculate upper and lower dbar/ubar bounds
ct14 = pd.read_csv('CT14nlo.dat')
#ct14 = pd.read_csv('ct14-q2-25-60.dat')
ct14['hi_dbar_ubar'] = ct14.apply(lambda x: max(x.dbar1/x.ubar1, x.dbar2/x.ubar2), axis=1)
ct14['low_dbar_ubar'] = ct14.apply(lambda x: min(x.dbar1/x.ubar1, x.dbar2/x.ubar2), axis=1)
# Cut off values here for aesthetics
ct14 = ct14.query('x<0.6025')

# Read in MMHT2014 data and calculate upper and lower dbar/ubar bounds
mmht14 = pd.read_csv('MMHT14nlo.dat')
#mmht14 = pd.read_csv('mmht2014-q2-25-60.dat')
mmht14['hi_dbar_ubar'] = mmht14.apply(lambda x: max(x.dbar1/x.ubar1, x.dbar2/x.ubar2), axis=1)
mmht14['low_dbar_ubar'] = mmht14.apply(lambda x: min(x.dbar1/x.ubar1, x.dbar2/x.ubar2), axis=1)
# Cut off values here for aesthetics
mmht14 = mmht14.query('x<0.6025')

# Choose RGB colors for global PDFs
colors = [(0.0, 0.4470588235294118, 0.6980392156862745),
          (0.0, 0.6196078431372549, 0.45098039215686275),
          (0.8352941176470589, 0.3686274509803922, 0.0)]

# Create data for E-866 systematic bar
e866_sys = []
e866_edges = []
# Location of the bottom of the bar
e866_sys_base = 0.5
for i in range(0,len(e866.sys.values)):
    e866_edges.append(e866.low_x2[i])
    e866_edges.append(e866.high_x2[i])
    e866_sys.append(e866_sys_base+e866.sys[i])
    e866_sys.append(e866_sys_base+e866.sys[i])
    
# Create data for SeaQuest systematic bar
e906_sys = []
e906_edges = []
# Location  of the bottom of the bar
e906_sys_base = 0.13
for i in range(0,len(e906.sys)):
    e906_edges.append(e906.low_x2[i])
    e906_edges.append(e906.high_x2[i])
    e906_sys.append(e906_sys_base+e906.sys[i])
    e906_sys.append(e906_sys_base+e906.sys[i])


#### BEGIN PLOTTING ##############

fig, ax = plt.subplots(1, figsize=(8, 5))

# BS15
ax.plot(bs15.x, bs15.dbar1/bs15.ubar1, ls=':', lw=0.5, color=colors[0], label='_nolabel_')
ax.plot(bs15.x, bs15.dbar2/bs15.ubar2, ls=':', lw=0.5, color=colors[0], label='_nolabel_')
ax.fill_between(bs15.x, bs15.hi_dbar_ubar, bs15.low_dbar_ubar,
                alpha=0.2, label='BS15 NLO', color=colors[0])
# CT14
ax.plot(ct14.x, ct14.dbar1/ct14.ubar1, ls=':', lw=0.5, color=colors[1], label='_nolabel_')
ax.plot(ct14.x, ct14.dbar2/ct14.ubar2, ls=':', lw=0.5, color=colors[1], label='_nolabel_')
ax.fill_between(ct14.x, ct14.hi_dbar_ubar, ct14.low_dbar_ubar,
                alpha=0.2, label='CT14 NLO', color=colors[1])
# MMHT2014
ax.plot(mmht14.x, mmht14.dbar1/mmht14.ubar1, ls=':', lw=0.5, color=colors[2], label='_nolabel_')
ax.plot(mmht14.x, mmht14.dbar2/mmht14.ubar2, ls=':', lw=0.5, color=colors[2], label='_nolabel_')
ax.fill_between(mmht14.x, mmht14.hi_dbar_ubar, mmht14.low_dbar_ubar,
                alpha=0.2, label='MMHT2014 NLO', color=colors[2])

# Data
ax.errorbar(na51.x2, na51.dbar_ubar,
            xerr=[na51.x2-na51.low_x2, na51.high_x2-na51.x2],
            yerr=[na51.low_stat, na51.high_stat],
	    fmt='b^',
   	    label='NA51')
ax.errorbar(e866.x2, e866.dbar_ubar,
	    xerr=[e866.x2-e866.low_x2, e866.high_x2-e866.x2],
            yerr=[e866.low_stat, e866.high_stat],
	    fmt='ks',
	    label='E-866 NLO')
ax.errorbar(e906.x2, e906.dbar_ubar,
            xerr=[e906.x2-e906.low_x2, e906.high_x2-e906.x2],
            yerr=[e906.low_stat, e906.high_stat],
            fmt='ro',
	    label='SeaQuest LO')

# Systematics
ax.fill_between(e866_edges, e866_sys, e866_sys_base, facecolor='k', alpha=0.4)
ax.fill_between(e906_edges, e906_sys, e906_sys_base, facecolor='red', alpha=0.4)

# Systematics label
ax.text(0.03, 0.56, 'E-866 sys.')
ax.text(0.105, 0.28, 'SeaQuest sys.')

# Axis Labels
ax.set_ylabel(r'$\frac{\bar{d}(x)}{\bar{u}(x)}$', rotation=0, fontsize=32)
ax.set_xlabel(r'$x$', fontsize=25)
ax.yaxis.labelpad = 30 # Add space between y-axis label and y-axis

# Aesthetics
ax.grid()
ax.set_xlim([0,0.607])
ax.set_ylim([0,3.0])
plt.axhline(y=1.0, xmin=0.0, xmax=1.0, linewidth=1, color = 'k') # Line at 1
ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.8, numpoints=1)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=14)
fig.set_tight_layout(True)

# Output image
plt.savefig('dbar_ubar.png', dpi=800)


## BEGIN LOG RATIO SECTION #####################

# Translate values to plot ln(dbar/ubar)
e866['log_dbar_ubar'] = np.log(e866.dbar_ubar)
e866['log_low_stat'] = np.divide(e866.low_stat,e866.dbar_ubar)
e866['log_high_stat'] = np.divide(e866.high_stat,e866.dbar_ubar)
e866['log_sys'] = np.divide(e866.sys,e866.dbar_ubar)

e906['log_dbar_ubar'] = np.log(e906.dbar_ubar)
e906['log_low_stat'] = np.divide(e906.low_stat,e906.dbar_ubar)
e906['log_high_stat'] = np.divide(e906.high_stat,e906.dbar_ubar)
e906['log_sys'] = np.divide(e906.sys,e906.dbar_ubar)

na51['log_dbar_ubar'] = np.log(na51.dbar_ubar)
na51['log_low_stat'] = np.divide(na51.low_stat,na51.dbar_ubar)
na51['log_high_stat'] = np.divide(na51.high_stat,na51.dbar_ubar)

# Translate sys. uncertainties
e866_log_sys = []
e866_log_sys_base = -0.2
for i in range(0,len(e866.log_sys.values)):
    e866_log_sys.append(e866_log_sys_base+e866.log_sys[i])
    e866_log_sys.append(e866_log_sys_base+e866.log_sys[i])
    
e906_log_sys = []
e906_log_sys_base = -0.4
for i in range(0,len(e906.log_sys)):
    e906_log_sys.append(e906_log_sys_base+e906.log_sys[i])
    e906_log_sys.append(e906_log_sys_base+e906.log_sys[i])

# Translate global fit values
ct14['log_hi_dbar_ubar'] = np.log(ct14.hi_dbar_ubar)
ct14['log_low_dbar_ubar'] = np.log(ct14.low_dbar_ubar)
bs15['log_hi_dbar_ubar'] = np.log(bs15.hi_dbar_ubar)
bs15['log_low_dbar_ubar'] = np.log(bs15.low_dbar_ubar)
mmht14['log_hi_dbar_ubar'] = np.log(mmht14.hi_dbar_ubar)
mmht14['log_low_dbar_ubar'] = np.log(mmht14.low_dbar_ubar)

## BEGIN PLOTTING LOG RATIO ##################

fig, ax = plt.subplots(1, figsize=(8, 5))

# BS15
ax.plot(bs15.x, bs15.log_hi_dbar_ubar, ls=':', lw=0.5, color=colors[0], label='_nolegend_')
ax.plot(bs15.x, bs15.log_low_dbar_ubar, ls=':', lw=0.5, color=colors[0], label='_nolegend_')
ax.fill_between(bs15.x, bs15.log_hi_dbar_ubar, bs15.log_low_dbar_ubar,
                alpha=0.2, label='BS15 NLO', color=colors[0])
# CT14
ax.plot(ct14.x, ct14.log_hi_dbar_ubar, ls=':', lw=0.5, color=colors[1], label='_nolegend_')
ax.plot(ct14.x, ct14.log_low_dbar_ubar, ls=':', lw=0.5, color=colors[1], label='_nolegend_')
ax.fill_between(ct14.x, ct14.log_hi_dbar_ubar, ct14.log_low_dbar_ubar,
                alpha=0.2, label='CT14 NLO', color=colors[1])
# MMHT2014
ax.plot(mmht14.x, mmht14.log_hi_dbar_ubar, ls=':', lw=0.5, color=colors[2], label='_nolegend_')
ax.plot(mmht14.x, mmht14.log_low_dbar_ubar, ls=':', lw=0.5, color=colors[2], label='_nolegend_')
ax.fill_between(mmht14.x, mmht14.log_hi_dbar_ubar, mmht14.log_low_dbar_ubar,
                alpha=0.2, label='MMHT2014 NLO', color=colors[2])

# Data
ax.errorbar(na51.x2, na51.log_dbar_ubar,
            xerr=[na51.x2-na51.low_x2, na51.high_x2-na51.x2],
            yerr=[na51.log_low_stat, na51.log_high_stat],
	    fmt='b^',
	    label='NA51')
ax.errorbar(e866.x2, e866.log_dbar_ubar,
	    xerr=[e866.x2-e866.low_x2, e866.high_x2-e866.x2],
            yerr=[e866.log_low_stat, e866.log_high_stat],
	    fmt='ks',
	    label='E-866 NLO')
ax.errorbar(e906.x2, e906.log_dbar_ubar,
            xerr=[e906.x2-e906.low_x2, e906.high_x2-e906.x2],
            yerr=[e906.log_low_stat, e906.log_high_stat],
            fmt='ro',
	    label='SeaQuest LO')

# Sys. unc. bar
ax.fill_between(e866_edges, e866_log_sys, e866_log_sys_base, facecolor='k', alpha=0.4)
ax.fill_between(e906_edges, e906_log_sys, e906_log_sys_base, facecolor='red', alpha=0.4)
ax.text(0.02, -0.15, 'E-866 sys.')
ax.text(0.105, -0.32, 'SeaQuest sys.')

# Axis Labels
ax.set_ylabel(r'$ln\left(\frac{\bar{d}(x)}{\bar{u}(x)}\right)$', rotation=0, fontsize=22)
ax.set_xlabel(r'$x$', fontsize=22)
ax.yaxis.labelpad = 30 # Add space between y-axis label and y-axis

# Aesthetics
ax.grid()
ax.set_xlim([0,0.607])
ax.set_ylim([-1.3,1.3])
plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1, color = 'k') # Line at 1
ax.legend(loc='lower left', fontsize=10, frameon=True, framealpha=0.8, numpoints=1)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=14)
fig.set_tight_layout(True)

# Output
plt.savefig('dbar_ubar_log.png', dpi=800)

