#!/usr/bin/python

import numpy as np
import pandas as pd
import pymysql as mdb
import matplotlib.pyplot as plt
from matplotlib.pylab import text, rcParams, rc
from uncertainties import unumpy as unp

rcParams['figure.figsize'] = 12, 8
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
rc('text', usetex=True)
font = {'family' : 'serif', 'weight': 'bold', 'size': 16}
rc('font', **font)

road57_min = 8912; road57_max = 10420
road62_min = 11075; road62_max = 12438
road67_min = 12525; road67_max = 15789
roadset = { 57: {'label': 'R57',
                 'range': (road57_min, road57_max)},
            62: {'label': 'R62',
                 'range': (road62_min, road62_max)},
            67: {'label': 'R67',
                 'range': (road67_min, road67_max)}}

def get_nhist(df, query, kin, bin_range, nbins=20, binned='weight'):

    weights = unp.nominal_values(df.query(query)[binned])

    # Get histogram and bin edges
    hist, bin_edges = np.histogram(df.query(query)[kin],
                                   range=bin_range,
                                   bins=nbins)
                                   #weights=weights)
    # Get normalized histogram
    nhist, bin_edges = np.histogram(df.query(query)[kin],
                                    range=bin_range,
                                    bins=nbins,
                                    density=True)
                                    #weights=weights)
    
    # Get sums of squares of weights for histogram
    #if binned in ['weight', 'weight_keff']:
    #    binned = ("%s_sq" % (binned))
    #	sqhist, bin_edges = np.histogram(df.query(query)[kin],
    #					      range=bin_range,
    #					      bins=nbins,
    #					      weights=weights)

    # Get array of bin values with uncertainty
    #uhist = unp.uarray(hist, std_devs=np.sqrt(sqhist))
    uhist = unp.uarray(hist, std_devs=np.sqrt(hist))

    # Get scale factor for normalization
    scale = 0
    for i in range(0, len(nhist)-1):
        if nhist[i] > 0:
            scale = np.divide(hist, nhist)[i]

    nuhist = uhist/scale
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return (nuhist, bin_centers)


def get_hist_range(df, query_list, kin, nbins=20):
    
    min_val = None
    max_val = None
    for query in query_list:
        hist, bin_edges = np.histogram(df.query(query)[kin], bins=nbins)
        if not min_val:
            min_val = bin_edges[0]
        elif bin_edges[0] < min_val:
            min_val = bin_edges[0]
        if not max_val:
            max_val = bin_edges[-1]
        elif bin_edges[0] < max_val:
            max_val = bin_edges[-1]
    
    return (min_val, max_val)


def plot_chisq(bin_centers, hist1, hist2, ax):
    
    ylims = ax.get_ylim()
    ylen = abs(ylims[1]-ylims[0])

    new_ylim = (-(ylen/2.0), ylen/2.0)
    
    diff = np.subtract(hist2, hist1)
    
    ax1 = ax.twinx()
    
    ax1.errorbar(bin_centers,
                        unp.nominal_values(diff),
                        yerr=unp.std_devs(diff),
                        lw=1,
                        fmt='o',
                        color='red',
                        label='templabel')
    
    ax1.axhline(color='red')
    
    ax1.set_ylim(new_ylim)
    
    degrees = len(diff) - 1
    chisq = 0.0
    for x in diff:
        if unp.std_devs(x) is None or unp.std_devs(x)==0:
            degrees = degrees-1
        else:
            chisq += (unp.nominal_values(x)/unp.std_devs(x))**2
    
    if degrees > 0:
        chisq_pdf = np.round(chisq / degrees,2)
    else:
        chisq_pdf = np.NaN
        
    text(0.24, 0.95, r'$\chi^2/d.o.f. = %.02f$' % (chisq_pdf),
               ha='center', va='center', transform=ax1.transAxes)
    
    return ax1


def plt_kin_ax(df, kin, targetPos, target, ax, roadset1, roadset2, chisq):
    
    nbins = 15
    
    query1 = ('(%i <= runID <= %i)' % roadset1['range'])
    query2 = ('(%i <= runID <= %i)' % roadset2['range'])
    if targetPos:
        query1 += (' and targetPos==%i' % targetPos)
        query2 += (' and targetPos==%i' % targetPos)
        
    query_list = [query1, query2]
    kin_range = get_hist_range(df, query_list, kin, nbins=nbins)
    
    original_kin = kin
    if roadset2['label'] in ('R67', 'R70'):
        if kin in ('pz1','pt1'):
            list_kin = list(kin)
            list_kin[2]='2'
            kin = "".join(list_kin)
        if kin in ('pz2','pt2'):
            list_kin = list(kin)
            list_kin[2]='1'
            kin = "".join(list_kin)

    if targetPos in [1,3,5,6,7]:
	binned = 'weight'
    else:
	binned = 'weight_keff'

    hist1, bin_centers = get_nhist(df, query1, kin, kin_range, nbins=nbins, binned=binned)
    
    ax.errorbar(bin_centers,
                unp.nominal_values(hist1),
                yerr=unp.std_devs(hist1),
                label=roadset1['label'],
                lw=1,
                drawstyle = 'steps-mid')
    
    kin = original_kin
    
    if roadset2['label'] in ('R67', 'R70'):
        if kin in ('pz1','pt1'):
            list_kin = list(kin)
            list_kin[2]='2'
            kin = "".join(list_kin)
        if kin in ('pz2','pt2'):
            list_kin = list(kin)
            list_kin[2]='1'
            kin = "".join(list_kin)
    hist2, bin_centers = get_nhist(df, query2, kin, kin_range, nbins=nbins, binned=binned)
    
    ax.errorbar(bin_centers,
                       unp.nominal_values(hist2),
                       yerr=unp.std_devs(hist2),
                       label=roadset2['label'],
                       lw=1,
                       drawstyle = 'steps-mid')
    
    
    ax_list = [ax]
    
    ax.relim()
    ax.autoscale_view()
    ax.set_ylim(0, ax.get_ylim()[1])
    
    if chisq:
        ax1 = plot_chisq(bin_centers, hist1, hist2, ax)
        ax_list += [ax1]
    
    ax.set_title(target)
    ax.set_ylabel('')
    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',        # ticks along the bottom edge are off
        labelleft='off',   # labels along the bottom edge are off
        labeltop='off',
        top='off')
    ax.ticklabel_format(axis='y', style='plain')
    
    if chisq:
        ax1.set_ylabel('')
        ax1.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            right='off',       # ticks along the bottom edge are off
            labelright='off',  # labels along the bottom edge are off
            labeltop='off',
            top='off')
        ax1.ticklabel_format(axis='y', style='plain')
    
    return ax_list


def plt_kin(df, kin, axis_label, rsn1, rsn2, chisq=True):

    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = \
        plt.subplots(2, 4, sharex=True, figsize=(22.5,15))
    
    roadset1 = roadset[rsn1]
    roadset2 = roadset[rsn2]
    
    plt_kin_ax(df, kin, 2, "Empty", ax1, roadset1, roadset2, chisq)
    plt_kin_ax(df, kin, 4, "None", ax2, roadset1, roadset2, chisq)
    plt_kin_ax(df, kin, 1, "LH2", ax3, roadset1, roadset2, chisq)
 
    # Process the legend here
    # Make sure to have the chisq points labelled properly and the line at
    #   zero included and labelled
    ax_lst = plt_kin_ax(df, kin, 3, "LD2", ax4, roadset1, roadset2, chisq)
    display = (0,1)
    if chisq:
        zero = plt.Line2D((0,1),(0,0), color='red', linestyle='-')
    handles = []; labels = [];
    for ax in ax_lst:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles += ax_handles
        labels += ax_labels
    if chisq:
        ax4.legend([handle for i,handle in enumerate(handles)] + [zero],
                   [label for i,label in enumerate(labels) if i in display] +
                   ['%s-%s' % (roadset2['label'], roadset1['label']),
                    '%s-%s=0' % (roadset2['label'], roadset1['label'])],
                   bbox_to_anchor=(1.05, 1), loc=2)
    else:
        ax4.legend([handle for i,handle in enumerate(handles)],
                   [label for i,label in enumerate(labels)],
                   bbox_to_anchor=(1.05, 1), loc=2)
 
    plt_kin_ax(df, kin, 6, "C", ax5, roadset1, roadset2, chisq)
    ax5.set_xlabel(axis_label, fontsize=24)
    plt_kin_ax(df, kin, 5, "Fe", ax6, roadset1, roadset2, chisq)
    ax6.set_xlabel(axis_label, fontsize=24)
    plt_kin_ax(df, kin, 7, "W", ax7, roadset1, roadset2, chisq)
    ax7.set_xlabel(axis_label, fontsize=24)
    plt_kin_ax(df, kin, None, "All", ax8, roadset1, roadset2, chisq)
    ax8.set_xlabel(axis_label, fontsize=24)
    
    f.suptitle(axis_label, fontsize=28)
    filename = ('figures/%s_%s_compare/%s_%s_compare_%s.png' % 
                (roadset1['label'], roadset2['label'], roadset1['label'],
                 roadset2['label'], kin))
    plt.savefig(filename, bbox_inches='tight')
    
    plt.show()

def get_spill_df(server, schema_list):

    query = """
            SELECT spillID, s.runID, targetPos, value AS 'G2SEM', liveProton
            FROM Spill s INNER JOIN Beam b USING(spillID)
            WHERE name='S:G2SEM' AND
                dataQuality = 0 AND liveProton IS NOT NULL
            """
    if server == "seaquel.physics.illinois.edu":
        port = 3283
    else:
        port = 3306

    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                             read_default_group='guest',
                             host=server,
                             port=port)
        cur = db.cursor()

        spill_df = pd.DataFrame()
        for schema in schema_list:
            cur.execute("USE %s" % (schema))
            spill_df_tmp = pd.read_sql(query, db)
            if len(spill_df) == 0:
                spill_df = spill_df_tmp.copy()
            else:
                spill_df = pd.concat([spill_df, spill_df_tmp])

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])

    spill_df['G2SEM'] = spill_df['G2SEM'].astype(float)
    spill_df['liveProton'] = spill_df['liveProton'].astype(float)

    return spill_df
