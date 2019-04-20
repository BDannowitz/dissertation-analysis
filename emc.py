#!/usr/bin/python
# coding=utf-8

import pymysql as Mdb
import pandas as pd
import numpy as np
from numpy import nan
import math
from matplotlib import pylab
from matplotlib import pyplot as plt
import matplotlib
from uncertainties import unumpy as unp
import uncertainties as unc
from kdimuon import make_analysis_table, drop_analysis_table, fill_analysis_table
from servers import server_dict
from productions import table_exists, schema_exists, get_roadset
from targets import target_df, mc_target_df
from os import path

import sys
sys.path.append('../Thesis')

from strfunct import *

#pylab.rcParams['figure.figsize'] = 12, 8 
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
font = {'family' : 'serif', 'weight': 'bold'}
matplotlib.rc('font', **font)

CFG_PATH = path.abspath(path.dirname(__file__))
CONFIG_FILE = ('%s/emc.cfg' % CFG_PATH)

# Store the E772 results
e772_dict = {}
# DY Ratio for C/D vrs Xtgt - E772
# --------------------------------
e772_dict['C/D'] = {}
e772_dict['C/D']['xT'] = [0.041, 0.062, 0.087, 0.111, 0.136,
                          0.161, 0.186, 0.216, 0.269]
e772_dict['C/D']['ratio'] = unp.uarray(
    [0.981, 0.974, 1.013, 1.011, 0.979, 1.049, 1.117, 1.151, 1.044],
    [0.017, 0.014, 0.016, 0.020, 0.027, 0.044, 0.074, 0.110, 0.202])

# DY Ratio for Fe/D vrs Xtgt - E772
# --------------------------------
e772_dict['Fe/D'] = {}
e772_dict['Fe/D']['xT'] = [0.041, 0.062, 0.087, 0.111, 0.136,
                           0.161, 0.186, 0.219, 0.271]
e772_dict['Fe/D']['ratio'] = unp.uarray(
    [0.954, 0.976, 1.009, 0.992, 0.984, 1.009, 0.953, 1.016, 0.984],
    [0.014, 0.011, 0.012, 0.014, 0.018, 0.027, 0.037, 0.050, 0.088])

# DY Ratio for W/D vrs Xtgt - E772
# --------------------------------
e772_dict['W/D'] = {}
e772_dict['W/D']['xT'] = [0.041, 0.062, 0.087, 0.111, 0.136,
                          0.161, 0.186, 0.216, 0.269]
e772_dict['W/D']['ratio'] = unp.uarray(
    [0.954, 0.976, 1.009, 0.992, 0.984, 1.009, 0.953, 1.016, 0.984],
    [0.016, 0.013, 0.016, 0.020, 0.026, 0.043, 0.067, 0.099, 0.192])


def make_analysis_schema(server, schema):
    try:
        db = Mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        cur.execute('CREATE DATABASE IF NOT EXISTS %s' % schema)

        if db:
            db.close()

    except Mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def fetch_analysis_table(server, schema, table,
                         truth_mc=False, tracked_mc=False):
    if truth_mc:
        query = """
                SELECT mass, xF, xT, xB, dz,
                       dpx, dpy, dpz, phi_gam,
                       phi_mu, theta_mu,
                       sigWeight/eventsThrown AS `weight`, targetPos
                FROM %s.%s
                """
    elif tracked_mc:
        query = """
                SELECT dx, dy, dz, dpx, dpy, dpz, mass, xF, xB, xT,
                       costh, phi, trackSeparation, chisq_dimuon,
                       px1, py1, pz1, px2, py2, pz2, targetPos,
                       sigWeight/eventsThrown AS `weight`
                FROM %s.%s
                """
    else:
        query = """
                SELECT * FROM %s.%s
                """
    try:
        db = Mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=server,
                         port=server_dict[server]['port'])

        dimuon_df = pd.read_sql(query % (schema, table), db)

        if db:
            db.close()

    except Mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return dimuon_df


def add_target(df):

    pos_to_targ = {1: 'LH2', 2: 'Empty',
                   3: 'LD2', 4: 'None',
                   5: 'Fe', 6: 'C', 7: 'W'}
    if 'targetPos' not in df.columns:
        print ("Error: no 'targetPos' field in the dataframe. Aborting...")
        return df
    else:
        df['target'] = df.replace({'targetPos':pos_to_targ})['targetPos']

    return df


def add_roadset(df):

    if 'runID' not in df.columns:
        print ("'runID' column needed for assigning roadset.")
        return df
    df['roadset'] = df['runID'].apply(get_roadset)

    return df


def add_weight(df):

    fit_dict = {57:
                {1: unc.ufloat(3.43405e-06, 1.33679e-06),
                 2: unc.ufloat(-6.32977e-06, 3.25173e-06),
                 3: unc.ufloat(1.20659e-05, 1.52869e-06),
                 4: unc.ufloat(-2.48544e-05, 2.60058e-06),
                 5: unc.ufloat(9.53102e-06, 2.11621e-06),
                 6: unc.ufloat(7.40493e-06, 1.74959e-06),
                 7: unc.ufloat(9.00549e-06, 1.9774e-06)},
                62:
                {1: unc.ufloat(5.95561e-06, 1.18103e-06),
                 2: unc.ufloat(-1.24607e-06, 1.29412e-06),
                 3: unc.ufloat(1.28203e-05, 1.25617e-06),
                 4: unc.ufloat(-1.31643e-06, 2.15771e-06),
                 5: unc.ufloat(1.44961e-05, 1.85336e-06),
                 6: unc.ufloat(8.34177e-06, 2.02136e-06),
                 7: unc.ufloat(1.07145e-05, 1.81487e-06)},
                67:
                {1: unc.ufloat(5.1264e-06, 7.49e-07),
                 2: unc.ufloat(-4.89179e-06, 5.25423e-07),
                 3: unc.ufloat(1.08734e-05, 7.48282e-07),
                 4: unc.ufloat(-1.95272e-05, 6.14235e-07),
                 5: unc.ufloat(1.07482e-05, 1.13617e-06),
                 6: unc.ufloat(8.1867e-06, 1.20171e-06),
                 7: unc.ufloat(8.78054e-06, 1.00238e-06)}}
    fit_dict[59] = fit_dict[57]
    fit_dict[61] = fit_dict[57]
    fit_dict[70] = fit_dict[67]
    
    #required_cols = ('targetPos', 'Intensity_p')
    #if not all(col in df.columns for col in required_cols):
    #    print ("""Required columns are not present for the weight
    #              calculation. Aborting...""")
    #    return df

    #def calc_weight(roadset, target, intensity):
    #    return unp.exp(unc.nominal_value(fit_dict[roadset][target]) * intensity)
    #df['weight'] = df.apply(lambda x: calc_weight(x['roadset'], x['targetPos'], x['Intensity_p']), axis=1)
    if all(col in df.columns for col in ('negRoadEff', 'posRoadEff')):
        df['efficiency'] = df['posRoadEff'] * df['negRoadEff']
        df['weight'] = 1.0 / df['efficiency']
               
    return df


def get_dimuon_df(server='e906-db3.fnal.gov',
                  analysis_schema='user_dannowitz_analysis',
                  source_schema_list=(),
                  analysis_table='kDimuon_analysis',
                  fresh_start=False,
                  truth_mc=False,
                  tracked_mc=False,
                  likesign=None):
    
    if len(source_schema_list) == 0:
        print ("Please pass one or many source schemas.")

    if not schema_exists(server, analysis_schema):
        make_analysis_schema(server, analysis_schema)
    if not fresh_start and not table_exists(server, analysis_schema, analysis_table):
        make_analysis_table(server, analysis_schema,
                            analysis_table, source_schema_list[0],
                            truth_mc=truth_mc, tracked_mc=tracked_mc)

    if fresh_start:
        drop_analysis_table(server, analysis_schema, analysis_table)
        make_analysis_table(server, analysis_schema,
                            analysis_table,
                            source_schema_list[0],
                            truth_mc=truth_mc,
                            tracked_mc=tracked_mc,
                            likesign=likesign)
        for source_schema in source_schema_list:
            print ('>>> Retrieving and Cleaning Dimuon Data for %s' %
                   source_schema)
            fill_analysis_table(server, analysis_schema,
                                analysis_table, source_schema,
                                tracked_mc=tracked_mc,
                                truth_mc=truth_mc,
                                likesign=likesign)

    dimuon_df = fetch_analysis_table(server, analysis_schema, analysis_table,
                                     truth_mc=truth_mc,
                                     tracked_mc=tracked_mc)
    
    if all(field in dimuon_df.columns for field in ('dpx', 'dpy')):
        dimuon_df['dpt'] = np.sqrt(dimuon_df['dpx'] ** 2 + dimuon_df['dpy'] ** 2)
    if all(field in dimuon_df.columns for field in ('px1', 'px2', 'py1', 'py2')):
        dimuon_df['pt1'] = np.sqrt(dimuon_df['px1'] ** 2 + dimuon_df['py1'] ** 2)
        dimuon_df['pt2'] = np.sqrt(dimuon_df['px2'] ** 2 + dimuon_df['py2'] ** 2)
   
    dimuon_df = add_target(dimuon_df)
    
    if not truth_mc and not tracked_mc:
        dimuon_df = add_roadset(dimuon_df)
        dimuon_df = add_weight(dimuon_df)

    if 'weight' in dimuon_df.columns:
        dimuon_df['weight_sq'] = dimuon_df['weight'] ** 2

    return dimuon_df


def get_spill_df(server='e906-db3.fnal.gov',
                 analysis_schema='user_dannowitz_analysis_Apr1',
                 merged_schema_list=('merged_roadset57_R004_V005',
                                     'merged_roadset59_R004_V005',
                                     'merged_roadset62_R004_V005')):

    spill_df = pd.DataFrame()
    for schema in merged_schema_list:
        query = ("""
                 SELECT runID, spillID, targetPos, liveProton
                 FROM %s.Spill
                 WHERE
                     Spill.runID IN( SELECT DISTINCT runID FROM kTrack ) AND
                     Spill.dataQuality = 0
                 """ % schema)
        try:
            db = Mdb.connect(read_default_file='../.my.cnf',
                             read_default_group='guest',
                             db=schema,
                             host=server,
                             port=server_dict[server]['port'])

            tmp_spill_df = pd.read_sql(query, db)
            spill_df = pd.concat([spill_df, tmp_spill_df])

            if db:
                db.close()

        except Mdb.Error, e:
            print ("Error %d: %s" % (e.args[0], e.args[1]))
            return 1
    
    spill_df['liveProton_x10^16'] = np.divide(spill_df['liveProton'],
                                              math.pow(10, 16))
    spill_df = add_target(spill_df)

    return spill_df


def get_livep_from_spill(spill_df):

    required = ('target', 'liveProton', 'liveProton_x10^16')
    if all(item in spill_df.columns for item in required):
        summed_cols = ['liveProton', 'liveProton_x10^16']
        live_p_df = spill_df.groupby(by='target')[summed_cols].sum()
    else:
        print ("Not all required fields present. Aborting...")
        return None

    return live_p_df 


def combine_emc(emc_df_list):

    if len(emc_df_list) < 2:
        print ("Please use a list of two or more emc results")
        return None

    nominal = []
    inv_sq_sig = []
    for emc_df in emc_df_list:
        #tmp_nominal = emc_df[emc_df.columns[1:]].applymap(unp.nominal_values)
        tmp_nominal = emc_df.applymap(unp.nominal_values)
        nominal.append(tmp_nominal)

        #tmp_sig = emc_df[emc_df.columns[1:]].applymap(unp.std_devs)
        tmp_sig = emc_df.applymap(unp.std_devs)
        tmp_inv_sq_sig = 1.0/np.square(tmp_sig)
        inv_sq_sig.append(tmp_inv_sq_sig)

    numerator = 0.0
    denominator = 0.0
    for i in range(0, len(emc_df_list)):
        numerator += inv_sq_sig[i] * nominal[i]
        denominator += inv_sq_sig[i]

    combined_emc_nom = np.divide(numerator, denominator)
    combined_emc_sig = np.divide(1.0,np.sqrt(denominator))

    combined_emc = pd.DataFrame(unp.uarray(combined_emc_nom,
                                           std_devs=combined_emc_sig),
                                columns=combined_emc_nom.columns,
                                index=combined_emc_nom.index)

    if 'xT' not in combined_emc.columns:
        combined_emc.insert(0, 'xT', emc_df_list[0]['xT'])
    
    return combined_emc


def iso_correct(df, col, A, Z):

    if 'D/H' not in df.columns or col not in df.columns:
        print ("Require a 'D/H' and %s column in order to apply "
               "isoscaler correction." % col)
        return None
    
    N = A - Z

    correction_factor = np.divide((A * df['D/H']),(Z + N * (2.0 * df['D/H'] - 1.0)))
    iso_corrected_series = correction_factor * df[col]

    return iso_corrected_series


def subtract_empty_none_bg(df):

    df['ncounts_bg'] = None

    for target in ('LD2', 'LH2'):
        if target in df.index and 'Empty' in df.index:
            subtract_res = df.ix[[target]].ncounts.values - df.ix[['Empty']].ncounts.values
            df.set_value(target, 'ncounts_bg', subtract_res)
    for target in ('Fe', 'W', 'C'):
        if target in df.index and 'None' in df.index:
            subtract_res = df.ix[[target]].ncounts.values - df.ix[['None']].ncounts.values
            df.set_value(target, 'ncounts_bg', subtract_res)
    for target in ('None', 'Empty'):
        if target in df.index:
            df.set_value(target, 'ncounts_bg', None)

    return df

    
def fix_nan(value):
    """Helper function to set unwanted NaN's to 0."""
    if np.isnan(unc.std_dev(value)):
        if np.isnan(unc.nominal_value(value)):
            return unc.ufloat(0.0, 0.0)
        else:
            return unc.ufloat(unc.nominal_value(value),0)
    else:
        return value


def calculate_emc(dimuon_df, spill_df, bin_edges=[], qie_correction=False,
                  truth_mc=False, tracked_mc=False):
    # We will be grouping everything into these bins, so let's get it binned

    if not truth_mc and not tracked_mc:
        # Keep track of target attributes
        live_p_df = get_livep_from_spill(spill_df)
        if live_p_df is None:
            print ("Error getting live proton df from spill_df. Exiting...")
            return None

    # Binning very important. If no bins specified, use default.
    if len(bin_edges) == 0:
        bin_edges = [0.08, 0.14, 0.16, 0.18, 0.21, 0.25, 0.31, 0.53]

    # Create names of indexes that specify bin ranges.
    xranges = []
    for i in range(0,len(bin_edges)-1):
        xranges.append(("(%.02f, %.02f]" % (bin_edges[i], bin_edges[i+1])))

    if truth_mc:
        group_cols = ['mass', 'xF', 'xT', 'xB', 'dz', 'dpx', 'dpy', 'dpz',
                      'phi_gam', 'phi_mu', 'theta_mu', 'weight', 'weight_sq']
    elif tracked_mc:
        group_cols = ['mass', 'dz', 'dpz', 'dpt', 'pz1', 'pz2', 'pt1', 'pt2',
                      'xF', 'xB', 'xT', 'costh', 'phi', 'trackSeparation',
                      'chisq_dimuon', 'weight', 'weight_sq']
    else:
        group_cols = ['mass', 'dz', 'dpz', 'dpt', 'pz1', 'pz2', 'pt1', 'pt2',
                      'xF', 'xB', 'xT', 'costh', 'phi', 'trackSeparation',
                      'chisq_dimuon', 'QIESum', 'weight', 'weight_sq']

    if not all(col in dimuon_df.columns for col in group_cols):
        print ("Not all of these columns are in the dimuon_df:")
        print group_cols
        print ("Please add them or alter this analysis code. Exiting...")
        return None


    groups = dimuon_df[group_cols].groupby(by=[dimuon_df.target,
                                               pd.cut(dimuon_df.xT,
                                                      bin_edges)])

    # Calculate the counts in each bin
    if qie_correction:
        dimuon_df_copy = dimuon_df[unp.isnan(dimuon_df['weight']) == False].copy()
        counts = dimuon_df_copy[group_cols].groupby(
            by=[dimuon_df_copy.target, pd.cut(dimuon_df_copy.xT, bin_edges)]).weight.sum()
        unc_df = pd.DataFrame(dimuon_df_copy[group_cols].groupby(
            by=[dimuon_df_copy.target, pd.cut(dimuon_df_copy.xT, bin_edges)]).weight_sq.sum())
        unc_df = unc_df.apply(unp.sqrt, axis=0)
        del dimuon_df_copy
    else:
        if truth_mc or tracked_mc:
            counts = dimuon_df[group_cols].groupby(
                by=[dimuon_df.target, pd.cut(dimuon_df.xT, bin_edges)]).weight.sum()
            unc_df = pd.DataFrame(dimuon_df[group_cols].groupby(
                by=[dimuon_df.target, pd.cut(dimuon_df.xT, bin_edges)]).weight_sq.sum())
            unc_df = unc_df.apply(unp.sqrt, axis=0)
        else:
            counts = dimuon_df[group_cols].groupby(
                by=[dimuon_df.target, pd.cut(dimuon_df.xT, bin_edges)]).dpz.count()
    counts_df = pd.DataFrame(counts)    
    
    # Calculate the mean kinematic values in each bin for each target
    means_df = groups[group_cols].mean()
    std_dev_df = groups[group_cols].std()
    means_df = pd.DataFrame(unp.uarray(means_df, std_dev_df),
                            columns=means_df.columns,
                            index=means_df.index)


    # Add the counts to the means dataframe
    means_df['counts'] = counts_df

    if qie_correction or truth_mc or tracked_mc:
        means_df['unc'] = unc_df
        means_df['counts'] = unp.uarray(means_df['counts'],
                                        means_df['unc'])
        means_df.drop('unc', axis=1, inplace=True)
    else:
        means_df['counts'] = unp.uarray(means_df['counts'],
                                        np.sqrt(means_df['counts']))

    # When working with low-stat data, there may be some NaN's. Handle them.
    means_df = means_df.applymap(fix_nan)

    # Normalize to live proton count
    if truth_mc or tracked_mc:
        means_df['ncounts'] = means_df['counts']
    else:
        means_df['ncounts'] = means_df['counts'] / live_p_df['liveProton_x10^16']

    if not truth_mc and not tracked_mc:
        # Subtract Empty counts from LD2 and LH2 counts
        # Subtract None counts from C, Fe, and W counts
        means_df = subtract_empty_none_bg(means_df)
    else:
        means_df['ncounts_bg'] = means_df['ncounts']
   
    means_df['ncounts_bg_ctm'] = None
    if not truth_mc and not tracked_mc:
        # 5. Use LH2 and LD2 data, proportions, to get Deuterium counts
        a = 1.00
        b = 0.00
        c = 0.0714
        d = 0.9152
        scale = 1.0 / (c + d)
        c = scale * c
        d = scale * d
        c = 0.0724
        d = 0.9276

        contam_adjusted_vals = np.multiply(
            np.add(
                np.multiply(means_df.ix[['LD2']].ncounts_bg.values, a),
                np.negative(np.multiply(means_df.ix[['LH2']].ncounts_bg.values, c))
            ), (1 / (d * a - b * c))
        )

        means_df.set_value('LD2', 'ncounts_bg_ctm', contam_adjusted_vals)
    else:
        means_df.set_value('LD2', 'ncounts_bg_ctm', means_df.ix[['LD2']]['ncounts_bg'].values)

    if truth_mc or tracked_mc:
        tmp_target_df = mc_target_df.copy()
    else:
        tmp_target_df = target_df.copy()


    # 7. Calculate LD2/LH2 ratio values
    ratio_list = []
    ratio_label_list = []
    if all(target in means_df.index for target in ('LH2', 'LD2')):
        ratio = np.divide(
            means_df.ix[['LD2']]['ncounts_bg_ctm'].values / tmp_target_df.ix['LD2'].Scale,
            means_df.ix[['LH2']]['ncounts_bg'].values / tmp_target_df.ix['LH2'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('D/H')
    if all(target in means_df.index for target in ('C', 'LD2')):
        ratio = np.divide(means_df.ix[['C']]['ncounts_bg'].values / tmp_target_df.ix['C'].Scale,
                          means_df.ix[['LD2']]['ncounts_bg_ctm'].values / tmp_target_df.ix['LD2'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('C/D')
    if all(target in means_df.index for target in ('Fe', 'LD2')):
        ratio = np.divide(means_df.ix[['Fe']]['ncounts_bg'].values / tmp_target_df.ix['Fe'].Scale,
                          means_df.ix[['LD2']]['ncounts_bg_ctm'].values / tmp_target_df.ix['LD2'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('Fe/D')
    if all(target in means_df.index for target in ('W', 'LD2')):
        ratio = np.divide(means_df.ix[['W']]['ncounts_bg'].values / tmp_target_df.ix['W'].Scale,
                          means_df.ix[['LD2']]['ncounts_bg_ctm'].values / tmp_target_df.ix['LD2'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('W/D')
    if all(target in means_df.index for target in ('C', 'LD2')):
        ratio = np.divide(means_df.ix[['C']]['ncounts_bg'].values / tmp_target_df.ix['C'].Scale,
                          means_df.ix[['LH2']]['ncounts_bg'].values / tmp_target_df.ix['LH2'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('C/H')
    if all(target in means_df.index for target in ('Fe', 'LH2')):
        ratio = np.divide(means_df.ix[['Fe']]['ncounts_bg'].values / tmp_target_df.ix['Fe'].Scale,
                          means_df.ix[['LH2']]['ncounts_bg'].values / tmp_target_df.ix['LH2'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('Fe/H')
    if all(target in means_df.index for target in ('W', 'LH2')):
        ratio = np.divide(means_df.ix[['W']]['ncounts_bg'].values / tmp_target_df.ix['W'].Scale,
                          means_df.ix[['LH2']]['ncounts_bg'].values / tmp_target_df.ix['LH2'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('W/H')
    if all(target in means_df.index for target in ('Fe', 'C')):
        ratio = np.divide(means_df.ix[['Fe']]['ncounts_bg'].values / tmp_target_df.ix['Fe'].Scale,
                          means_df.ix[['C']]['ncounts_bg'].values / tmp_target_df.ix['C'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('Fe/C')
    if all(target in means_df.index for target in ('W', 'C')):
        ratio = np.divide(means_df.ix[['W']]['ncounts_bg'].values / tmp_target_df.ix['W'].Scale,
                          means_df.ix[['C']]['ncounts_bg'].values / tmp_target_df.ix['C'].Scale)
        ratio_list.append(ratio)
        ratio_label_list.append('W/C')
    
    bin_centers = means_df['xT']['LD2'].values

    emc_df = pd.DataFrame([bin_centers] + ratio_list,
                          columns=xranges,
                          index=['xT'] + ratio_label_list).T

    return emc_df, means_df


def emc_analysis(dimuon_df, spill_df, run_ranges, bin_edges=[],
                 qie_correction=False, truth_mc=False, tracked_mc=False):

    emc_list = []
    means_list = []
    avg_means_df = pd.DataFrame() 

    if not truth_mc and not tracked_mc:
        for run_range in run_ranges:
            query = ('(%i <= runID <= %i)' % (run_range[0], run_range[1]))
            dimuon_subset = dimuon_df.query(query)
            spill_subset = spill_df.query(query)
            emc_df, means_df = calculate_emc(dimuon_subset,
                                             spill_subset,
                                             bin_edges,
                                             qie_correction=qie_correction)
            emc_list.append(emc_df)
            means_list.append(means_df)

        if len(emc_list) > 1:
            emc_df = combine_emc(emc_list)
            avg_means_df = combine_emc(
                [means.drop(['counts','ncounts', 'ncounts_bg',
                             'ncounts_bg_ctm'], axis=1) for means in means_list])
        else:
            avg_means_df = means_df 

    else:
        emc_df, means_df = calculate_emc(dimuon_df,
                                         None,
                                         bin_edges,
                                         qie_correction=qie_correction,
                                         truth_mc=truth_mc,
                                         tracked_mc=tracked_mc)

    if 'W/D' in emc_df.columns:
        emc_df['W/D(iso)'] = iso_correct(emc_df, 'W/D', 183.0, 73.0)
    if 'W/C' in emc_df.columns:
        emc_df['W/C(iso)'] = iso_correct(emc_df, 'W/C', 183.0, 73.0)
    if 'Fe/D' in emc_df.columns:
        emc_df['Fe/D(iso)'] = iso_correct(emc_df, 'Fe/D', 56.0, 26.0)
    if 'Fe/C' in emc_df.columns:
        emc_df['Fe/C(iso)'] = iso_correct(emc_df, 'Fe/C', 56.0, 26.0)

    return emc_df, means_list, avg_means_df


def tick_loc_label(lims):

    min_ticks = 5
    xsig_digits = 1
    ysig_digits = 1
    xdiff = lims[1]-lims[0]
    ydiff = lims[3]-lims[2]
    nxticks = int(round(xdiff*10))
    nyticks = int(round(ydiff*10))
    if nxticks < min_ticks:
        nxticks = int(round(xdiff/5.0*100))
        xsig_digits = 2
    if nyticks < min_ticks:
        nyticks = int(round(ydiff/5.0*100))
        ysig_digits = 2
    xstep = xdiff/nxticks
    ystep = ydiff/nyticks
    xstep = round(xstep, xsig_digits)
    ystep = round(ystep, ysig_digits)
    xtick_loc = [round(lims[0],xsig_digits) + (i * xstep) for i in range(0,nxticks+1)]
    ytick_loc = [round(lims[2],ysig_digits) + (i * ystep) for i in range(0,nyticks+1)]
    
    #if xsig_digits==1:
    #    xtick_label = [r'$\mathbf{%.1f}$' % x for x in xtick_loc] 
    #else:
    #    xtick_label = [r'$\mathbf{%.2f}$' % x for x in xtick_loc] 
    #if ysig_digits==1:
    #    ytick_label = [r'$\mathbf{%.1f}$' % x for x in ytick_loc] 
    #else:
    #    ytick_label = [r'$\mathbf{%.2f}$' % x for x in ytick_loc] 
    if xsig_digits==1:
        xtick_label = ['%.1f' % x for x in xtick_loc] 
    else:
        xtick_label = ['%.2f' % x for x in xtick_loc] 
    if ysig_digits==1:
        ytick_label = ['%.1f' % x for x in ytick_loc] 
    else:
        ytick_label = ['%.2f' % x for x in ytick_loc] 

    return xtick_loc, xtick_label, ytick_loc, ytick_label


def plot_emc(bin_centers, uvalues, label, fmt, color,
             ax=None, xerr=[], yerr=[],
             markerfacecolor=None):
    """The most common plotting method and formatting of 
       my standard EMC plots."""

    # If not part of a larger figure, make a figure
    new_plot = False
    if not ax:
        f, ax = plt.subplots(1, 1, figsize=(7,7))
        new_plot = True
        
    if not markerfacecolor:
        markerfacecolor=color
        
    if len(yerr)==0:
        yerr = unp.std_devs(uvalues)
        
    if len(xerr) == 0:
        (_, caps, barlines) = ax.errorbar(bin_centers,
                                   unp.nominal_values(uvalues),
                                   yerr=yerr,
                                   fmt=fmt,
                                   ms=11,
                                   elinewidth=1,
                                   markerfacecolor=markerfacecolor,
                                   color=color,
                                   label=label)

    else:
        (_, caps, barlines) = ax.errorbar(bin_centers,
                                   unp.nominal_values(uvalues),
                                   yerr=yerr,
                                   xerr=xerr,
                                   fmt=fmt,
                                   ms=14,
                                   elinewidth=3.5,
                                   markerfacecolor=markerfacecolor,
                                   color=color,
                                   label=label)

   
    # Make xerr bars more like shaded regions with no caps
    # These indicate bin range, not actual uncertainty
    if len(barlines) > 1:
        barlines[0].set_alpha(0.15)
        barlines[0].set_linewidth(4)
        caps[0].set_markeredgewidth(0)
        caps[1].set_markeredgewidth(0)

    #if new_plot:
    #    plt.show()

    return ax


def plot_emc_format(ax, title, xlabel=r'$x_2$', log=False):
    """Some formatting specific to my standard EMC plots."""

    ax.set_title(title, fontsize=28)
    if log:
        ax.axhline(y=0, c='black', ls='--', linewidth=1.5)
    else:
        ax.axhline(y=1, c='black', ls='--', linewidth=1.5)
    plt.setp(ax.get_xticklabels(), fontsize=23)
    plt.setp(ax.get_yticklabels(), fontsize=23)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    legend = ax.legend(handles, labels,
                       loc='best',
                       markerscale=0.75,
                       numpoints=1,
                       prop={'size':15})

    return None


def plot_three_emc(bin_centers, kin, bins,
                   c_value_list, c_label_list,
                   fe_value_list, fe_label_list,
                   w_value_list, w_label_list,
                   lims=None,
                   savefile=False,
                   ylabel=r'$\boldsymbol{R \left(\frac{A}{D}\right)}$',
                   xlabel=r'$x_2$',
                   log=False):

    lower_val = np.subtract(bin_centers,bins[:-1])
    upper_val = np.subtract(bins[1:], bin_centers)
    asymmetric_bars = [lower_val, upper_val]
     
    colors = ['red', 'blue', 'green', 'magenta']
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(19,7))

    ax1.set_ylabel(ylabel, fontsize=30, rotation=0, labelpad=45)
    if kin=='xT' and not log:
        plot_emc(e772_dict['C/D']['xT'],
                 e772_dict['C/D']['ratio'],
                 label='E-772 C/D',
                 fmt='s',
                 markerfacecolor='white',
                 color='black',
                 ax=ax1)
    for value_label in zip(c_value_list, c_label_list, colors):
        plot_emc(bin_centers,
                 value_label[0].values,
                 xerr=asymmetric_bars,
                 fmt='o',
                 color=value_label[2],
                 label=value_label[1],
                 ax=ax1)
    plot_emc_format(ax1, 'C', xlabel, log=log)
    #ax1.text(0.24, 0.05, r'$\chi^2/d.o.f. (57,62) = 1.97$',
    #         ha='center', va='center', transform=ax1.transAxes)
    #ax1.text(0.24, 0.12, r'$\chi^2/d.o.f. (62,67) = 2.74$',
    #         ha='center', va='center', transform=ax1.transAxes)
    #ax1.text(0.24, 0.19, r'$\chi^2/d.o.f. (57,67) = 1.13$',
    #         ha='center', va='center', transform=ax1.transAxes)
        
    if kin=='xT' and not log:
        plot_emc(e772_dict['Fe/D']['xT'],
                 e772_dict['Fe/D']['ratio'],
                 label='E-772 Fe/D',
                 fmt='s',
                 markerfacecolor='white',
                 color='black',
                 ax=ax2)
    for value_label in zip(fe_value_list, fe_label_list, colors):
        plot_emc(bin_centers,
                 value_label[0].values,
                 xerr=asymmetric_bars,
                 fmt='o',
                 color=value_label[2],
                 label=value_label[1],
                 ax=ax2)
    plot_emc_format(ax2, 'Fe', xlabel, log=log)
    #ax2.text(0.24, 0.05, r'$\chi^2/d.o.f. (57,62) = 1.12$',
    #         ha='center', va='center', transform=ax2.transAxes)
    #ax2.text(0.24, 0.12, r'$\chi^2/d.o.f. (62,67) = 1.32$',
    #         ha='center', va='center', transform=ax2.transAxes)
    #ax2.text(0.24, 0.19, r'$\chi^2/d.o.f. (57,67) = 0.82$',
    #         ha='center', va='center', transform=ax2.transAxes)

    if kin=='xT' and not log:
        plot_emc(e772_dict['W/D']['xT'],
                e772_dict['W/D']['ratio'],
                label='E-772 W/D',
                fmt='s',
                color='black',
                markerfacecolor='white',
                ax=ax3)
    for value_label in zip(w_value_list, w_label_list, colors):
        plot_emc(bin_centers,
                 value_label[0].values,
                 xerr=asymmetric_bars,
                 fmt='o',
                 color=value_label[2],
                 label=value_label[1],
                 ax=ax3)
    plot_emc_format(ax3, 'W', xlabel, log=log)
    #ax3.text(0.24, 0.05, r'$\chi^2/d.o.f. (57,62) = 0.94$',
    #         ha='center', va='center', transform=ax3.transAxes)
    #ax3.text(0.24, 0.12, r'$\chi^2/d.o.f. (62,67) = 0.33$',
    #         ha='center', va='center', transform=ax3.transAxes)
    #ax3.text(0.24, 0.19, r'$\chi^2/d.o.f. (57,67) = 0.57$',
    #         ha='center', va='center', transform=ax3.transAxes)
    
    if not lims:
        if kin=='xT':
            lims = (0, 0.54, 0.75, 1.27)
        #else:
            lims = plt.xlim() + plt.ylim()
            
    ax1.grid(); ax2.grid(); ax3.grid()

    #xtick_loc, xtick_label, ytick_loc, ytick_label = tick_loc_label(lims)
    #plt.xticks(xtick_loc, xtick_label)
    #plt.yticks(ytick_loc, ytick_label)
    
    # Set x and y limits
    ax1.axis(lims); ax2.axis(lims); ax3.axis(lims);

    plt.tight_layout()
    plt.setp([a.get_xticklabels() for a in f.axes], weight='bold')
    plt.setp([a.get_yticklabels() for a in f.axes], weight='bold')
    plt.setp([a.get_yticklabels() for a in f.axes[1:]], visible=False)

    if savefile:
        plt.savefig(savefile, bbox_inches='tight', transparent=False)

    return None 


def plot_one_emc(bin_centers, asymmetric_bars,
                 value_list, label_list,
                 lims=(0, 0.54, 0.75, 1.27),
                 savefile=False,
                 title=None,
                 ylabel=r'$\boldsymbol{R \left(\frac{A}{D}\right)}$'):

    colors = ['red', 'blue', 'green', 'magenta']
    f, ax = plt.subplots(1, 1, figsize=(7,7))

    ax.set_ylabel(ylabel, fontsize=30, rotation=0, labelpad=55)
    for value_label in zip(value_list, label_list, colors):
        plot_emc(bin_centers,
                 value_label[0].values,
                 xerr=asymmetric_bars,
                 fmt='o',
                 color=value_label[2],
                 label=value_label[1],
                 ax=ax)
    plot_emc_format(ax, title)

    xtick_loc, xtick_label, ytick_loc, ytick_label = tick_loc_label(lims)
    plt.xticks(xtick_loc, xtick_label)
    plt.yticks(ytick_loc, ytick_label)

    # Set x and y limits
    ax.axis(lims)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, bbox_inches='tight', transparent=False)

    return None 


def main():
    print "Hello World!"


if __name__ == '__main__':
    main()
