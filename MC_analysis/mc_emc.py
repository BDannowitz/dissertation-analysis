#!/usr/bin/python
# coding=utf-8

import pymysql as mdb
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from uncertainties import unumpy as unp

from kdimuon import *

sys.path.append('../pylib')
from servers import server_dict
from productions import table_exists, schema_exists
from targets import target_dict, mc_target_df


def mc_make_kDimuon_table(server, schema, table, source_schema):
    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         db=source_schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        query = "CREATE TABLE %s.%s LIKE kDimuon"
        cur.execute(query % (schema, table))
        query = """ALTER TABLE %s.%s
                   ADD targpos SMALLINT DEFAULT NULL,
                   ADD sigWeight DOUBLE DEFAULT NULL,
                   ADD eventsThrown DOUBLE DEFAULT NULL,
                   DROP PRIMARY KEY
                """
        cur.execute(query % (schema, table))

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def mc_truth_make_mDimuon_table(server, schema, table, source_schema):
    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         db=source_schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        query = "CREATE TABLE %s.%s LIKE mDimuon"
        cur.execute(query % (schema, table))
        query = """ALTER TABLE %s.%s
                   ADD targpos SMALLINT DEFAULT NULL,
                   ADD eventsThrown DOUBLE DEFAULT NULL
                   # DROP PRIMARY KEY
                """
        cur.execute(query % (schema, table))

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def drop_analysis_table(server, schema, table):
    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         db=schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        query = "DROP TABLE IF EXISTS %s"
        cur.execute(query % table)

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def make_analysis_schema(server, schema):

    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        cur.execute('CREATE DATABASE IF NOT EXISTS %s' % schema)

        if db:
            db.close()

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def mc_get_dimuon_df(server, schema, table):

    query = """
            SELECT dimuonID, runID, dz, mass, xF, xB, xT,
                targpos, dpx, dpy, sigWeight/eventsThrown AS weight
            FROM %s.%s
            """
    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=server,
                         port=server_dict[server]['port'])

        dimuon_df = pd.read_sql(query % (schema, table), db)

        if db:
            db.close()

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return dimuon_df


def mc_truth_get_dimuon_df(server, schema, table):

    query = """
            SELECT eventID AS 'dimuonID', runID, dz, mass,
                   SQRT(POW(dpx,2)+POW(dpy,2)) AS `dpt`, xF, xB, xT,
                   phi_gam, phi_mu, theta_mu, targpos,
                   sigWeight/eventsThrown AS weight
            FROM %s.%s
            """
    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=server,
                         port=server_dict[server]['port'])

        dimuon_df = pd.read_sql(query % (schema, table), db)

        if db:
            db.close()

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return dimuon_df


def mc_get_analysis_data(server='seaquel.physics.illinois.edu',
                         analysis_schema='user_dannowitz_mc',
                         mc_schema_list=['mc_drellyan_LD2_M001_S001'],
                         analysis_table='kDimuon_mc',
                         fresh_start=False):

    if not schema_exists(server, analysis_schema):
        make_analysis_schema(server, analysis_schema)
    if not table_exists(server, analysis_schema, analysis_table):
        mc_make_kDimuon_table(server, analysis_schema,
                              analysis_table, mc_schema_list[0])

    if fresh_start:
        drop_analysis_table(server, analysis_schema, analysis_table)
        mc_make_kDimuon_table(server, analysis_schema,
                              analysis_table, mc_schema_list[0])

    if fresh_start:
        for mc_schema in mc_schema_list:
            mc_fill_kDimuon_table(server, analysis_schema,
                                  analysis_table, mc_schema)

    dimuon_df = mc_get_dimuon_df(server, analysis_schema, analysis_table)

    return dimuon_df


def mc_truth_get_analysis_data(server='seaquel.physics.illinois.edu',
                               analysis_schema='user_dannowitz_mc',
                               mc_schema_list=['mc_drellyan_LD2_M001_S001'],
                               analysis_table='kDimuon_mc_truth',
                               fresh_start=False, *wargs):

    if not schema_exists(server, analysis_schema):
        make_analysis_schema(server, analysis_schema)
    if not table_exists(server, analysis_schema, analysis_table):
        mc_truth_make_mDimuon_table(server, analysis_schema,
                                    analysis_table, mc_schema_list[0])

    if fresh_start:
        drop_analysis_table(server, analysis_schema, analysis_table)
        mc_truth_make_mDimuon_table(server, analysis_schema,
                                    analysis_table, mc_schema_list[0])

    if fresh_start:
        for merged_schema in mc_schema_list:
            mc_truth_fill_dimuon_table(server, analysis_schema,
                                       analysis_table, merged_schema)

    dimuon_df = mc_truth_get_dimuon_df(server, analysis_schema, analysis_table)

    return dimuon_df


def mc_emc_analysis(dimuon_df, n_bins, kin, kin_range=(0.0, 0.5)):
    # We will be grouping everything into these bins, so let's get it binned already

    bin_size = (kin_range[1] - kin_range[0])/n_bins
    bins = [kin_range[0]]
    val = kin_range[0]
    for i in range(n_bins):
        bins.append(val+bin_size)
        val = val+bin_size

    dimuon_df['weight_sq'] = dimuon_df['weight'] ** 2
    dimuon_df['dpt'] = np.sqrt(np.square(dimuon_df['dpx'])+np.square(dimuon_df['dpy']))

    groups = dimuon_df[['mass', 'xF', 'xB', 'xT', 'dpt']].groupby(by=[dimuon_df.target,
                                                               pd.cut(dimuon_df[kin], bins)])

    # Calculate the counts in each bin
    counts_df = pd.DataFrame(dimuon_df[['mass', 'xF', 'xB', 'xT', 'dpt', 'weight']].groupby(by=[dimuon_df.target,
                                                                             pd.cut(dimuon_df[kin], bins)]).weight.sum())

    unc_df = pd.DataFrame(dimuon_df[['mass', 'xF', 'xB', 'xT', 'dpt', 'weight_sq']].groupby(by=[dimuon_df.target,
                                                                             pd.cut(dimuon_df[kin], bins)]).weight_sq.sum())
    unc_df = unc_df.apply(np.sqrt, axis=0)

    # Calculate the mean kinematic values in each bin for each target
    means_df = groups['mass', 'xF', 'xB', 'xT', 'dpt'].mean()

    # Add the counts to the means dataframe
    means_df['counts'] = counts_df
    means_df['unc'] = unc_df
    means_df['normcounts'] = unp.uarray(means_df['counts'], means_df['unc'])

    # 7. Calculate LD2/LH2 ratio values
    #d2_h2_ratio = np.divide(means_df.ix[['LD2']]['normcounts'].values / mc_target_df.ix['LD2'].scale,
    #                        means_df.ix[['LH2']]['normcounts'].values / mc_target_df.ix['LH2'].scale)
    c_d2_ratio = np.divide(means_df.ix[['C']]['normcounts'].values / mc_target_df.ix['C'].Scale,
                           means_df.ix[['LD2']]['normcounts'].values / mc_target_df.ix['LD2'].Scale)
    fe_d2_ratio = np.divide(means_df.ix[['Fe']]['normcounts'].values / mc_target_df.ix['Fe'].Scale,
                            means_df.ix[['LD2']]['normcounts'].values / mc_target_df.ix['LD2'].Scale)
    #w_d2_ratio = np.divide(means_df.ix[['W']]['normcounts'].values / mc_target_df.ix['W'].scale,
    #                       means_df.ix[['LD2']]['normcounts'].values / mc_target_df.ix['LD2'].scale)

    bin_centers = means_df.ix[['LD2']][kin].values

    # Apply isoscalar correction using the D/H ratio values!
    #A = 183.0
    #Z = 73.0
    #N = A - Z

    #w_d2_ratio_iso = []
    #for i in range(len(w_d2_ratio)):
    #    if d2_h2_ratio[i] and w_d2_ratio[i]:
    #        correction_factor = ( A * d2_h2_ratio[i] ) / ( Z + N * ( 2 * d2_h2_ratio[i] - 1 ) )
    #        corrected_value = correction_factor * w_d2_ratio[i]
    #        w_d2_ratio_iso.append(corrected_value)
    #    else:
    #        w_d2_ratio_iso.append(w_d2_ratio[i])

    # Create super bonus isoscalar correction using the D/H ratio values!
    #A = 56.0
    #Z = 26.0
    #N = A - Z

    #fe_d2_ratio_iso = []
    #for i in range(len(w_d2_ratio)):
    #    if d2_h2_ratio[i] and fe_d2_ratio[i]:
    #        correction_factor = ( A * d2_h2_ratio[i] ) / ( Z + N * ( 2 * d2_h2_ratio[i] - 1 ) )
    #        corrected_value = correction_factor * fe_d2_ratio[i]
    #        fe_d2_ratio_iso.append(corrected_value)
    #    else:
    #        fe_d2_ratio_iso.append(fe_d2_ratio[i])

    emc_df = pd.DataFrame([bin_centers, c_d2_ratio, fe_d2_ratio],
                          columns=means_df.index.levels[1],
                          index=[kin, 'C/D', 'Fe/D']).T
    #emc_df = pd.DataFrame([bin_centers, d2_h2_ratio, c_d2_ratio, fe_d2_ratio, fe_d2_ratio_iso, w_d2_ratio, w_d2_ratio_iso],
                          #columns=means_df.index.levels[1],
                          #index=['xT', 'D/H', 'C/D', 'Fe/D', 'Fe/D(iso)', 'W/D', 'W/D(iso)']).T

    return emc_df, means_df

if __name__ == '__main__':
    main()
