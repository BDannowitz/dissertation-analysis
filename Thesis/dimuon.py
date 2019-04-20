import pymysql as Mdb
import pandas as pd
import numpy as np
from sys import path
from copy import deepcopy
import uncertainties as unc

path.append('..')
path.append('../pylib')
from kdimuon import fill_analysis_table
from productions import get_roadset

class dimuon(object):
    
    def __init__(self, runrange, source, server,
                 schema, table, kin, fresh_start):
        
        self.data       = pd.DataFrame()
        self.runrange   = runrange
        self.source     = source
        self.server     = server
        self.port       = 3306
        if self.server == 'seaquel.physics.illinois.edu':
            self.port = 3283
        self.schema     = schema
        self.table      = table
        self.kin        = kin
        self.keff       = None
        
        if fresh_start:
            print (">> Applying all analysis cuts (may take a long time).")
            fill_analysis_table(self.server, self.schema,
                                self.table, self.source)
        
        print (">> Fetching dimuon data from the database")
        self.data = self.fetch_dimuon_data()
        
        if all(field in self.data.columns for field in ('dpx', 'dpy')):
            self.data['dpt'] = np.sqrt(self.data['dpx'] ** 2 + self.data['dpy'] ** 2)
        if all(field in self.data.columns for field in ('px1', 'px2', 'py1', 'py2')):
            self.data['pt1'] = np.sqrt(self.data['px1'] ** 2 + self.data['py1'] ** 2)
            self.data['pt2'] = np.sqrt(self.data['px2'] ** 2 + self.data['py2'] ** 2)

        self.add_target()
        self.add_roadset()
        
        print ">> #Dimuons:", len(self.data)
        
        return None
    
    def fetch_dimuon_data(self):
        
        query = ("""
                 SELECT * FROM %s.%s
                 WHERE runID BETWEEN %i AND %i
                 AND chamber_intensity < 60000
                 """ %
                 (self.schema, self.table,
                  self.runrange[0], self.runrange[1]))
        
        for kin in self.kin:
            if not isinstance(self.kin[kin], int):
                query += (" AND %s > %f AND %s < %f " % 
                          (kin, self.kin[kin][0], kin, self.kin[kin][-1]))
        try:
            db = Mdb.connect(read_default_file='../.my.cnf',
                             read_default_group='guest',
                             host=self.server,
                             port=self.port)

            dimuon_df = pd.read_sql(query, db)

            if db:
                db.close()

        except Mdb.Error, e:

            print "Error %d: %s" % (e.args[0], e.args[1])
            return 1

        return dimuon_df
    
    def add_target(self):

        pos_to_targ = {1: 'LH2', 2: 'Empty',
                       3: 'LD2', 4: 'None',
                       5: 'Fe', 6: 'C', 7: 'W'}
        if 'targetPos' not in self.data.columns:
            print ("Error: no 'targetPos' field in the dataframe. Aborting...")
            return None
        else:
            self.data['target'] = self.data.replace({'targetPos':pos_to_targ})['targetPos']

        return None


    def add_roadset(self):

        if 'runID' not in self.data.columns:
            print ("'runID' column needed for assigning roadset.")
            return None
        self.data['roadset'] = self.data['runID'].apply(get_roadset)

        return None
    
    
    def apply_keff(self, kin, keff):
        data_slice_list = []
        data_slice = pd.DataFrame()
        
        self.kin = deepcopy(kin)
        self.keff = deepcopy(keff)
        
        # Only analyze dimuons within these bounds
        for kin in self.kin:
            if not isinstance(self.kin[kin], int):
                query = ("%s > %f and %s < %f " % 
                          (kin, self.kin[kin][0], kin, self.kin[kin][-1]))
        
        self.data = self.data.query(query).copy()
        
        for i in range(1,8):
            data_slice_list.append(self.data.query('targetPos==@i').copy())
            data_slice_list[i-1]['weight_keff'] = self.keff[i].kEff(
                data_slice_list[i-1][self.kin.keys()+['chamber_intensity']], inv=True)
        self.data = pd.concat(data_slice_list)
        
        self.data['weight_keff_sq'] = self.data['weight_keff'] ** 2
        
        return None
    
    def apply_bg_correction(self, B, p1_e, p1_n):
        data_slice_list = []
        data_slice = pd.DataFrame()
        index = 0
        
        def bg_func(x, p1):
            return 1 + p1 * np.square(x)
        
        for i in [1,3]:
            
            data_slice_list.append(self.data.query('targetPos==@i').copy())
            data_slice_list[index]['weight_bg'] = (1.0-B[i]*bg_func(data_slice_list[index]['chamber_intensity'],p1_e))
            data_slice_list[index]['weight_bg_sq'] = data_slice_list[index]['weight_bg'] ** 2
            data_slice_list[index]['weight'] = np.multiply(data_slice_list[index]['weight_bg'],
                                                           data_slice_list[index]['weight_keff'])
            data_slice_list[index]['weight_sq'] = data_slice_list[index]['weight'] ** 2
            index += 1
            
        for i in [5,6,7]:
            
            data_slice_list.append(self.data.query('targetPos==@i').copy())
            data_slice_list[index]['weight_bg'] = (1.0-B[i]*bg_func(data_slice_list[index]['chamber_intensity'],p1_n))
            data_slice_list[index]['weight_bg_sq'] = data_slice_list[index]['weight_bg'] ** 2
            data_slice_list[index]['weight'] = np.multiply(data_slice_list[index]['weight_bg'],
                                                           data_slice_list[index]['weight_keff'])
            data_slice_list[index]['weight_sq'] = data_slice_list[index]['weight'] ** 2
            index += 1
        
        for i in [2,4]:
            data_slice_list.append(self.data.query('targetPos==@i').copy())
            data_slice_list[index]['weight_bg'] = None
            data_slice_list[index]['weight_bg_sq'] = None
            data_slice_list[index]['weight'] = None
            data_slice_list[index]['weight_sq'] = None
            index += 1
            
        self.data = pd.concat(data_slice_list)
        
        return None
    
    
    def combine_empty_none(self):
        self.data.targetPos.replace({4:2}, inplace=True)
        
        return None

