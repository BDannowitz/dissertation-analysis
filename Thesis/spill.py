import pymysql as Mdb
import pandas as pd
import numpy as np
import math

class spill(object):
    
    def __init__(self, runrange, source, server, schema, table, fresh_start):
        self.data       = pd.DataFrame()
        self.live_p     = pd.DataFrame()
        self.runrange   = runrange
        self.source     = source
        self.server     = server
        self.port       = 3306
        if self.server == 'seaquel.physics.illinois.edu':
            self.port = 3283
        self.schema     = schema
        self.table      = table
        
        if fresh_start:
            self.fill_spill_table()
        
        self.data = self.fetch_spill_data()
        
        self.data['liveProton_x10^16'] = np.divide(self.data['liveProton'],
                                                   math.pow(10, 16))
        self.add_target()
        
        print (">> Getting Live Proton data.")
        self.live_p = self.calc_live_p()
        
        return None
    
    def calc_live_p(self):

        required = ('target', 'liveProton', 'liveProton_x10^16')
        if all(item in self.data.columns for item in required):
            summed_cols = ['liveProton', 'liveProton_x10^16']
            live_p_df = self.data.groupby(by='target')[summed_cols].sum()
        else:
            print ("Not all required fields present. Aborting...")
            return None

        return live_p_df 
    
    def fill_spill_table(self):
        query = ("""
                 INSERT INTO %s.%s
                 SELECT *
                 FROM Spill
                 WHERE
                     Spill.runID IN( SELECT DISTINCT runID FROM kTrack ) AND
                     Spill.dataQuality = 0
                 """ % (self.schema, self.table))
        try:
            db = Mdb.connect(read_default_file='../.my.cnf',
                             read_default_group='guest',
                             host=self.server,
                             db=self.source,
                             port=self.port)
            cur = db.cursor()
            cur.execute(query)
            
            if db:
                db.close()

        except Mdb.Error, e:

            print "Error %d: %s" % (e.args[0], e.args[1])
            return 1

        return None
    
    def fetch_spill_data(self):
        
        query = ("""
                 SELECT * FROM %s.%s
                 WHERE runID BETWEEN %i AND %i
                 """ %
                 (self.schema, self.table,
                  self.runrange[0], self.runrange[1]))
        try:
            db = Mdb.connect(read_default_file='../.my.cnf',
                             read_default_group='guest',
                             host=self.server,
                             port=self.port)

            spill_df = pd.read_sql(query, db)

            if db:
                db.close()

        except Mdb.Error, e:

            print "Error %d: %s" % (e.args[0], e.args[1])
            return 1
        
        return spill_df
    
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
                 
