#!/usr/bin/python

from os import path, remove         # Check if files exist
import MySQLdb as mdb               # Raw data source is MySQL
import pandas as pd                 # Workhorse data management tool
import numpy as np                  # For matrices, arrays, matrixes, NaN's
import matplotlib.pyplot as plt     # For plotting some distributions
import seaborn as sns               # For easy, pretty plotting
from math import floor
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation

sns.set_style("darkgrid")
sns.set_context("talk", font_scale=1.4)

SERVER = 'e906-db3.fnal.gov'                        # Source MySQL server
PORT = 3306
SOURCE_SCHEMA = 'merged_roadset62_R004_V005'        # Source MySQL schema
ANALYSIS_SCHEMA = 'user_dannowitz_target_analysis'  # Temp storage schema name
ANALYSIS_TABLE = 'target_analysis'                  # Temp storage table name
TEST_FILE = 'roadset62_targetPos_test.csv'          # Local file to store data
TRAIN_FILE = 'roadset62_targetPos_train.csv'        # Local file to store data


def get_train_readouts():

    if not path.exists(TRAIN_FILE): 
        data_df = get_train_data_from_db()

        pivoted_df = data_df.pivot('spillID', 'name', 'value')
        pivoted_df = pivoted_df.replace(-9999,np.nan).dropna(axis=0,how='any')
        
        zero_std_series = (pivoted_df.describe().ix['std'] == 0)
        zero_std_features = zero_std_series[zero_std_series == True].index.values

        _ = pivoted_df.drop(zero_std_features, axis=1, inplace=True)

        targpos_df = data_df[['spillID','targetPos']].drop_duplicates().sort('spillID')

        full_df = pd.merge(pivoted_df, targpos_df, how='left', right_on='spillID', left_index=True)
        full_df = full_df.set_index('spillID')

        full_df.to_csv(TRAIN_FILE)
    else:
        full_df = pd.read_csv(TRAIN_FILE, index_col='spillID')

    return full_df


def get_test_readouts():

    if not path.exists(TEST_FILE): 
        data_df = get_test_data_from_db()

        pivoted_df = data_df.pivot('spillID', 'name', 'value')
        pivoted_df = pivoted_df.replace(-9999,np.nan).dropna(axis=0,how='any')
        
        zero_std_series = (pivoted_df.describe().ix['std'] == 0)
        zero_std_features = zero_std_series[zero_std_series == True].index.values

        _ = pivoted_df.drop(zero_std_features, axis=1, inplace=True)

        #full_df = pd.merge(pivoted_df, data_df['spillID'], how='left',
        #                   right_on='spillID', left_index=True)
        #print pivoted_df.head()
        #full_df = pivoted_df.set_index('spillID')
        pivoted_df.to_csv(TEST_FILE)
    else:
        pivoted_df = pd.read_csv(TEST_FILE, index_col='spillID')

    return pivoted_df


def get_test_data_from_db():

    create_query = """
                   CREATE TABLE IF NOT EXISTS %s.%s_test
                   (
                       spillID MEDIUMINT NOT NULL,
                       name VARCHAR(64),
                       value DOUBLE NOT NULL
                   )
                   """
    scaler_query =  """
                    INSERT INTO %s.%s_test
                    SELECT s.spillID, scalerName AS `name`, value
                    FROM Scaler INNER JOIN Spill s USING(spillID)
                    WHERE scalerName IS NOT NULL AND
                    s.spillID BETWEEN 416207 AND 424180 AND
                    spillType='EOS' AND
                    s.dataQuality & ~POW(2,26) = 0
                    """

    beam_query = """
                 INSERT INTO %s.%s_test
                 SELECT s.spillID, name, value
                 FROM Beam INNER JOIN Spill s USING(spillID)
                 WHERE name IS NOT NULL AND
                 LEFT(name,3)!='F:M' AND
                 name!='F:NM2SEM' AND
                 name!='U:TODB25' AND
                 name!='S:KTEVTC' AND
                 s.spillID BETWEEN 416207 AND 424180 AND
                 s.dataQuality & ~POW(2,26) = 0
                 """

    fetch_query = """SELECT * FROM %s.%s_test"""

    readouts_df = pd.DataFrame()
    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=SERVER,
                         db=SOURCE_SCHEMA,
                         port=PORT)

        cur = db.cursor()
        try:
            cur.execute("CREATE DATABASE IF NOT EXISTS %s" % ANALYSIS_SCHEMA)
        except:
            pass
        try:
            cur.execute(create_query % (ANALYSIS_SCHEMA, ANALYSIS_TABLE))
        except:
            pass
        cur.execute("TRUNCATE TABLE %s.%s_test" % (ANALYSIS_SCHEMA, ANALYSIS_TABLE))
        cur.execute(scaler_query % (ANALYSIS_SCHEMA, ANALYSIS_TABLE))
        cur.execute(beam_query % (ANALYSIS_SCHEMA, ANALYSIS_TABLE))

        readouts_df = pd.read_sql(fetch_query %
                                  (ANALYSIS_SCHEMA, ANALYSIS_TABLE), db)

        if db:
            db.close()

    except mdb.Error, e:

            print "Error %d: %s" % (e.args[0], e.args[1])

    return readouts_df


def get_train_data_from_db():

    # Aggregate data into our analysis schema and table.
    # Table defined here:
    create_query = """
                   CREATE TABLE IF NOT EXISTS %s.%s
                   (
                       spillID MEDIUMINT NOT NULL,
                       name VARCHAR(64),
                       value DOUBLE NOT NULL,
                       targetPos INT NOT NULL
                   )
                   """
    scaler_query =  """
                    INSERT INTO %s.%s
                    SELECT s.spillID, scalerName AS `name`, value, targetPos
                    FROM Scaler
                    INNER JOIN Spill s
                    USING(spillID)
                    WHERE scalerName IS NOT NULL AND
                    s.spillID NOT BETWEEN 409000 AND 430000 AND
                    s.spillID NOT BETWEEN 416207 AND 424180 AND
                    s.spillID NOT BETWEEN 482574 AND 484924 AND
                    spillType='EOS' AND
                    s.dataQuality = 0
                    """

    beam_query = """
                INSERT INTO %s.%s
                SELECT s.spillID, name, value, targetPos
                FROM Beam
                INNER JOIN Spill s
                USING(spillID)
                WHERE name IS NOT NULL AND
                LEFT(name,3)!='F:M' AND
                name!='F:NM2SEM' AND
                name!='U:TODB25' AND
                name!='S:KTEVTC' AND
                s.spillID NOT BETWEEN 409000 AND 430000 AND
                s.spillID NOT BETWEEN 416207 AND 424180 AND
                s.spillID NOT BETWEEN 482574 AND 484924 AND
                s.dataQuality = 0
                """

    fetch_query = """SELECT * FROM %s.%s"""

    readouts_df = pd.DataFrame()
    try:
        db = mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=SERVER,
                         db=SOURCE_SCHEMA,
                         port=PORT)

        cur = db.cursor()

        cur.execute("SHOW DATABASES LIKE '%s'" % ANALYSIS_SCHEMA)

        if cur.rowcount != 0:
            cur.execute("DROP DATABASE %s" % ANALYSIS_SCHEMA)

        cur.execute("CREATE DATABASE %s" % ANALYSIS_SCHEMA)
        cur.execute(create_query % (ANALYSIS_SCHEMA, ANALYSIS_TABLE))

        cur.execute(scaler_query % (ANALYSIS_SCHEMA, ANALYSIS_TABLE))
        cur.execute(beam_query % (ANALYSIS_SCHEMA, ANALYSIS_TABLE))

        readouts_df = pd.read_sql(fetch_query %
                                  (ANALYSIS_SCHEMA, ANALYSIS_TABLE), db)

        if db:
            db.close()

    except mdb.Error, e:

            print "Error %d: %s" % (e.args[0], e.args[1])

    return readouts_df

def confusion(labels, results, names):
    plt.figure(figsize=(10, 10))

    # Make a 2D histogram from the test and result arrays
    pts, xe, ye = np.histogram2d(labels.astype(int), results.astype(int), bins=len(names))

    # For simplicity we create a new DataFrame
    pd_pts = pd.DataFrame(np.flipud(pts.astype(int)), index=np.flipud(names), columns=names )

    # Display heatmap and add decorations
    hm = sns.heatmap(pd_pts, annot=True, fmt="d", cbar=False)

    _, ylabels = plt.xticks()
    _, xlabels = plt.yticks()
    plt.setp(xlabels, rotation=45)
    plt.setp(ylabels, rotation=45)
    plt.xlabel("Actual", size=22)
    plt.ylabel("Prediction", size=22)

    plt.show()

    return pts


def per_target_accuracy(hist2d_pts, names):

    for i in range(len(names)):
        rowsum = np.sum(hist2d_pts.T[i])
        if rowsum>0:
            print names[i] + ":   \t" + str(round((hist2d_pts[i][i] / np.sum(hist2d_pts.T[i]))*100,2)) + "%"
        else:
            print names[i] + ":   \tN/A"

    return 0


def relabel(label_array):
    label_array_revised = label_array.copy()
    label_array_revised[label_array_revised == 4] = 2
    label_array_revised[label_array_revised == 5] = 4
    label_array_revised[label_array_revised == 6] = 5
    label_array_revised[label_array_revised == 7] = 6
                    
    return label_array_revised


def norm_to_g2sem(data_df):

    normed_df = pd.DataFrame( (data_df.values / data_df[['S:G2SEM']].values) * 5000000000000.0,
            columns=data_df.columns )
    _ = normed_df.drop('S:G2SEM', axis=1, inplace=True)

    return normed_df


def test_rfc(data_df, labels):

    normed_df = norm_to_g2sem(data_df)

    data = normed_df.values
    scale = StandardScaler().fit(data)
    data_scaled = scale.transform(data)

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt',
                                 min_samples_split=1, random_state=2)

    d_train, d_test, l_train, l_test \
            = cross_validation.train_test_split(data_scaled, labels, test_size=0.33, random_state=2)
    rfc.fit(d_train, l_train)
    result = rfc.predict(d_test)

    # Define names for the target positions
    names = ['Hydrogen','Empty/None','Deuterium','Carbon','Iron','Tungsten']

    pts = confusion(l_test, result, names)
    per_target_accuracy(pts, names)

    return rfc


def get_useful_features(data_df, rfc):

    features = data_df.drop('S:G2SEM', axis=1).columns.values
    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]

    useful_feature_list = []
    for f in range(25):
        useful_feature_list.append(features[indices[f]])

    return useful_feature_list


def make_full_rfc(data_df, labels, useful_features):
    
    normed_df = norm_to_g2sem(data_df)
    train_data = normed_df[useful_features].values

    scale = StandardScaler().fit(train_data)
    train_data_scaled = scale.transform(train_data)

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt',
                                 min_samples_split=1, random_state=2).fit(train_data_scaled, labels)

    return (rfc, scale)


def main():
    
    # Get training data and labels
    full_df = get_train_readouts()

    # Extract and process labels
    labels = full_df.values[:,-1]
    labels_revised = relabel(labels)
    
    # Extract training data
    data_df = full_df.drop('targetPos', axis=1)

    # Make a test rfc and validate it
    rfc = test_rfc(data_df, labels_revised)

    # Use that rfc to glean the most relevant features
    useful_features = get_useful_features(full_df, rfc)

    # Train an rfc using only useful features of all the data
    rfc, scale = make_full_rfc(data_df, labels_revised, useful_features)

    # Now, get the test data, scale it, and predict the labels
    test_data_df = get_test_readouts()
    norm_test_data_df = norm_to_g2sem(test_data_df)
    norm_test_data = norm_test_data_df[useful_features].values
    test_scale = StandardScaler().fit(norm_test_data)
    test_data_scaled = test_scale.transform(norm_test_data)
    #test_data_scaled = scale.transform(test_data)
    results = rfc.predict(test_data_scaled)
    
    hist, bin_edges = np.histogram(results, bins=6)
    print hist
    # shifter_labels = get_shifter_labels()

    return 0


if __name__=='__main__':
    main()
