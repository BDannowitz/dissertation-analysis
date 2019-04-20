#!/usr/bin/python

import pymysql as Mdb
from servers import server_dict
import ConfigParser as Cfg
import json
from os import path

CFG_PATH = path.abspath(path.dirname(__file__))
CONFIG_FILE = ('%s/emc.cfg' % CFG_PATH)


def parse_cfg_items(cfg, section):

    items_dict = {('%s' % x[0]): json.loads(x[1]) for x in cfg.items(section)}

    return items_dict


def make_analysis_table(server, schema, table, source_schema,
                        truth_mc=False, tracked_mc=False,
                        likesign=False):
    try:
        db = Mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         db=source_schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()
        
        if truth_mc:
            query = ("CREATE TABLE %s.%s LIKE mDimuon" % (schema, table))
            cur.execute(query)
            query = ("""
                     ALTER TABLE %s.%s
                     ADD targetPos SMALLINT DEFAULT NULL,
                     ADD eventsThrown DOUBLE DEFAULT NULL
                     """ % (schema, table))
            cur.execute(query)
        if tracked_mc:
            query = ("CREATE TABLE %s.%s LIKE kDimuon" % (schema, table))
            cur.execute(query)
            query = ("""ALTER TABLE %s.%s
                        DROP PRIMARY KEY,
                        ADD sigWeight DOUBLE DEFAULT NULL,
                        ADD eventsThrown DOUBLE DEFAULT NULL
                     """ % (schema, table))
            cur.execute(query)
        if not truth_mc and not tracked_mc:
            if likesign:
                source_table = "kDimuonMM"
            else:
                source_table = "kDimuon"
            query = ("CREATE TABLE %s.%s LIKE %s" % (schema, table, source_table))
            cur.execute(query)
            query = ("""ALTER TABLE %s.%s 
                        ADD QIESum FLOAT DEFAULT NULL,
                        ADD chamber_intensity FLOAT DEFAULT NULL,
                        ADD trigger_intensity FLOAT DEFAULT NULL,
                        ADD weight FLOAT DEFAULT NULL
                     """ % (schema, table))
            cur.execute(query)

    except Mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def drop_analysis_table(server, schema, table):
    try:
        db = Mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         db=schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        query = "DROP TABLE IF EXISTS %s.%s"
        cur.execute(query % (schema, table))

    except Mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def get_dimuons(schema, table, cur, cfg_dict, truth_mc=False):
    """Load subset of dimuons into designated table."""
    if truth_mc:
        if 'mass' not in cfg_dict:
            print ("Missing some configuration items. Exiting...")
            return -1
    else:
        required_cfg = ('maxChisq', 'mass')
        if not all(item in cfg_dict for item in required_cfg):
            print ("Missing some configuration items. Exiting...")
            return -1

    if truth_mc:
        query = ("""
                 INSERT INTO %s.%s
                 SELECT m.*
                 FROM mDimuon m
                 WHERE mass > %f AND mass < %f
                 """ % (schema, table,
                        cfg_dict['mass'][0], cfg_dict['mass'][1]))
    
    else:
        query = ("""
                 INSERT INTO %s.%s
                 SELECT k.*
                 FROM kDimuon k
                 WHERE
                     chisq_dimuon < %f AND
                     mass > %f AND mass < %f
                 """ % (schema, table, cfg_dict['maxChisq'],
                        cfg_dict['mass'][0], cfg_dict['mass'][1]))

    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    if truth_mc:
        print ("%i entries copied over with mass between %f and %f GeV " %
               (dimuon_count, cfg_dict['mass'][0], cfg_dict['mass'][1]))
    else:
        print ("%i entries copied over with mass between %f and %f GeV "
               "and chisq<%f" % (dimuon_count, cfg_dict['mass'][0],
                                 cfg_dict['mass'][1], cfg_dict['maxChisq']))

    return dimuon_count


def xrange_cut(schema, table, cur, cfg_dict):
    """Load subset of dimuons into designated table."""

    required_cfg = ('xB', 'xT', 'xF')

    if not all(item in cfg_dict for item in required_cfg):
        print ("Missing some configuration items. Exiting...")
        return -1

    query = ("""
             DELETE FROM %s.%s
             WHERE
             xB <= %f OR
             xB >= %f OR
             xT <= %f OR
             xT >= %f OR
             xF <= %f OR
             xF >= %f
             """ % (schema, table,
                    cfg_dict['xB'][0], cfg_dict['xB'][1],
                    cfg_dict['xT'][0], cfg_dict['xT'][1],
                    cfg_dict['xF'][0], cfg_dict['xF'][1]))

    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def dimuon_cut(schema, table, cur, cfg_dict, truth_mc=False):
    """Load subset of dimuons into designated table."""

    if truth_mc:
        required_cfg = ('dz', 'dx', 'dy', 'dpx', 'dpy', 'dpz')
    else:
        required_cfg = ('dz', 'dx', 'dy', 'dpx', 'dpy',
                        'dpz', 'trackSeparation')

    if not all(item in cfg_dict for item in required_cfg):
        print ("Missing some configuration items. Exiting...")
        return -1

    query = ("""
             DELETE FROM %s.%s
             WHERE
                dz <= %f OR
                dz >= %f OR
                dx <= %f OR
                dx >= %f OR
                dy <= %f OR
                dy >= %f OR
                dpx <= %f OR
                dpx >= %f OR
                dpy <= %f OR
                dpy >= %f OR
                dpz <= %f OR
                dpz >= %f
             """
             % (schema, table,
                cfg_dict['dz'][0], cfg_dict['dz'][1],
                cfg_dict['dx'][0], cfg_dict['dx'][1],
                cfg_dict['dy'][0], cfg_dict['dy'][1],
                cfg_dict['dpx'][0], cfg_dict['dpx'][1],
                cfg_dict['dpy'][0], cfg_dict['dpy'][1],
                cfg_dict['dpz'][0], cfg_dict['dpz'][1]))

    if not truth_mc:
        query += (""" OR
                  trackSeparation <= %f OR
                  trackSeparation >= %f OR
                  px1 <= 0 OR px2 >= 0
                  """ %
                  (cfg_dict['trackSeparation'][0],
                   cfg_dict['trackSeparation'][1]))
        # If the polarity bug is fixed, add in something like this:
        # if '67' in source_schema or '70' in source_schema:
        #    query += "px1 >= 0 OR px2 <= 0"
        # else:
        #    query += "px1 <= 0 OR px2 >= 0"

    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def spill_cut(schema, table, source, cur, cfg_dict):
    """Load subset of dimuons into designated table."""

    if 'SpillDQ' not in cfg_dict:
        print ("Missing some configuration items. Exiting...")
        return -1


    query = ("""
             DELETE %s.%s
             FROM %s.%s INNER JOIN %s.Spill s USING(spillID)
             WHERE s.dataQuality ^ %i != 0
             """
             % (schema, table, schema, table, source,
                cfg_dict['SpillDQ']))

    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def track_cut(schema, table, cur, cfg_dict, truth_mc=False):
    """Load subset of dimuons into designated table."""

    if truth_mc:

        query = ("""
                 DELETE %s.%s FROM %s.%s
                 INNER JOIN mTrack kpos
                    ON mTrackID1 = kpos.mTrackID AND %s.runID = kpos.runID
                 INNER JOIN mTrack kneg
                    ON mTrackID2 = kneg.mTrackID AND %s.runID = kneg.runID
                 WHERE
                    kpos.px0 <= 0 OR
                    kneg.px0 >= 0
                 """ % (schema, table, schema, table, table, table))
    else:
        required_cfg = ('minNumHits', 'numHitPz1', 'maxChisqPDF', 'z0')

        if not all(item in cfg_dict for item in required_cfg):
            print ("Missing some configuration items. Exiting...")
            return -1

        query = ("""
                 DELETE %s.%s FROM %s.%s
                 INNER JOIN kTrack kpos
                    ON posTrackID = kpos.trackID AND %s.runID = kpos.runID
                 INNER JOIN kTrack kneg
                    ON negTrackID = kneg.trackID AND %s.runID = kneg.runID
                 WHERE
                    kpos.numHits < %f OR
                    kneg.numHits < %f OR
                    (kpos.numHits < 18 AND kpos.pz1 <= %f) OR
                    (kneg.numHits < 18 AND kneg.pz1 <= %f) OR
                    (kpos.chisq / (kpos.numHits - 5)) >= %f OR
                    (kneg.chisq / (kneg.numHits - 5)) >= %f OR
                    kpos.z0 <= %f OR
                    kpos.z0 >= %f OR
                    kneg.z0 <= %f OR
                    kneg.z0 >= %f OR
                    kpos.roadID=0 OR
                    kneg.roadID=0 OR 
                    SIGN(kpos.roadID)*SIGN(kneg.roadID)>0
                 """ % (schema, table, schema, table, table, table,
                        cfg_dict['minNumHits'], cfg_dict['minNumHits'],
                        cfg_dict['numHitPz1'], cfg_dict['numHitPz1'],
                        cfg_dict['maxChisqPDF'], cfg_dict['maxChisqPDF'],
                        cfg_dict['z0'][0], cfg_dict['z0'][1],
                        cfg_dict['z0'][0], cfg_dict['z0'][1]))

    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def low_rf_cut(schema, table, source, cur):
    
    query = ("""
             DELETE %s.%s FROM
             %s.%s
             INNER JOIN %s.QIE USING(runID, eventID)
             WHERE `RF-16`<1 OR
                   `RF-15`<1 OR
                   `RF-14`<1 OR
                   `RF-13`<1 OR
                   `RF-12`<1 OR
                   `RF-11`<1 OR
                   `RF-10`<1 OR
                   `RF-09`<1 OR
                   `RF-08`<1 OR
                   `RF-07`<1 OR
                   `RF-06`<1 OR
                   `RF-05`<1 OR
                   `RF-04`<1 OR
                   `RF-03`<1 OR
                   `RF-02`<1 OR
                   `RF-01`<1 OR
                   `RF+00`<1 OR
                   `RF+16`<1 OR
                   `RF+15`<1 OR
                   `RF+14`<1 OR
                   `RF+13`<1 OR
                   `RF+12`<1 OR
                   `RF+11`<1 OR
                   `RF+10`<1 OR
                   `RF+09`<1 OR
                   `RF+08`<1 OR
                   `RF+07`<1 OR
                   `RF+06`<1 OR
                   `RF+05`<1 OR
                   `RF+04`<1 OR
                   `RF+03`<1 OR
                   `RF+02`<1 OR
                   `RF+01`<1
               """ % (schema, table, schema, table, source))
    
    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def rf_thresh_cut(schema, table, source, cur):
    
    query = ("""
             DELETE %s.%s FROM
             %s.%s
             INNER JOIN %s.QIE USING(runID, eventID)
             INNER JOIN %s.BeamDAQ USING(spillID)
             WHERE `RF-08`>Inh_thres OR
                   `RF-07`>Inh_thres OR
                   `RF-06`>Inh_thres OR
                   `RF-05`>Inh_thres OR
                   `RF-04`>Inh_thres OR
                   `RF-03`>Inh_thres OR
                   `RF-02`>Inh_thres OR
                   `RF-01`>Inh_thres OR
                   `RF+00`>Inh_thres OR
                   `RF+08`>Inh_thres OR
                   `RF+07`>Inh_thres OR
                   `RF+06`>Inh_thres OR
                   `RF+05`>Inh_thres OR
                   `RF+04`>Inh_thres OR
                   `RF+03`>Inh_thres OR
                   `RF+02`>Inh_thres OR
                   `RF+01`>Inh_thres
               """ % (schema, table,
                      schema, table,
                      source, source))
    
    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count



def dump_radius_cut(schema, table, cur, cfg_dict):
    """Load subset of dimuons into designated table."""
    
    if 'MinDumpMinusTarget' not in cfg_dict:
        print ("Missing some configuration items. Exiting...")
        return -1

    query = ("""
             DELETE %s.%s FROM %s.%s
             INNER JOIN kTrack kpos
                 ON posTrackID = kpos.trackID AND %s.runID = kpos.runID
             INNER JOIN kTrack kneg
                ON negTrackID = kneg.trackID AND %s.runID = kneg.runID
             WHERE
                kpos.chisq_dump - kpos.chisq_target <= %f OR
                kneg.chisq_dump - kneg.chisq_target <= %f
             """ % (schema, table, schema, table, table, table,
                    cfg_dict['MinDumpMinusTarget'],
                    cfg_dict['MinDumpMinusTarget']))

    try:
        cur.execute(query)
        dimuon_count = cur.rowcount

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def event_cut(schema, table, source, cur, cfg_dict):
    """If enabled, cut out cases where Event.MATRIX1 is not equal to 1."""

    if "RequireMatrix1" not in cfg_dict:
        print ("Missing some configuration items. Exiting...")
        return -1

    query = ("""
             DELETE %s.%s 
             FROM %s.%s
                LEFT JOIN %s.Event
                USING(runID, eventID)
             WHERE MATRIX1 != 1
             """
             % (schema, table, schema, table, source))

    try:
        if cfg_dict['RequireMatrix1']:
            cur.execute(query)
            dimuon_count = cur.rowcount
        else:
            dimuon_count = 0

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def kevent_cut(schema, table, source, cur, cfg_dict):
    """If enabled, cut out cases where kEvent is not equal to zero."""

    if "RequireKEventStatus" not in cfg_dict:
        print ("Missing some configuration items. Exiting...")
        return -1

    query = ("""
             DELETE %s.%s 
             FROM %s.%s
                LEFT JOIN %s.kEvent
                USING(spillID, eventID)
             WHERE status != 0
             """
             % (schema, table, schema, table, source))

    try:
        if cfg_dict['RequireKEventStatus']:
            cur.execute(query)
            dimuon_count = cur.rowcount
        else:
            dimuon_count = 0

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        print ("Error executing this query: \n%s" % query)
        return -1

    return dimuon_count


def add_fill_qie(schema, table, source, cur):

    alter_query = ("""
                   ALTER TABLE %s.%s
                   ADD QIESum FLOAT DEFAULT NULL
                   """ % (schema, table))
    update_query = ("""
                    UPDATE %s.%s k
                    INNER JOIN %s.BeamDAQ b USING(spillID)
                    SET k.QIESum = b.QIESum
                    """ % (schema, table, source))
    try:
        cur.execute(alter_query)
        cur.execute(update_query)

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        return -1

    return 0


def add_fill_intensity(schema, table, source, cur):

    alter_query = ("""
                   ALTER TABLE %s.%s
                   ADD chamber_intensity DOUBLE DEFAULT NULL,
                   ADD trigger_intensity DOUBLE DEFAULT NULL,
                   ADD weight DOUBLE DEFAULT NULL
                   """ % (schema, table))
    update_query1 = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.QIE q USING(runID, eventID)
                     SET k.chamber_intensity = q.Intensity_p
                     """ % (schema, table, source))
    update_query2 = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.QIE q USING(spillID, eventID)
                     INNER JOIN %s.Beam b USING(spillID)
                     INNER JOIN %s.BeamDAQ bd USING(spillID)
                     SET k.trigger_intensity = 
                        (q.`RF+00`)*(b.value/(bd.QIEsum-(588*360000*36.791)))
                     WHERE b.name='S:G2SEM'
                     """ % (schema, table, source, source, source))
    try:
        cur.execute(alter_query)
        cur.execute(update_query1)
        cur.execute(update_query2)

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        return -1

    return 0


def add_fill_roadID(schema, table, source, cur):
    
    alter_query = ("""
                   ALTER TABLE %s.%s
                   ADD posRoadID INT DEFAULT NULL,
                   ADD negRoadID INT DEFAULT NULL
                   """ % (schema, table))
    update_query1 = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.kTrack t ON
                     k.runID = t.runID AND k.posTrackID = t.trackID
                     SET k.posRoadID = t.roadID
                     """ % (schema, table, source))
    update_query2 = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.kTrack t ON
                     k.runID = t.runID AND k.negTrackID = t.trackID
                     SET k.negRoadID = t.roadID
                     """ % (schema, table, source))
    try:
        cur.execute(alter_query)
        cur.execute(update_query1)
        cur.execute(update_query2)

    except Mdb.Error, e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))
        return -1

    return 0


def fill_analysis_table(server, schema, table, source_schema,
                        tracked_mc=False, truth_mc=False, likesign=False):

    cfg = Cfg.RawConfigParser()
    cfg.optionxform = str
    cfg.read(CONFIG_FILE)
   
    if truth_mc:
        source_table = "mDimuon"
    else:
        source_table = "kDimuon"
    
    try:
        db = Mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         db=source_schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        # Peek and see how many total dimuons there are
        query = ("SELECT COUNT(*) FROM %s" % source_table)
        cur.execute(query)
        print ('%i dimuons from source table.' % (cur.fetchone()[0]))

        # Create temporary table for storing selected dimuons
        tmp_table = table + '_tmp'
        query = ("CREATE TEMPORARY TABLE %s.%s LIKE %s")
        cur.execute(query % (schema, tmp_table, source_table))

        # Select subset of dimuons to work with #####################
        dimuon_count = get_dimuons(schema, tmp_table, cur,
                                   parse_cfg_items(cfg, 'DimuonCuts'),
                                   truth_mc=truth_mc)
        if dimuon_count == -1:
            print ("Error fetching dimuons.")
            return 1
        
        # Get rid of bad x-range dimuons ############################
        rows_deleted = xrange_cut(schema, tmp_table, cur,
                                  parse_cfg_items(cfg, 'XrangeCuts'))
        if rows_deleted == -1:
            print ("Error removing bad x-ranges.")
            return 1
        dimuon_count -= rows_deleted
        print ('%i dimuons: after %i entries deleted for x-range cuts' %
               (dimuon_count, rows_deleted))

        # Get rid of dimuons with bad position and momenta ##########
        rows_deleted = dimuon_cut(schema, tmp_table, cur,
                                  parse_cfg_items(cfg, 'DimuonCuts'),
                                  truth_mc=truth_mc)
        if rows_deleted == -1:
            print ("Error applying dimuon-level cuts.")
            return 1
        dimuon_count -= rows_deleted
        print ('%i dimuons: after %i entries deleted for dimuon-level cuts' %
               (dimuon_count, rows_deleted))

        # Get rid of dimuons from bad spills ########################
        if not truth_mc and not tracked_mc:
            rows_deleted = spill_cut(schema, tmp_table, source_schema, cur,
                                     parse_cfg_items(cfg, 'SpillCuts'))
            if rows_deleted == -1:
                print ("Error applying spill-level cuts.")
                return 1
            dimuon_count -= rows_deleted
            print ('%i dimuons: after %i entries deleted for bad spill cuts' %
                   (dimuon_count, rows_deleted))

        # Get rid of dimuons with at least one bad track #############
        rows_deleted = track_cut(schema, tmp_table, cur,
                                 parse_cfg_items(cfg, 'TrackCuts'),
                                 truth_mc=truth_mc)
        if rows_deleted == -1:
            print ("Error applying track-level cuts.")
            return 1
        dimuon_count -= rows_deleted
        print ('%i dimuons: after %i entries deleted for track-level cuts' %
               (dimuon_count, rows_deleted))

        if not truth_mc:
            # Get rid of dimuons that fail the dump-radius check #########
            rows_deleted = dump_radius_cut(schema, tmp_table, cur,
                                           parse_cfg_items(cfg, 'DumpRadiusCut'))
            if rows_deleted == -1:
                print ("Error applying dump-radius cut.")
                return 1
            dimuon_count -= rows_deleted
            print ('%i dimuons: after %i entries deleted for dump-radius cut' %
                   (dimuon_count, rows_deleted))
       
            if not tracked_mc:
                # Get rid of dimuons that weren't triggered by MATRIX1 #######
                rows_deleted = event_cut(schema, tmp_table, source_schema, cur,
                                         parse_cfg_items(cfg, 'EventCuts'))
                if rows_deleted == -1:
                    print ("Error applying Event cut.")
                    return 1
                dimuon_count -= rows_deleted
                print ('%i dimuons: after %i entries deleted for event level cuts' %
                       (dimuon_count, rows_deleted))
                
                # Get rid of dimuons that have a non-zero kEvent status #######
                rows_deleted = kevent_cut(schema, tmp_table, source_schema, cur,
                                         parse_cfg_items(cfg, 'EventCuts'))
                if rows_deleted == -1:
                    print ("Error applying kEvent cut.")
                    return 1
                dimuon_count -= rows_deleted
                print ('%i dimuons: after %i entries deleted for kEvent status cut' %
                       (dimuon_count, rows_deleted))
                
                rows_deleted = low_rf_cut(schema, tmp_table, source_schema, cur)
                if rows_deleted == -1:
                    print ("Error applying low RF value cut.")
                    return 1
                dimuon_count -= rows_deleted
                print ('%i dimuons: after %i entries deleted for low RF value cut' %
                       (dimuon_count, rows_deleted))
                
                rows_deleted = rf_thresh_cut(schema, tmp_table, source_schema, cur)
                if rows_deleted == -1:
                    print ("Error applying RF threshold cut.")
                    return 1
                dimuon_count -= rows_deleted
                print ('%i dimuons: after %i entries deleted for RF threshold cut' %
                       (dimuon_count, rows_deleted))
                
                print("Adding QIE information to Dimuon table.")
                status = add_fill_qie(schema, tmp_table, source_schema, cur)
                if status:
                    print ("Error adding and populating QIEsum field.")
                
                print("Adding chamber_intensity information to Dimuon table.")
                status = add_fill_intensity(schema, tmp_table, source_schema, cur)
                if status:
                    print ("Error adding and populating chamber_intensity field.")

        if truth_mc:

            query = ("""
                     ALTER TABLE %s.%s
                     ADD targetPos SMALLINT DEFAULT NULL,
                     ADD eventsThrown DOUBLE DEFAULT NULL
                     """ % (schema, tmp_table))
            cur.execute(query)

            # Populate the targets used
            query = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.Spill s USING(spillID)
                     SET k.targetPos = s.targetPos
                     """ % (schema, tmp_table, source_schema))
            cur.execute(query)

            # Populate the eventsThrown value
            query = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.mRun mr USING(runID)
                     SET k.eventsThrown = mr.eventsThrown""" %
                     (schema, tmp_table, source_schema))
            cur.execute(query)

        elif tracked_mc:
            query = ("""
                     ALTER TABLE %s.%s
                     ADD sigWeight DOUBLE DEFAULT NULL,
                     ADD eventsThrown DOUBLE DEFAULT NULL
                     """ % (schema, tmp_table))
            cur.execute(query)

            # Populate the targets used
            query = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.mDimuon m USING(eventID)
                     SET k.sigWeight = m.sigWeight
                     """ % (schema, tmp_table, source_schema))
            cur.execute(query)

            # Populate the eventsThrown value
            query = ("""
                     UPDATE %s.%s k
                     INNER JOIN %s.mRun mr USING(runID)
                     SET k.eventsThrown = mr.eventsThrown""" %
                     (schema, tmp_table, source_schema))
            cur.execute(query)

        query = "SELECT COUNT(*) FROM %s.%s"
        cur.execute(query % (schema, tmp_table))
        print ('%i dimuons after all cuts' % (cur.fetchone()[0]))

        query = "INSERT INTO %s.%s SELECT * FROM %s.%s"
        cur.execute(query % (schema, table, schema, tmp_table))

        if db:
            db.close()

    except Mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1


def main():
    print 'Hello World!'

    cfg = Cfg.RawConfigParser()
    cfg.optionxform = str
    cfg.read(CONFIG_FILE)

    print cfg.sections()
    section_dict = {('%s' % x[0]): json.loads(x[1]) for x in cfg.items('DimuonCuts')}
    print section_dict

    print parse_cfg_items(cfg, 'DimuonCuts')

    return 0


if __name__ == '__main__':
    main()
