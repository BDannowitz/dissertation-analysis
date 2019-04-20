#!/usr/bin/python

import MySQLdb as mdb
from servers import server_dict


def mc_fill_kDimuon_table(server, schema, table, source_schema):

    try:
        db = mdb.connect(read_default_file='../.my.cnf', read_default_group='guest',
                         db=source_schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        query = "SELECT COUNT(*) FROM kDimuon"
        cur.execute(query)
        print str(cur.fetchone()[0]) + ' dimuons from source table.\n'

        tmp_table = table + '_tmp'
        query = "CREATE TEMPORARY TABLE %s.%s LIKE kDimuon"
        cur.execute(query % (schema, tmp_table))

        query = """
                INSERT INTO %s.%s
                SELECT k.*
                FROM kDimuon k
                WHERE
                    # Good fit to tracks
                    chisq_dimuon < 15 AND
                    # Desired mass
                    mass > 4.2 AND mass < 10
                """

        cur.execute(query % (schema, tmp_table))
        dimuon_count = cur.rowcount
        print str(cur.rowcount) + " entries copied over with mass between 4.2 and 10 GeV and chisq<25"

        query = """DELETE FROM %s.%s
                      WHERE
                          xB NOT BETWEEN 0 AND 1 OR
                          xT NOT BETWEEN 0 AND 1 OR
                          xF NOT BETWEEN -1 AND 1"""

        cur.execute(query % (schema, tmp_table))
        dimuon_count -= cur.rowcount
        print str(dimuon_count) + ' dimuons: after ' + str(cur.rowcount) + " entries deleted for x-range cuts"

        query = """DELETE FROM %s.%s
                      WHERE
                          dz NOT BETWEEN -300 AND -90 OR
                          ABS(dx) >= 2 OR
                          ABS(dy) >= 2 OR
                          ABS(dpx) >= 3 OR
                          ABS(dpy) >= 3 OR
                          dpz NOT BETWEEN 30 AND 120 OR
                          ABS(trackSeparation) >= 200 OR
                          px1 <= 0 OR
                          px2 >= 0
                          """
        # I'M A COMMENT! LOOK AT ME! CHECK YOUR POLARITIES! NORMAL POLARITY SHOULD MAKE px1 POSITIVE!

        cur.execute(query % (schema, tmp_table))
        dimuon_count -= cur.rowcount
        print str(dimuon_count) + ' dimuons: after ' + str(cur.rowcount) + " entries deleted for dimuon positional and momentum cuts"

        query = """DELETE %s.%s FROM %s.%s
                            INNER JOIN kTrack kpos
                            ON %s.runID = kpos.runID AND %s.eventID = kpos.eventID AND posTrackID = kpos.trackID
                            INNER JOIN kTrack kneg
                            ON %s.runID = kneg.runID AND %s.eventID = kneg.eventID AND negTrackID = kneg.trackID
                       WHERE
                            kpos.numHits < 15 OR
                            kneg.numHits < 15 OR
                            (kpos.numHits < 18 AND kpos.pz1 <= 18) OR
                            (kneg.numHits < 18 AND kneg.pz1 <= 18) OR
                            (kpos.chisq / (kpos.numHits - 5)) >= 6.0 OR
                            (kneg.chisq / (kneg.numHits - 5)) >= 6.0 OR
                            kpos.z0 <= -300 OR
                            kpos.z0 >= 0 OR
                            kneg.z0 <= -300 OR
                            kneg.z0 >= 0
                            """

        cur.execute(query % (schema, tmp_table, schema, tmp_table, tmp_table, tmp_table, tmp_table, tmp_table))
        dimuon_count -= cur.rowcount
        print str(dimuon_count) + ' dimuons: after ' + str(cur.rowcount) + " Rows deleted for track-level cuts"

        query = """DELETE %s.%s FROM %s.%s
                        # Join the positive trackID with kTrack to get track details
                        INNER JOIN kTrack kpos
                        ON %s.runID = kpos.runID AND %s.eventID = kpos.eventID AND posTrackID = kpos.trackID
                        # Join the negative trackID with kTrack to get track details
                        INNER JOIN kTrack kneg
                        ON %s.runID = kneg.runID AND %s.eventID = kneg.eventID AND negTrackID = kneg.trackID
                   WHERE
                        SQRT(POW(kpos.x_dump,2)+POW(kpos.y_dump,2)) - SQRT(POW(kpos.x_target,2)+POW(kpos.y_target,2)) <=
                            (9.4431 - 0.356141*kpos.pz0 + 0.00566071*POW(kpos.pz0,2) - 3.05556e-5*POW(kpos.pz0,3))
                        OR
                        SQRT(POW(kneg.x_dump,2)+POW(kneg.y_dump,2)) - SQRT(POW(kneg.x_target,2)+POW(kneg.y_target,2)) <=
                            (9.4431 - 0.356141*kneg.pz0 + 0.00566071*POW(kneg.pz0,2) - 3.05556e-5*POW(kneg.pz0,3))
                """

        cur.execute(query % (schema, tmp_table, schema, tmp_table, tmp_table, tmp_table, tmp_table, tmp_table))
        dimuon_count -= cur.rowcount
        print str(dimuon_count) + ' dimuons: after ' + str(cur.rowcount) + " Rows deleted for dump radius cut"

        query = "ALTER TABLE %s.%s ADD targpos SMALLINT DEFAULT NULL"
        cur.execute(query % (schema, tmp_table))

        # Populate the targets used
        query = """UPDATE %s.%s k
                        INNER JOIN %s.Spill s USING(spillID)
                        SET k.targpos = s.targetPos"""
        cur.execute(query % (schema, tmp_table, source_schema))

        query = "ALTER TABLE %s.%s ADD sigWeight DOUBLE DEFAULT NULL"
        cur.execute(query % (schema, tmp_table))

        # Populate the targets used
        query = """UPDATE %s.%s k
                        INNER JOIN %s.mDimuon md USING(spillID, eventID)
                        SET k.sigWeight = md.sigWeight"""
        cur.execute(query % (schema, tmp_table, source_schema))

        query = "ALTER TABLE %s.%s ADD eventsThrown DOUBLE DEFAULT NULL"
        cur.execute(query % (schema, tmp_table))

        # Populate the targets used
        query = """UPDATE %s.%s k
                        INNER JOIN %s.mRun mr USING(runID)
                        SET k.eventsThrown = mr.eventsThrown"""
        cur.execute(query % (schema, tmp_table, source_schema))

        query = "SELECT COUNT(*) FROM %s.%s"
        cur.execute(query % (schema, tmp_table))
        print str(cur.fetchone()[0]) + ' dimuons total after all cuts.'

        query = "INSERT INTO %s.%s SELECT * FROM %s.%s"
        cur.execute(query % (schema, table, schema, tmp_table))

        if db:
            db.close()

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def mc_truth_fill_dimuon_table(server, schema, table, source_schema):

    try:
        db = mdb.connect(read_default_file='../.my.cnf', read_default_group='guest',
                         db=source_schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()

        query = "SELECT COUNT(*) FROM mDimuon"
        cur.execute(query)
        print str(cur.fetchone()[0]) + ' dimuons from source table.\n'

        tmp_table = table + '_tmp'
        query = "CREATE TEMPORARY TABLE %s.%s LIKE mDimuon"
        cur.execute(query % (schema, tmp_table))

        query = """
                INSERT INTO %s.%s
                SELECT k.*
                FROM mDimuon k
                WHERE
                    # Desired mass
                    mass > 4.2 AND mass < 10
                """

        cur.execute(query % (schema, tmp_table))
        dimuon_count = cur.rowcount
        print str(cur.rowcount) + " entries copied over with mass between 4.2 and 10 GeV and chisq<25"

        query = """DELETE FROM %s.%s
                      WHERE
                          xB NOT BETWEEN 0 AND 1 OR
                          xT NOT BETWEEN 0 AND 1 OR
                          xF NOT BETWEEN -1 AND 1"""

        cur.execute(query % (schema, tmp_table))
        dimuon_count -= cur.rowcount
        print str(dimuon_count) + ' dimuons: after ' + str(cur.rowcount) + " entries deleted for x-range cuts"

        query = """DELETE FROM %s.%s
                      WHERE
                          dz NOT BETWEEN -300 AND -90 OR
                          ABS(dx) >= 2 OR
                          ABS(dy) >= 2 OR
                          ABS(dpx) >= 3 OR
                          ABS(dpy) >= 3 OR
                          dpz NOT BETWEEN 30 AND 120
                          """

        cur.execute(query % (schema, tmp_table))
        dimuon_count -= cur.rowcount
        print str(dimuon_count) + ' dimuons: after ' + str(cur.rowcount) + " entries deleted for dimuon positional and momentum cuts"

        query = """DELETE %s.%s FROM %s.%s
                            INNER JOIN mTrack kpos
                            ON mTrackID1 = kpos.mTrackID AND %s.runID = kpos.runID
                            INNER JOIN mTrack kneg
                            ON mTrackID2 = kneg.mTrackID AND %s.runID = kneg.runID
                       WHERE
                            kpos.px0 <= 0 OR
                            kneg.px0 >= 0
                            """
        # I'M A COMMENT! LOOK AT ME! CHECK YOUR POLARITIES! NORMAL POLARITY SHOULD MAKE px1 POSITIVE!

        cur.execute(query % (schema, tmp_table, schema, tmp_table, tmp_table, tmp_table))
        dimuon_count -= cur.rowcount
        print str(dimuon_count) + ' dimuons: after ' + str(cur.rowcount) + " Rows deleted for track-level cuts"

        query = "ALTER TABLE %s.%s ADD targpos SMALLINT DEFAULT NULL"
        cur.execute(query % (schema, tmp_table))

        # Populate the targets used
        query = """UPDATE %s.%s k
                        INNER JOIN %s.Spill s USING(spillID)
                        SET k.targpos = s.targetPos"""
        cur.execute(query % (schema, tmp_table, source_schema))

        query = "ALTER TABLE %s.%s ADD eventsThrown DOUBLE DEFAULT NULL"
        cur.execute(query % (schema, tmp_table))

        # Populate the targets used
        query = """UPDATE %s.%s k
                        INNER JOIN %s.mRun mr USING(runID)
                        SET k.eventsThrown = mr.eventsThrown"""
        cur.execute(query % (schema, tmp_table, source_schema))

        query = "SELECT COUNT(*) FROM %s.%s"
        cur.execute(query % (schema, tmp_table))
        print str(cur.fetchone()[0]) + ' dimuons total after all cuts.'

        query = "INSERT INTO %s.%s SELECT * FROM %s.%s"
        cur.execute(query % (schema, table, schema, tmp_table))

        if db:
            db.close()

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return 1

    return 0


def main():
    print 'Hello World!'


if __name__ == '__main__':
    main()
