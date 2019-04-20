#!/usr/bin/python

import pymysql as Mdb
from servers import server_dict

roadset_dict = {49: {'lower': 8184, 'upper': 8896},
                56: {'lower': 8897, 'upper': 8908},
                57: {'lower': 8910, 'upper': 10420},
                59: {'lower': 10421, 'upper': 10912},
                61: {'lower': 10914, 'upper': 11028},
                62: {'lower': 11040, 'upper': 12438},
                67: {'lower': 12525, 'upper': 15789},
                70: {'lower': 15793, 'upper': 16076}}

revision_list = ['R001', 'R003', 'R004', 'R005']


def get_productions(revision='R005', server=None, roadset=None):

    prod_dict = {}
    server_dict = servers.server_dict

    for server_entry in server_dict:
        if (server is None) or (server_entry == server):

            prod_dict[server_entry] = []

            try:
                db = Mdb.connect(read_default_file='../.my.cnf',
                                 read_default_group='guest',
                                 host=server_entry,
                                 port=server_dict[server_entry]['port'])

                if not db:
                    print "No connection!"
                cur = db.cursor()

                for revision_entry in revision_list:
                    if (revision is None) or (revision_entry == revision):
                        query = "SHOW DATABASES LIKE 'run\_______\_" + revision_entry + "'"
                        cur.execute(query)
                        rows = cur.fetchall()

                        for row in rows:
                            run = int(row[0][4:10])
                            for roadset_entry in roadset_dict:
                                if ((roadset is None) and (row[0] not in prod_dict[server_entry])) or \
                                   ((roadset == roadset_entry) and \
                                   (roadset_dict[roadset_entry]['lower'] <= run <= roadset_dict[roadset_entry]['upper'])):
                                    prod_dict[server_entry].append(row[0])

                if db:
                    db.close()

            except Mdb.Error, e:
                try:
                    print "MySQL Error [%d]: %s" % (e.args[0], e.args[1])
                except IndexError:
                    print "MySQL Error: %s" % str(e)

    return prod_dict

def schema_exists(server, schema):
    """
    Takes a server and schema
    Returns:
        1 if schema exists (case-sensitive)
        0 if schema does not exist
       -1 if query or connection error occurs
    """

    exists = -1

    try:

        db = Mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()
        cur.execute("SHOW DATABASES LIKE '" + schema + "'")
        exists = 1 if cur.rowcount > 0 else 0

        return exists

    except Mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        return -1


def table_exists(server, schema, table):
    """
    Takes a server, schema, and table name
    Returns:
        1 if table exists (case-sensitive)
        0 if table does not exist
       -1 if query or connection error occurs
    """
    exists = -1

    try:
        db = Mdb.connect(read_default_file='../.my.cnf',
                         read_default_group='guest',
                         db=schema,
                         host=server,
                         port=server_dict[server]['port'])
        cur = db.cursor()
        cur.execute("SHOW TABLES LIKE '" + table + "'")
        exists = 1 if cur.rowcount > 0 else 0

    except Mdb.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
        return -1

    return exists


def get_roadset(run):
    for roadset in roadset_dict:
        if roadset_dict[roadset]['lower'] <= run <= roadset_dict[roadset]['upper']:
            return roadset
    return None


def MyProductions():
    get_productions()


if __name__ == "__main__":
    MyProductions()
