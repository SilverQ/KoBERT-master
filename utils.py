# -*- coding:utf -*-
import psycopg2 as pg2
from psycopg2.extras import DictCursor
# import time
import numpy as np
# import csv
from utils import *

def conn_string():
    product_connection_string = "dbname={dbname} user={user} host={host} password={password} port={port}".format(dbname='db_ipc', user='scorpion', host='pgdb05.nips.local', password='scorpion', port=5432)
    return product_connection_string

def iter_row(cursor, size=10):
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            break
        for row in rows:
            yield row

def execute(cur, query):
    cur.execute(query)
    return cur.fetchall()

def proj_id(cur, pr_nm):
    query = "select project_id from biz.t_stat_project where project_name = '{pr_nm}'".format(pr_nm=pr_nm)
#     print(query)
    cur.execute(query)
    results = cur.fetchone()
#     print(results)
    return results[0]

def escape_string(raw):
    return raw.replace("\"", "\\\"").replace("'", "''")

def connect(conn_info, application_name=''):
    conn = pg2.connect(conn_info)

    # this is setting for ERROR, invalid byte sequence for encoding "UTF8": 0x00
    # conn.cursor().execute("SET standard_conforming_strings=on")
    
    if len(application_name) > 0:
        cur = conn.cursor()
        cur.execute("SET application_name TO {0}".format(application_name))
        cur.close()
        
    return conn
