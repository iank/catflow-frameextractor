import psycopg2


def vector_db_connect(pgconfig):
    conn = psycopg2.connect(
        host=pgconfig["host"],
        dbname=pgconfig["dbname"],
        user=pgconfig["username"],
        password=pgconfig["password"],
        port=pgconfig["port"],
    )
    return conn


def check_db(pgconfig):
    conn = None
    retval = None
    try:
        conn = vector_db_connect(pgconfig)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        retval = True
    except psycopg2.Error:
        retval = False
    finally:
        if conn is not None:
            conn.close()

    return retval
