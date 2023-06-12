import psycopg2

def vector_db_connect(pgconfig):
    conn = psycopg2.connect(
        host=pgconfig['host'], dbname=pgconfig['dbname'], user=pgconfig['username'], password=pgconfig['password'], port=pgconfig['port']
    )
    return conn
