import sqlite3


DATABASE = "surveillance.db"


def get_connection():
    return sqlite3.connect(DATABASE)


def create_tables():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            activity TEXT,
            confidence REAL,
            anomaly_score REAL,
            status TEXT,
            evidence_path TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            activity TEXT,
            person_count INTEGER,
            confidence REAL
        )
    """)

    connection.commit()
    connection.close()