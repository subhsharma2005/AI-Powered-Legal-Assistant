import psycopg2
from psycopg2 import OperationalError

DB_CONNECTION_STRING = "postgresql://postgres.btetznmuyvxgulfrlzdc:FYNBFEe4TDEx6Vfn@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres"

def test_db_connection():
    connection = None
    try:
        # Attempt to connect
        print("Connecting to Supabase PostgreSQL...")
        connection = psycopg2.connect(DB_CONNECTION_STRING)
        
        # Create a cursor to perform a simple operation
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        print(" Connection successful!")
        print(f"PostgreSQL version: {db_version[0]}")
        
    except OperationalError as e:
        print(f" The error '{e}' occurred")
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed.")

if __name__ == "__main__":
    test_db_connection()