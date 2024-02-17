import pymysql
from sqlalchemy import Column, ForeignKey, Integer, String, Date, DateTime, text, create_engine, MetaData
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy import text


class MySQLClient:
    def __init__(self, host, username, password, database):
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.engine = None
        self.metadata = MetaData()

    def connect(self):
        """Establishes a connection to the MySQL database."""
        try:
            # Create an engine to connect to the MySQL database
            # print(f"mysql+pymysql://{self.username}:{self.password}@{self.host}/{self.database}?charset=utf8mb4")
            self.engine = create_engine(f"mysql+mysqldb://{self.username}:{self.password}@{self.host}/{self.database}")

            print("Connected to MySQL database")
        except Exception as e:
            print(f"Error connecting to MySQL database: {e}")

    def disconnect(self):
        """Closes the connection to the MySQL database."""
        if self.engine:
            self.engine.dispose()
            print("Disconnected from MySQL database")

    def execute_query(self, query):
        """Executes an SQL query and returns the result."""
        if not self.engine:
            print("No connection to MySQL database")
            return None
        
        with self.engine.connect() as connection:
            try:
                result = connection.execute(text(query))
                return result.fetchall()
            except Exception as e:
                print(f"Error executing query: {e}")
                return None

def make_client(host, usernamae, password, database):
    # ??
    pymysql.install_as_MySQLdb()

    client = MySQLClient(host=host, username=usernamae, password=password, database=database)
    client.connect()

    return client
