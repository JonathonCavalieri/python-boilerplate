import sqlalchemy as sa
import pandas as pd
import urllib


#################################
class SQLUpsert(object):
    """"""

    """
    example usage
    driver = 'ODBC Driver 17 for SQL Server'
    host = 'localhost'
    database = 'Sandbox'

    df_update = pd.DataFrame([{'test1': 'xxx3','test2': '2'},{'test1': 'xxx2','test2': '2'}])
    targetTable = 'test_table'
    idColumns = ['test1']
    deleteColumns=['test1']


    a_SQLUpsert = SQLUpsert(driver, host, database)

    a_SQLUpsert.SQLOutputUpsert(df_update, targetTable, idColumns, deleteColumns)
    a_SQLUpsert.messages
    """

    def __init__(self, driver, server, database):
        connection_string = (
            f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE={database};"
            r"Trusted_Connection=Yes;"
        )
        self.messages = []
        sqlalchemy_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(
            connection_string
        )
        self.engine = sa.create_engine(sqlalchemy_url, fast_executemany=True)
        self.SQLConnectionReady = True

    def AddMessage(self, type, message):
        self.messages.append({"type": type, "message": message})

    def SQLCheckColumnsMatching(
        self, df_update, targetTable, idColumns, print_results=True
    ):
        self.AddMessage(
            "info", f"Checking columns for {targetTable} table compared to Datafrane"
        )
        if not self.SQLConnectionReady:
            self.AddMessage("error", "SQL Server connection not setup")
            return False

        sourceColumnNames = set(df_update.columns)
        with self.engine.begin() as conn:
            # Get columns in table trying to write to
            SQLREQUEST = conn.execute(
                sa.text("SELECT TOP 0 * from {}".format(targetTable))
            )
            targetColumnNames = {col[0] for col in SQLREQUEST.cursor.description}

            # find common columns
            columns_common = targetColumnNames.copy()
            columns_common = list(columns_common.intersection(sourceColumnNames))

            # find missing columns in dataframe IE new columns in data that dont exist in DB
            columns_new = targetColumnNames.copy()
            columns_new.difference_update(sourceColumnNames)

            # Find columns missing in dataframe that exist in DB
            columns_missing = sourceColumnNames.copy()
            columns_missing.difference_update(targetColumnNames)

            # check if id columns exist
            idCheck = {x: x in columns_common for x in idColumns}

            if print_results:
                self.AddMessage("info", f"New Columns : {columns_new}")
                self.AddMessage("info", f"Missing Columns : {columns_missing}")
                self.AddMessage("info", f"Common Columns : {columns_common}")
                self.AddMessage("info", f"Id Checks : {idCheck}")

            return columns_new, columns_missing, columns_common, idCheck

    def SQLOutputUpsert(self, df_update, targetTable, idColumns, deleteColumns=None):
        if not self.SQLConnectionReady:
            self.AddMessage("error", "SQL Server connection not setup")
            return False

        sourceColumnNames = df_update.columns
        with self.engine.begin() as conn:
            # Get columns in table trying to write to
            SQLREQUEST = conn.execute(
                sa.text("SELECT TOP 0 * from {}".format(targetTable))
            )
            targetColumnNames = [col[0] for col in SQLREQUEST.cursor.description]

            # find common columns between target table and source data and check id fields in
            columns = list(set(targetColumnNames).intersection(sourceColumnNames))
            if not columns:
                self.AddMessage("error", "No common columns")
                return False

            # check if id columns exist
            idCheck = [x in columns for x in idColumns]
            if False in idCheck:
                self.AddMessage("error", "ID column not in Target/Source")
                return False
            # probably should put in error check to notify which columns mismatch

            if deleteColumns:
                print(deleteColumns)

                deleteCheck = [x in columns + ["All"] for x in deleteColumns]
                if False in deleteCheck:
                    self.AddMessage("error", "Delete columns not in Target/Source")
                    return False

            # Only keep columns that exist in target table
            df_update = df_update[columns].copy()
            try:
                # Load Data into temporary table
                df_update.to_sql(
                    "temp_load_data_table", conn, index=False, if_exists="replace"
                )

                # Create Update query based off columns
                updateQuery = "UPDATE t \nSET"
                # for each column ad a line to query to set it
                for col in columns:
                    updateQuery = updateQuery + "\n\tt.{0} = s.{0},".format(col)
                # Get rid of last comma and add on tabe/join section
                updateQuery = updateQuery[
                    :-1
                ] + "\nFROM {} t \nJOIN temp_load_data_table s ON ".format(targetTable)
                # create join conditions for each of the ID fields
                for col in idColumns:
                    updateQuery = updateQuery + "\n\tt.{0} = s.{0} and".format(col)
                updateQuery = updateQuery[:-3]  # Get rid of AND for last one

                # create Insert query based off columns
                # Get Columns to be updated
                insertColumns = ""
                for col in columns:
                    insertColumns = insertColumns + "{},".format(col)
                insertColumns = insertColumns[:-1]  # Remove the last comma
                # Create Query
                insertQuery = (
                    "INSERT INTO {0} ({1}) \n"
                    "SELECT "
                    "\n\t{1} "
                    "\nFROM temp_load_data_table s "
                    "\nWHERE NOT EXISTS "
                    "\n\t(SELECT NULL"
                    "\n\t FROM {0} t"
                    "\n\tWhere ".format(targetTable, insertColumns)
                )
                # add id conditions
                for col in idColumns:
                    insertQuery = insertQuery + "\n\t\tt.{0} = s.{0} and".format(col)
                insertQuery = insertQuery[:-3] + ")"  # Get rid of AND for last one

                if deleteColumns == ["All"]:
                    deleteQuery = (
                        f"TRUNCATE TABLE {targetTable}; \n"
                        "SELECT @rows_deleted = @@ROWCOUNT; \n"
                    )

                elif deleteColumns:
                    deleteCondition = ""
                    for col in deleteColumns:
                        deleteCondition = (
                            deleteCondition + f"\n\t{targetTable}.{col} = s.{col} and"
                        )
                    deleteCondition = deleteCondition[:-3]

                    deleteQuery = (
                        f"DELETE FROM {targetTable} \n"
                        "WHERE EXISTS (SELECT 1 \n"
                        "FROM temp_load_data_table s \n"
                        f"WHERE {deleteCondition}); \n"
                        "SELECT @rows_deleted = @@ROWCOUNT; \n"
                    )

                else:
                    deleteQuery = ""

                mainQuery = (
                    "SET NOCOUNT ON;\n"
                    "DECLARE @rows_updated INT = 0;\n"
                    "DECLARE @rows_inserted INT = 0;\n"
                    "DECLARE @rows_deleted INT = 0;\n"
                    "\n"
                    f"{deleteQuery}\n"
                    f"{updateQuery};\n"
                    "SELECT @rows_updated = @@ROWCOUNT;\n"
                    " \n"
                    f"{insertQuery};\n"
                    "SELECT @rows_inserted = @@ROWCOUNT;\n"
                    "\n"
                    "SELECT @rows_updated AS rows_updated, @rows_inserted AS rows_inserted, @rows_deleted AS rows_deleted;\n"
                )

                # Run Query
                result = conn.execute(sa.text(mainQuery)).fetchone()
                self.AddMessage(
                    "result",
                    f"{result[0]} row(s) updated, {result[1]} row(s) inserted, {result[2]} row(s) deleted to {targetTable} table",
                )
            except Exception as err:
                self.AddMessage("error", err)
            finally:
                # drop the temporary table
                # conn.execute(sa.text("DROP TABLE IF EXISTS temp_load_data_table"))
                print("test")

            return True
