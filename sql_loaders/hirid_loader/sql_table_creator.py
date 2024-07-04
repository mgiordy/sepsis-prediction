columns = []
with open("./interesting_variables.csv", "r") as f:
    columns = f.readlines()

columns_sql = {}
last_col = ""
col_rep = 0
for col in columns:
    # Removing new line character
    col = col.strip()
    col = col.split(",")
    code = col[0]
    name = col[1]
    # Truncating name to 62, postgres can't handle more than 64 char
    name = name[0:62]
    # Columns might be repeated twice
    if name == last_col:
        name = name + "_" + str(col_rep)
        col_rep += 1
    else:
        last_col = name
        col_rep = 0
    # Removing unsupported character
    name = name.replace(' ', '_').replace('-', '_').replace('%', '').replace('[', '').replace(']', '').replace('/', '').replace('.', '').replace('#', '').replace('+', '').replace(';', '')
    # Saving column and code
    columns_sql[name] = code

sql_query = "DROP TABLE IF EXISTS combined_pharma_table;\n"
sql_query += "CREATE TABLE combined_pharma_table(\n\tpatientid integer,\n\tgivenat timestamp without time zone,"
for name in columns_sql.keys():
    sql_query += "\n\t" + name + " real,"
sql_query = sql_query[:-1] # remove last comma
sql_query += ");\n\n\n"


with open("./table_builder.sql", "w") as f:
    f.write(sql_query)
    for col in columns_sql:
        f.write("INSERT INTO combined_pharma_table (patientid, givenat, " + col + ") SELECT patientid, givenat, givendose FROM pharma_table where pharmaid in (" + columns_sql[col] + ");\n")
    f.write("\nCREATE INDEX combined_pharma_table_index ON combined_pharma_table (patientid)")
    # f.write("\nSELECT DISTINCT patientid INTO  from patientid_table")