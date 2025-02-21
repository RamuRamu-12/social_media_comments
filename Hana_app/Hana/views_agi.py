
USER = 'test_owner'
PASSWORD = 'tcWI7unQ6REA'
HOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432'
DATABASE = 'test'

@csrf_exempt
def gen_txt_response(request):
    if request.method == "POST":
        file = request.FILES.get('file')
        table_name = file.name.split('.')[0]

        # Define upload directory
        upload_dir = os.path.join(os.getcwd(), 'upload')
        os.makedirs(upload_dir, exist_ok=True)

        # File path within upload directory
        file_path = os.path.join(upload_dir, file.name)
        with default_storage.open(file_path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        df= file_to_sql(file_path, table_name, USER, PASSWORD, HOST, DATABASE)
        print(df.head(5))

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])
        query = request.POST["query"]


def create_mysql_engine(user, password, host, db_name):
    from sqlalchemy import create_engine, text

    if db_name:
        connection_str = f'postgresql://{user}:{password}@{host}/{db_name}'
    else:
        connection_str = f'postgresql://{user}:{password}@{host}/'
    engine = create_engine(connection_str)
    return engine


def file_to_sql(file_path, table_name, user, password, host, db_name):
    import pandas as pd
    import os
    from sqlalchemy import create_engine

    # engine = create_engine(f"postgresql://{user}:{password}@{host}/{db_name}")
    engine = create_mysql_engine(user, password, host, db_name)

    if not table_name:
        table_name = os.path.splitext(os.path.basename(file_path))[0]

    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide an Excel (.xlsx) or CSV (.csv) file.")

    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    return f"Data from '{file_path}' stored in table '{table_name}'."


# Function to execute a custom SQL query and print results
def execute_query(query, user, password, host, db_name):
    from sqlalchemy import create_engine, text

    engine = create_mysql_engine(user, password, host, db_name)
    with engine.connect() as connection:
        try:
            result_set = connection.execute(text(query))
            output = []
            for row in result_set:
                print(row)
                output.append(str(row))
            return output
        except Exception as e:
            return str(e)


def get_metadata(host, user, password, db, tables):
    metadata = []
    for table in tables:
        table_info = {}
        query_columns = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}';
        """
        table_info['columns'] = execute_query(query_columns, user, password, host, db)

        query_sample = f'SELECT * FROM "{table}" LIMIT 5;'
        table_info['sample_rows'] = execute_query(query_sample, user, password, host, db)

        metadata.append({table: table_info})

    return metadata
