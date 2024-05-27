import os
import pickle
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping
from dotenv import load_dotenv
import psycopg

load_dotenv()

NUMERICAL = [
    'Size',
    'Weight',
    'Sweetness',
    'Crunchiness',
    'Juiciness',
    'Ripeness',
    'Acidity'
]

TARGET = 'Quality'

COL_MAPPING = ColumnMapping(
    prediction='prediction',
    numerical_features=NUMERICAL,
    target=TARGET
)

# host, port, user, password
CONNECT_STRING = f"host={os.getenv('POSTGRES_HOST')} port={os.getenv('POSTGRES_PORT')} user={os.getenv('POSTGRES_USER')} password={os.getenv('POSTGRES_PASSWORD')}"

def prep_db():
    create_table_query = """
    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics(
        id SERIAL PRIMARY KEY,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float
    );
    """

    with psycopg.connect(CONNECT_STRING, autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect(f"{CONNECT_STRING} dbname=test") as conn:
            conn.execute(create_table_query)

def prep_data():
    ref_data = pd.read_csv('data/processed_dataset.csv')
    with open('models/best_model/model.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
    
    raw_data = pd.read_csv('data/processed_dataset.csv')

    return ref_data, model, raw_data

def calculate_metrics(current_data, model, ref_data):
    current_data['prediction'] = model.predict(current_data[NUMERICAL].fillna(0))

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=COL_MAPPING
    )

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_cols = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_vals = result['metrics'][2]['result']['current']['share_of_missing_values']

    return prediction_drift, num_drifted_cols, share_missing_vals
    
def save_metrics_to_db(cursor, prediction_drift, num_drifted_cols, share_missing_vals):
    cursor.execute("""
    INSERT INTO metrics(
        prediction_drift,
        num_drifted_columns,
        share_missing_values
    )
    VALUES (%s, %s, %s);
    """, (prediction_drift, num_drifted_cols, share_missing_vals))

def monitor():
    prep_db()

    ref_data, model, raw_data = prep_data()

    with psycopg.connect(f"{CONNECT_STRING} dbname=test") as conn:
        with conn.cursor() as cursor:
            prediction_drift, num_drifted_cols, share_missing_vals = calculate_metrics(raw_data, model, ref_data)
            save_metrics_to_db(cursor, prediction_drift, num_drifted_cols, share_missing_vals)
            print("Data added")

if __name__ == '__main__':
    monitor()
