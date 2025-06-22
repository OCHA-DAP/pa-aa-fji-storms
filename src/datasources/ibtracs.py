import ocha_stratus as stratus
import pandas as pd


def load_storms():
    query = """
    SELECT * FROM storms.storms
    """
    df = pd.read_sql(query, stratus.get_engine("dev"))
    return df
