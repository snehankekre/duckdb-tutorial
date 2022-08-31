import random

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from duckdb import DuckDBPyConnection
from sklearn import datasets, metrics


@st.experimental_singleton
def init_connection(database: str = "mydb.duckdb") -> DuckDBPyConnection:
    """
    Initialize and cache a connection to the database.
    """
    conn = duckdb.connect(database, read_only=False)
    return conn


def execute(query: str) -> DuckDBPyConnection:
    return conn.execute(query)


def run_query(query: str) -> pd.DataFrame:
    return conn.execute(query).fetch_df()


def show_image(index: int) -> None:
    fig, ax = plt.subplots()
    ax.matshow(
        datasets.load_digits().images[index],
        cmap=plt.cm.gray_r,
        interpolation="nearest",
    )
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)


@st.experimental_singleton
def load_data():
    return datasets.load_digits()


def submit() -> None:
    execute(
        f"INSERT INTO mytable VALUES ('{st.session_state.image}', '{st.session_state.label}')"
    )


st.header("Data labeling with DuckDB and Streamlit")

conn = init_connection()
table = execute("SHOW TABLES").fetchone()

if table is None or not "mytable" in table:
    create_table_query = "CREATE TABLE mytable (image varchar(80), label varchar(80));"
    execute(create_table_query)

digits = load_data()
index = random.randint(0, len(digits.data))

col1, col2 = st.columns(2)

with st.sidebar.form(key="form", clear_on_submit=True):
    show_image(index)

    name = st.text_input(
        "Image",
        value=datasets.load_digits().target[index],
        key="image",
        disabled=True,
        help="Filename of the image to label",
    )
    pet = st.text_input(
        "Label", key="label", help="What digit does this image represent?"
    )

    btn = st.form_submit_button("Submit", on_click=submit)

df = run_query("SELECT * from mytable")

col1.dataframe(df)

if len(df) > 0:
    col2.metric(
        "Agreement",
        value="{:.1f}%".format(metrics.accuracy_score(df.image, df.label) * 100),
    )
