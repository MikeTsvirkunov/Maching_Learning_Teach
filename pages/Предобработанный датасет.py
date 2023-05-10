import streamlit as st
import time
import numpy as np
from json import load
import pandas as pd

with open('vars.json') as f:
    glob_vars = load(f)
df = pd.read_csv(glob_vars['classification_data_preprocessed'])

if 'count' not in st.session_state:
    st.session_state.count = 0
n_rows = glob_vars['num_of_rows_in_show_table']
st.set_page_config(page_title="Dataset preprocessed", page_icon="📈")
st.header('Датасет')
table = st.table(df[st.session_state.count:st.session_state.count + n_rows])


def increment_counter(increment_value):
    st.session_state.count += increment_value
    if st.session_state.count < 0:
        st.session_state.count = 0
    elif st.session_state.count - st.session_state.count % df.shape[0] > df.shape[0]:
        st.session_state.count -= increment_value

step_back, page_numeration, step_front = st.columns(3)
# step_next.button("Next")
# step_back.button("Prev")

if step_front.button("Next"):
    print(st.session_state.count)
    increment_counter(n_rows)
    n = st.session_state.count
    table.table(df[n:n+n_rows])

if step_back.button("Prev"):
    increment_counter(-1 * n_rows)
    n = st.session_state.count
    table.table(df[n:n+n_rows])

page_numeration.write(f'{(st.session_state.count+n_rows) // n_rows}/{df.shape[0] // n_rows + (df.shape[0] % n_rows > 0)}')
