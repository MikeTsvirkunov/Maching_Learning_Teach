import streamlit as st
import numpy as np
from json import load
import pandas as pd
import matplotlib.pyplot as plt

with open('vars.json') as f:
    glob_vars = load(f)
df = pd.read_csv(glob_vars['classification_data_preprocessed'])


tab1, tab2 = st.tabs(["Зависимости", "Корреляция"])

with tab1:
    st.markdown('Asd')


with tab2:
    st.
