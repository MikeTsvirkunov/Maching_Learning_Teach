import streamlit as st
import time
import numpy as np
from json import load
import pandas as pd

with open('vars.json') as f:
    glob_vars = load(f)
df = pd.read_csv(glob_vars['classification_data'])

st.set_page_config(page_title="Plotting Demo", page_icon="📈")
st.header('Данные')
st.markdown(f'Число строк: {df.shape[0]}')
st.markdown(f'Число параметров: {df.shape[1]}')
st.markdown(f'Число строк с NaN\'ами: {df[df.isna().any(axis=1)].shape[0]}')
st.markdown(f'Число NaN\'ов всего: {df.isna().sum().sum()}')
st.subheader('Основные сведения')
st.table(df.drop('id', axis=1).describe())

