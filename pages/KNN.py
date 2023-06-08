import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components
import time
import numpy as np
from json import load
import pandas as pd
import pickle
import joblib
from sys import path
path.append("MyMethods")

with open('vars.json') as f:
    glob_vars = load(f)
df = pd.read_csv(glob_vars['classification_data'])
with open(glob_vars['KNN'], 'rb') as handle:
    knn = pickle.load(handle)
with open(glob_vars['Scaler'], 'rb') as handle:
    scaler = joblib.load(handle)

st.set_page_config(page_title="Plotting Demo", page_icon="📈")
st.header('KNN')
tab1, tab2, tab3 = st.tabs(["Ограниченное предсказание", "Не ограниченное предсказание", "Множественное предсказание"])
modal_one = Modal("Результат:", key='streamlit-modal-default')
modal_multiply = Modal("Результаты:", key='streamlit-modal-default1')

with tab1:
    st.session_state['params_save'] = dict()
    for c in df.drop(glob_vars['predict_attribute'], axis=1).columns:
        st.session_state['params_save'][c] = st.slider(c, float(df[c].min())*glob_vars['authenticity'], 
                                                       float(df[c].max())*glob_vars['authenticity'], 0.01)
    if st.button('Безопасное предсказание'):
        st.session_state['res'] = knn.predict(np.array([list(st.session_state['params_save'].values())]))
        modal_one.open()


with tab2:
    st.session_state['params_unsave'] = dict()
    for c in df.drop(glob_vars['predict_attribute'], axis=1).columns:
        st.session_state['params_unsave'][c] = st.number_input(c)
    if st.button('Неограниченное предсказание'):
        st.session_state['res'] = knn.predict(np.array([list(st.session_state['params_unsave'].values())]))
        modal_one.open()

with tab3:
    uploaded_file = st.file_uploader("Choose a file")
    findout = st.button("Множественное предсказание")
    if findout and uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        dataframe = pd.read_csv(uploaded_file, sep=',', index_col=None)
        st.session_state['input_df'] = dataframe
        data_cleared = dataframe[[c for c in dataframe.columns if c in df.columns.to_list()]].dropna()
        print(data_cleared)
        data_scaled = scaler.transform(data_cleared)
        data_scaled = pd.DataFrame(data_scaled, columns=data_cleared.columns)
        st.session_state['res'] = knn.predict(data_scaled.to_numpy())
        modal_multiply.open()


if modal_one.is_open():
    with modal_one.container():
        html_string = glob_vars['predict_results_decorated'][str(int(st.session_state['res'][0]))] + '''
        <script language="javascript">
          document.querySelector("h1").style.fontFamily = "Impact,Charcoal,sans-serif";
        </script>
        '''
        components.html(html_string)


if modal_multiply.is_open():
    with modal_multiply.container():
        st.session_state['input_df'][glob_vars['predict_attribute']] = [glob_vars['predict_results'][str(int(i))] for i in st.session_state['res']]
        st.dataframe(st.session_state['input_df'])
        st.download_button(
            label="Download data as CSV",
            data=pd.DataFrame(data=st.session_state['res'], columns=['hazardous']).to_csv(index=False).encode('utf-8'),
            file_name='predict.csv',
            mime='text/csv',
        )

