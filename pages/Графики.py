import streamlit as st
import numpy as np
from json import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import plotly.figure_factory as ff
import matplotlib.gridspec as gridspec

with open('vars.json') as f:
    glob_vars = load(f)
df = pd.read_csv(glob_vars['classification_data_preprocessed'])
st.set_page_config(page_title="Plotting", page_icon="📈")

tab1, tab2 = st.tabs(["Зависимости", "Корреляция"])

with tab1:
    fig = plt.figure(tight_layout=True, facecolor='#0E1117')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    gs = gridspec.GridSpec(3, 1)
    
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.scatter(df['miss_distance'], df['relative_velocity'], df['est_diameter_max'], c=df['hazardous'])
    ax.view_init(30, 30)
    ax.set_facecolor("#0E1117")
    ax.set_xlabel('miss_distance', fontsize=10, rotation=150)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_ylabel('relative_velocity', fontsize=10)
    ax.set_zlabel('est_diameter_max', fontsize=10, rotation=60)

    ax = fig.add_subplot(gs[1, 0], projection='3d')
    ax.scatter(df['miss_distance'], df['relative_velocity'], df['absolute_magnitude'], c=df['hazardous'])
    ax.view_init(30, 30)
    ax.set_facecolor("#0E1117")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.set_xlabel('miss_distance', fontsize=10, rotation=150)
    ax.set_ylabel('relative_velocity', fontsize=10)
    ax.set_zlabel('absolute_magnitude', fontsize=10, rotation=60)
    
    # ax = fig.add_subplot(gs[2, 0], projection='3d')
    # ax.scatter(df['est_diameter_max'], df['relative_velocity'], df['absolute_magnitude'], c=df['hazardous'])
    # ax.view_init(30, 30)
    # ax.set_facecolor("#0E1117")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    # ax.zaxis.label.set_color('white')
    # ax.set_zlabel('est_diameter_max', fontsize=10, rotation=150)
    # ax.set_ylabel('relative_velocity', fontsize=10)
    # ax.set_zlabel('absolute_magnitude', fontsize=10, rotation=60)

    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(facecolor='#0E1117')
    ax = sns.heatmap(df.corr(), ax=ax, annot=True, cbar=False,
                     annot_kws={'fontsize': 12, 'color': 'w',
                           'rotation': 'vertical', 'verticalalignment': 'center'})
    for i, j in enumerate(ax.axes.get_yticklabels()):
        j.set_color("white")
    for i, j in enumerate(ax.axes.get_xticklabels()):
        j.set_color("white")

    st.write(fig)
