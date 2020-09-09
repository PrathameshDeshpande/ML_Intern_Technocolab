import pandas as pd
from xgboost import XGBClassifier,plot_importance
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve, auc,confusion_matrix
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
st.markdown("<h1 style='text-align: center; color: black;'>🌟XGBoost On Perkinson Data🌟</h1>", unsafe_allow_html=True)
def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
#_max_width_()
@st.cache(suppress_st_warning=True)
def main():
    get_data()
def get_data():
    df = pd.read_csv("C:/Users/91797/Desktop/ML_Intern_Technocolab/Mini_Project/data.csv")
    x = df.drop(["name","status"],axis=1)
    y = df["status"]
    y = pd.DataFrame(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    stratify=y,
                                                    test_size=0.20)
    st.write("# PERKINSON DATASET USED👇👇")
    st.dataframe(data=df,width=None, height=500)
    XGBC = XGBClassifier()
    XGBC.fit(x_train,y_train)
    y_pred = XGBC.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = float("%0.4f" % (accuracy))
    st.title("VISUALISATION👇👇")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
     )
    fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
     )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(autosize=False,
                  width=700, height=700,xaxis_title="TIME",
                  yaxis_title="PRICE",
                  legend_title="Legend Title",
                  font=dict(
                  family="Courier New, monospace",
                  size=18,
                  color="RebeccaPurple"),
                  margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)
    cm = confusion_matrix(y_test, y_pred)
    plt.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    plt.colorbar()
    #plt.plot(figsize=(5, 5))
    st.pyplot(use_container_width=True)
    st.write("# ACCURACY ACHIEVED👉",accuracy)
if __name__ == "__main__":
    main()