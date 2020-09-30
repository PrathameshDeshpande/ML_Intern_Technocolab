import streamlit as st
import pickle
import numpy as np
import pandas as pd


model=pickle.load(open('model.pkl','rb'))

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


_max_width_()


def predict_prob(LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6):
    input_data=np.array([[LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]]).astype(np.float64)
    prediction=model.predict_proba(input_data)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)


def main():
    st.markdown("<h1 style='text-align: center; color: black;'>📈💰Credit Card Defaulter Prediction💰📉</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Made by Prathamesh with ❤️</h1>", unsafe_allow_html=True)
    html_temp="""
    <div style="background-color:#B22222 ;padding:20px">
    <h2 style="color:white;text-align:center;">Credit Card Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    CHOICES_EDUCATE = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
    def format_func_edu(option):
        return CHOICES_EDUCATE[option]


    CHOICES_MARRIAGE = {1: "Married", 2: "Single", 3: "Others"}
    def format_func_marry(option):
        return CHOICES_MARRIAGE[option]


    CHOICES_REPAY = {-2:"Account started that month with zero balance, and never used any credit",-1:"Account had a balance that was paid in full",0:"At least minimum payment was made, but the entire balance wasn't paid",1:"Payment delay for 1 month",2:"Payment delay for 2 month",3:"Payment delay for 3 month",4:"Payment delay for 4 month",5:"Payment delay for 5 month",6:"Payment delay for 6 month",7:"Payment delay for 7 month",8:"Payment delay for 8 month"}
    def format_func_repay(option):
        return CHOICES_REPAY[option]
    st.write("### PLEASE ENTER YOUR FULL NAME👇")
    NAME = st.text_input("FULL NAME :","")
    st.write("### PLEASE ENTER YOUR BALANCE LIMIT👇")
    LIMIT_BAL = st.text_input("LIMIT_BAL","")
    st.write("### PLEASE SELECT YOR EDUCATION👇")
    EDUCATION = st.selectbox("Select your Education Level",options=list(CHOICES_EDUCATE.keys()), format_func=format_func_edu)
    st.write("### PLEASE SELECT YOUR MARITIAL STATUS👇")
    MARRIAGE = st.selectbox("Marital Status",options=list(CHOICES_MARRIAGE.keys()), format_func=format_func_marry)
    st.write("### PLEASE ENTER YOUR AGE👇")
    AGE = st.text_input("Age(in Years)","")
    st.write("### PLEASE SELECT YOUR REPAYMENT STATUS👇")
    PAY_1 = st.selectbox("Repayment Status",options=list(CHOICES_REPAY.keys()), format_func=format_func_repay)
    st.write("## AMOUNT RECIVED IN LAST SIX MONTHS👇")
    BILL_AMT1 = st.text_input("Last month Bill Amount","")
    BILL_AMT2 = st.text_input("2nd Last month Bill Amount -","")
    BILL_AMT3 = st.text_input("3rd Last month Bill Amount -","")
    BILL_AMT4 = st.text_input("4th Last month Bill Amount -","")
    BILL_AMT5 = st.text_input("5th Last month Bill Amount -","")
    BILL_AMT6 = st.text_input("6Th Last month Bill Amount -","")
    st.write("## AMOUNT PAID IN LAST SIX MONTHS👇")
    PAY_AMT1 = st.text_input("Amount paid in Last Month -","")
    PAY_AMT2 = st.text_input("Amount paid in 2nd Last Month -","")
    PAY_AMT3 = st.text_input("Amount paid in 3rd Last Month -","")
    PAY_AMT4 = st.text_input("Amount paid in 4th Last Month -","")
    PAY_AMT5 = st.text_input("Amount paid in 5th Last Month -","")
    PAY_AMT6 = st.text_input("Amount paid in 6th Last Month -","")
    safe_html="""  
      <div style="background-color:#32CD32;padding:20px >
       <h2 style="color:rede;text-align:center;"> This account will not be defaulted with a probablity of</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#00ffff;padding:20px >
       <h2 style="color:black ;text-align:center;"> This account will be defaulted with a probablity of</h2>
       </div>
    """
    if st.button("Prediction"):
        output=predict_prob(LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6)

        if output*100 <50:
            st.markdown(danger_html,unsafe_allow_html=True)
            st.success(output*100)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)
            st.success(output*100) 
if __name__=='__main__':
    main()
