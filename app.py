
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

st.title("Basic ML Model with CSV Upload")

uploaded_train = st.file_uploader("Upload Training CSV", type="csv")
uploaded_test = st.file_uploader("Upload Testing CSV", type="csv")

if uploaded_train and uploaded_test:
    dftrain = pd.read_csv(uploaded_train)
    dftest = pd.read_csv(uploaded_test)

    # Preprocessing
    dftrain = dftrain.astype('category')
    dftest = dftest.astype('category')

    for df in [dftrain, dftest]:
        for col in df.select_dtypes(['category']).columns:
            df[col] = df[col].cat.codes

    x_train = dftrain.drop('target', axis=1)
    y_train = dftrain['target']
    x_test = dftest.drop('target', axis=1)
    y_test = dftest['target']

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.success(f"Accuracy: {acc:.4f}")
    st.success(f"F1 Score: {f1:.4f}")
else:
    st.warning("Please upload both training and testing CSV files.")
