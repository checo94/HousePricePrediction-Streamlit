import streamlit as st
import pandas as pd
# import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# House Price Prediction App

This app predicts the **House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
data = pd.read_csv('train_houses.csv')
data.fillna(data.mean(), inplace=True)
target = data.SalePrice
# X = pd.DataFrame(data, columns=["LotArea", "YearBuilt", "FirstFlrSF", "SecondndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"])
# Y = pd.DataFrame(target, columns=["SalePrice"])

features = ['LotArea', 'YearBuilt', 'FirstFlrSF', 'SecondFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']


X = data[features]
Y = target

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    LotArea = st.sidebar.slider('LotArea', 1300, 215245)
    YearBuilt = st.sidebar.slider('YearBuilt', X.YearBuilt.min(), X.YearBuilt.max())
    FirstFlrSF = st.sidebar.slider('FirstFlrSF', X.FirstFlrSF.min(), X.FirstFlrSF.max())
    SecondFlrSF = st.sidebar.slider('SecondFlrSF', X.SecondFlrSF.min(), X.SecondFlrSF.max())
    FullBath = st.sidebar.slider('FullBath', X.FullBath.min(), X.FullBath.max())
    BedroomAbvGr = st.sidebar.slider('BedroomAbvGr', X.BedroomAbvGr.min(), X.BedroomAbvGr.max())
    TotRmsAbvGrd = st.sidebar.slider('TotRmsAbvGrd', X.TotRmsAbvGrd.min(), X.TotRmsAbvGrd.max())
    data = {'LotArea': LotArea,
            'YearBuilt': YearBuilt,
            'FirstFlrSF': FirstFlrSF,
            'SecondFlrSF': SecondFlrSF,
            'FullBath': FullBath,
            'BedroomAbvGr': BedroomAbvGr,
            'TotRmsAbvGrd': TotRmsAbvGrd}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=1)
model = RandomForestRegressor(random_state=1)
model.fit(X, Y)
# Apply Model to Make Prediction
model_val = model.predict(val_X)
model_mae = mean_absolute_error(model_val, val_y)
print(model_mae)
prediction = model.predict(df)

st.header('Prediction of Sale Price')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
