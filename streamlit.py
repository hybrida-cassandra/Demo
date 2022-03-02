import streamlit as st
import pandas as pd
import seaborn as sns
from joblib import load
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance, plot_partial_dependence
from PIL import Image

st.set_page_config(page_title='Cassandra - Scenario Simulator')

@st.cache(allow_output_mutation=True)
def load_model():
    print('Loading model...')

    return load('neural_spedire.joblib')

@st.cache()
def load_data():
    print('Loading data...')

    df = pd.read_csv('spedire.csv')

    return df

model = load_model()
df = load_data()

df=df.fillna( method='ffill')

st.title('Cassandra - Scenario Simulator')

st.sidebar.image(Image.open(f"logo.png"), width=200)
st.sidebar.title('1. Seasonal Variables')

user_input= {}
categorical = ['day_week', 'month','year']

for feat in categorical:
    unique_values = df[feat].unique()
    unique_values.sort()

    if feat == 'day_week':
        label = "Day of the week"
        display = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

        user_input[feat] = st.sidebar.selectbox(label, unique_values, format_func=lambda x: display[x])
    elif feat == 'month':
        label = "Month"
        display = ("Si", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "Dicember")

        user_input[feat] = st.sidebar.selectbox(label, unique_values, format_func=lambda x: display[x])
    elif feat == 'year':
        label = "Year"
        user_input[feat] = st.sidebar.selectbox(label, unique_values)

st.sidebar.title('2. Paid Media Variables')

paid_media = ['fb_spent', 'google_spent', 'bing_spent']

for feat in paid_media:
    v_min = float(df[feat].min())
    v_max =float(df[feat].max())

    if feat == "fb_spent":
        label = "Facebook Ads Spend"
    elif feat == "google_spent":
        label = "Google Ads Spend"
    elif feat == "bing_spent":
        label = "Bing Ads Spend"

    user_input[feat]=st.sidebar.slider(
        label,
        min_value= v_min,
        max_value=v_max,
        value= (v_min+v_max)/2
    )

st.sidebar.title('3. Organic Media Variables')

organic_media = ['google_organico', 'bing_organico', 'referral', 'email_sessioni', 'cambio_eur_dollaro']

for feat in organic_media:
    v_min = float(df[feat].min())
    v_max = float(df[feat].max())

    if feat == "google_organico":
        label = "Google Organic Sessions"
    elif feat == "bing_organico":
        label = "Bing Organic Sessions"
    elif feat == "referral":
        label = "Referral Sessions"
    elif feat == "email_sessioni":
        label = "Email Sessions"
    elif feat == "cambio_eur_dollaro":
        label = "EUR/USD Exchange"

    user_input[feat] = st.sidebar.slider(
        label,
        min_value=v_min,
        max_value=v_max,
        value=(v_min + v_max) / 2
    )

X = pd.DataFrame([user_input])

z=(X['fb_spent']+X['google_spent']+X['bing_spent'])
cpa= z/model.predict(X)

prediction = model.predict(X)

st.title('Predicted sales')

fig= go.Figure(
    go.Indicator(
        mode= 'gauge+number',
        value=prediction[0]
    )
)
st.plotly_chart(fig)

aov = 17.99
revenue = prediction * aov
profit = revenue*0.6-z

st.subheader("Revenue: " + str(round(revenue[0], 2)) + " $")
st.subheader("Profit: " + str(round(profit[0], 2)) + " $")
st.subheader("CPA: " + str(round(cpa[0], 2)) + " $")



