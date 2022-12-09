import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fbprophet import Prophet
import plotly.graph_objs as go

data = pd.read_csv('crime_merged.csv', index_col = [0]) 

df = data[['ID','Date','Primary Type','Location Description','Arrest','Domestic','Zipcode']]

df.Date = pd.to_datetime(df.Date,format = '%m/%d/%Y %I:%M:%S %p')
df.index = pd.DatetimeIndex(df.Date)

primary_types= ['All'] + list(set(df["Primary Type"]))
zipcodes = ['All'] + list(set(df["Zipcode"])) 

with st.form('timeseries'):

    incident_option = st.selectbox('Incident Types', primary_types)
    zipcode_option = st.selectbox('Zipcode', zipcodes)
    submitted = st.form_submit_button("Submit")
    if submitted:
        df_new = df.copy()

        if incident_option != 'All':
            df_new = df_new[df_new['Primary Type'] == incident_option]

        if zipcode_option != 'All':
            df_new = df_new[df_new['Zipcode'] == int(zipcode_option)]

        df_prophet = pd.DataFrame(df_new.resample('M').size().reset_index())

        df_prophet.columns = ['Date','Crime Count']
        df_prophet = df_prophet.rename(columns={'Date':'ds','Crime Count':'y'})

        m = Prophet()
        m.fit(df_prophet)

        pred = m.make_future_dataframe(periods=12,freq='M')
        forecast = m.predict(pred)


        # Lowercase the column names
        # df.columns = [df_prophet.columns]
        # Determine which is Y axis
        y_col = 'y'
        # Determine which is X axis
        ds_col = 'ds'
        # Determine what the aggregation is
        # agg = aggregation(ds_col)
        # Create the plotly figure
        yhat = go.Scatter(
          x = forecast['ds'],
          y = forecast['yhat'],
          mode = 'lines',
          marker = {
            'color': '#3bbed7'
          },
          line = {
            'width': 3
          },
          name = 'Forecast',
        )

        yhat_lower = go.Scatter(
          x = forecast['ds'],
          y = forecast['yhat_lower'],
          marker = {
            'color': 'rgba(0,0,0,0)'
          },
          showlegend = False,
          hoverinfo = 'none',
        )

        yhat_upper = go.Scatter(
          x = forecast['ds'],
          y = forecast['yhat_upper'],
          fill='tonexty',
          fillcolor = 'rgba(231, 234, 241,.75)',
          name = 'Confidence',
          hoverinfo = 'none',
          mode = 'none'
        )

        actual = go.Scatter(
          x = df_prophet['ds'],
          y = df_prophet['y'],
          mode = 'markers',
          marker = {
            'color': '#fffaef',
            'size': 4,
            'line': {
              'color': '#000000',
              'width': .75
            }
          },
          name = 'Actual'
        )

        layout = go.Layout(
          yaxis = {
            'title': 'Crime Rate',
            'tickformat': format(y_col),
            'hoverformat': format(y_col)
          },
          hovermode = 'x',
          xaxis = {
            'title': 'date'
          },
          margin = {
            't': 20,
            'b': 50,
            'l': 60,
            'r': 10
          },
          legend = {
            'bgcolor': 'rgba(0,0,0,0)'
          }
        )
        data = [yhat_lower, yhat_upper, yhat, actual]

        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig, use_container_width = True)

