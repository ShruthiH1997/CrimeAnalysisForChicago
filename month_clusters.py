import streamlit as st
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as offlin
import numpy as np
from plotly.graph_objs import Scatter, Figure, Layout

st.set_page_config(
        page_title="Chicago Crime",
        initial_sidebar_state="expanded"
    )

st.write('Select a date range between "2015-01-01" and ""2022-09-23"')
#st.write(start_date, end_date)

df = pd.read_csv('crime_merged.csv', index_col = [0])

df_ = df[[ 'Latitude', 'Longitude', 'ID', 'Case Number' , 'Date', 'Block', 'IUCR', 'Primary Type', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'FBI Code', 'Zipcode', 'Median_Income', 'Mean_Income']].rename(columns = {'Date' : 'DateTime'})
df_['DateTime'] = pd.to_datetime(df_['DateTime'])
df_['Date'] = df_['DateTime'].dt.date
df_ = df_.rename(columns =  {col : col.replace(' ', '_') for col in df.columns})
df_['day'] = df_['DateTime'].dt.day
df_['month'] = df_['DateTime'].dt.month

community_data = pd.read_csv('del.csv').dropna(axis = 1)
features = ['GEOID', 'EMP', 'UNEMP' ]

community_data_clean = community_data[features].rename(columns = {'GEOID': 'Community_Area'}).set_index('Community_Area')
community_data_clean['unEmpRate'] = community_data_clean['UNEMP'] / (community_data_clean['EMP'] + community_data_clean['UNEMP'] )
community_data_clean = community_data_clean.drop(['EMP', 'UNEMP'], axis = 1)

start_date, end_date = st.date_input('start date  - end date :', [])


df_date = df_.copy()
df_date = df_date[df_date.Date.isin(pd.date_range(start_date, end_date))]
clustering_df = df_date.copy()[['Community_Area', 'ID', 'Arrest', 'Mean_Income']]
crime_rate = clustering_df.groupby(['Community_Area']).agg({ 'ID' : 'count', 'Mean_Income' : 'mean', 'Arrest' : 'sum'}).rename(columns = {'ID' : 'Total_incidents'})
crime_rate['arrest_rate'] = crime_rate['Arrest']/crime_rate['Total_incidents']
clustering_df_1 = crime_rate.drop(columns = [ 'Arrest', 'Total_incidents']).merge(community_data_clean, left_index=True, right_index=True, how='inner')

X = np.array(clustering_df_1)
kmeans = KMeans(n_clusters=3, init = "k-means++")
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
pred = y_kmeans
kmeans = clustering_df_1.copy()
kmeans['Label'] = pred
#kmeans = kmeans.reset_index()
print('silhouette_score: ',silhouette_score(X, pred, metric='euclidean', n_jobs=-1))
print('calinski_harabasz_score: ',calinski_harabasz_score(X, pred))
print('davies_bouldin_score: ', davies_bouldin_score(X, pred))

PLOT = go.Figure()
for C in list(kmeans.Label.unique()):
    
    PLOT.add_trace(go.Scatter3d(x = kmeans[kmeans.Label == C]['unEmpRate'],
                                y = kmeans[kmeans.Label == C]['Mean_Income'],
                                z = kmeans[kmeans.Label == C]['arrest_rate'],
                                mode = 'markers', marker_size = 8, marker_line_width = 1,
                                name = 'Cluster ' + str(C)))
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                scene = dict(xaxis=dict(title = 'unEmpRate', titlefont_color = 'black'),
                                yaxis=dict(title = 'Mean_Income', titlefont_color = 'black'),
                                zaxis=dict(title = 'arrest_rate', titlefont_color = 'black')),
                font = dict(family = "Gilroy", color  = 'black', size = 12))

st.plotly_chart(PLOT, use_container_width=True)

Community_Area_mapping_kmeans = { row['Community_Area']: row['Label']  for i,row in kmeans.reset_index().iterrows()}
mapping_ = df_.groupby(['Community_Area'])['Latitude', 'Longitude'].agg('min').reset_index()
mapping_kmeans = mapping_.copy()
mapping_kmeans["Label"] = mapping_kmeans['Community_Area'].apply(lambda x: Community_Area_mapping_kmeans[x] if x in Community_Area_mapping_kmeans else None)
mapping_kmeans = mapping_kmeans.dropna()
colors_map = {0 : '#636EFA', 1: '#EF553B', 2: '#FFA15A'}
mapping_kmeans["color"] = mapping_kmeans['Label'].apply(lambda x: colors_map[x] )


#st.dataframe(mapping_kmeans)

fig = px.scatter_mapbox( mapping_kmeans , lat="Latitude", lon="Longitude",
                        color = 'Label', zoom=9, height=650)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


st.plotly_chart(fig)
st.dataframe(kmeans.groupby('Label').mean())