# import the necessary Core  libraries
import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
#%matplotlib inline
import plotly
import seaborn as sns
sns.set()
import pycountry
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py

from pywaffle import Waffle

py.init_notebook_mode(connected=True)
import folium 
from folium import plugins


plt.style.use("fivethirtyeight")# for pretty graphs

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


#Day by day data All countries 
#--------------------------------
# confirmed cases
df_confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
df_conf1 = df_confirmed.copy()
# recovered 
df_recovered = pd.read_csv('time_series_covid19_recovered_global.csv')
df_recov1 = df_recovered.copy()
# deaths 
df_deaths    = pd.read_csv('time_series_covid19_deaths_global.csv')
df_deaths1 = df_deaths.copy()


df_confirmed.head()

df_confirmed.columns

# Drop columns not needed for this Analysis  
df_confirmed.drop(['Province/State','Lat','Long'],axis=1,inplace=True)

# Rename to shorter column names 
df_confirmed.rename(columns= {'Country/Region':'Region'},inplace= True)



# Check DF after drop and rename 
df_confirmed.columns



# Create Listof countries to Analyse
plot_countries = ['India','Iran','Italy','Korea, South','Spain']

# subset by countries to plot 
df_conf_plot= df_confirmed[df_confirmed.Region.isin(plot_countries)]

# Transpose df
df_conf_plot_T = df_conf_plot.T

# Check Actual Col Names 
df_conf_plot_T.head()



# Rename Columns 
df_conf_plot_T.rename(columns= {131:'India',137:'Italy',133:'Iran',143:'Korea, South',201:'Spain'},inplace= True)

# drop first row 
df_conf_plot_T.drop(df_conf_plot_T.index[0],inplace=True)

# name Index 
df_conf_plot_T.index.name='DATE'

# Check 
df_conf_plot_T.head()



# Check data 
df_recovered.head()



# Drop columns not needed for this Analysis  
df_recovered.drop(['Province/State','Lat','Long'],axis=1,inplace=True)

# Rename to shorter column names 
df_recovered.rename(columns= {'Country/Region':'Region'},inplace= True)

# Check DF after drop and rename 
df_confirmed.columns



# subset by countries to plot 
df_recov_plot= df_recovered[df_recovered.Region.isin(plot_countries)]

# Transpose df
df_recov_plot_T = df_recov_plot.T

# Check Actual Col Names 
df_recov_plot_T.head()
# Rename Columns 
#df_recov_plot_T.rename(columns= {125:'India',131:'Italy',127:'Iran',137:'Korea, South',199:'Spain'},inplace= True)

# drop first row 
#df_recov_plot_T.drop(df_conf_plot_T.index[0],inplace=True)

# name Index 
#df_recov_plot_T.index.name='DATE'

# Check 
df_recov_plot_T.head()

df_temp  = df_recov_plot_T.iloc[1:]
del(df_recov_plot_T)
df_recov_plot_T = df_temp.copy()
df_recov_plot_T.head()

# Check data 
df_deaths.head()



# Get column names 
df_deaths.columns

# Drop columns not needed for this Analysis  
df_deaths.drop(['Province/State','Lat','Long'],axis=1,inplace=True)

# Rename to shorter column names 
df_deaths.rename(columns= {'Country/Region':'Region'},inplace= True)


# Check DF after drop and rename 
df_deaths.columns


# subset by countries to plot 
df_deaths_plot= df_deaths[df_deaths.Region.isin(plot_countries)]

# Transpose df
df_deaths_plot_T = df_deaths_plot.T

# Check Actual Col Names 
df_deaths_plot_T.head()

# Rename Columns 
df_deaths_plot_T.rename(columns= {131:'India',137:'Italy',133:'Iran',143:'Korea, South',201:'Spain'},inplace= True)

# drop first row 
df_deaths_plot_T.drop(df_deaths_plot_T.index[0],inplace=True)

# name Index 
df_deaths_plot_T.index.name='DATE'

# Check 
df_deaths_plot_T.head()
# copy DFs for plot
df_confirmed_All = df_conf_plot_T.copy()
df_recovered_All = df_recov_plot_T.copy()
df_deaths_All    = df_deaths_plot_T.copy()

XDate = df_confirmed_All.index
XDate_recov = df_recovered_All.index

#df_confirmed_All.plot();

#df_recovered_All.plot();
#df_deaths_plot_T.plot();

#df_recovered_All.plot();
df_recovered1 = df_recovered_All.copy()
df_recovered1.fillna(0,inplace=True)
#df_recovered1.plot();

#Cases in India vs Deaths
s1 = df_confirmed_All['Korea, South']
s2 = df_deaths_All['Korea, South']


fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.plot(XDate, s1, label='Actual Confirmed')
ax.plot(XDate, s2, label = 'Deaths')
ax.legend()
ax.tick_params(direction='out', length=10, width=10, colors='r')
ax.set_xlabel('Date',fontsize=25)
ax.set_ylabel('Cases count',fontsize=25)
ax.set_title('COVID 19 scene in India  as of 19th April 2020',fontsize=25)
fig.autofmt_xdate()

ax.grid(True)
fig.tight_layout()

plt.show()

# Extract date DF for All countries 
df_confirmed_1 = df_confirmed_All.copy()
# Create date column 
df_confirmed_All['DATE'] = df_confirmed_All.index
df_confirmed_All.head()
# Create date DF 
DT_df = df_confirmed_All[['DATE']]
DT_df  = DT_df.set_index('DATE')

# Extract Series for All countries
s_India   = df_confirmed_All['India']
s_Iran    = df_confirmed_All['Iran']
s_Italy   = df_confirmed_All['Italy']
s_Korea   = df_confirmed_All['Korea, South']
s_Spain   = df_confirmed_All['Spain']




# Extract date DF   for All countries
#--------------------------------------
n = 200  # cutoff case count for fitting model 
#------------------
India_DT_df      = DT_df[s_India >n]
Iran_DT_df       = DT_df[s_Iran >n]
Italy_DT_df      = DT_df[s_Italy >n]
Korea_DT_df      = DT_df[s_Korea >n]
Spain_DT_df      = DT_df[s_Spain >n]



# Create a Date column 
India_DT_df['Date']      = India_DT_df.index
Iran_DT_df['Date']       = Iran_DT_df.index
Italy_DT_df['Date']      = Italy_DT_df.index
Korea_DT_df['Date']      = Korea_DT_df.index
Spain_DT_df['Date']      = Spain_DT_df.index

# Get Series of All date DFs
India_DT_s  = India_DT_df['Date']      
Iran_DT_s   = Iran_DT_df['Date']     
Italy_DT_s  = Italy_DT_df['Date']      
Korea_DT_s  = Korea_DT_df['Date']      
Spain_DT_s  = Spain_DT_df['Date']

# subset each series for numbers > 100
n = 200
#-------------------------------------------------
# India
s_India_GE100 = s_India[s_India > n] 
s_India_GE100 = pd.to_numeric(s_India_GE100, errors='coerce').fillna(0, downcast='infer')
#----------------------------------------------
# Iran
s_Iran_GE100 = s_Iran[s_Iran > n] 
s_Iran_GE100 = pd.to_numeric(s_Iran_GE100, errors='coerce').fillna(0, downcast='infer')
#-------------------------------------------------
# Italy
s_Italy_GE100 = s_Italy[s_Italy > n] 
s_Italy_GE100 = pd.to_numeric(s_Italy_GE100, errors='coerce').fillna(0, downcast='infer')
#-------------------------------------------------
# Korea
s_Korea_GE100 = s_Korea[s_Korea > n] 
s_Korea_GE100 = pd.to_numeric(s_Korea_GE100, errors='coerce').fillna(0, downcast='infer')
#--------------------------------------------------
# Spain
s_Spain_GE100 = s_Spain[s_Spain > n] 
s_Spain_GE100 = pd.to_numeric(s_Spain_GE100, errors='coerce').fillna(0, downcast='infer')

# Model India 
import numpy as np
#--------------------------------------
# Y data 
Y = s_Korea_GE100
# X data 
X = np.arange(1,len(Y)+1)
Xdate = Korea_DT_s
# Fit 4th Degree polynomial capture coefficients 
Z = np.polyfit(X, Y, 4)
# Generate polynomial function with these coefficients 
P = np.poly1d(Z)
# Generate X data for forecast 
XP = np.arange(1,len(Y)+8)
# Generate forecast 
YP = P(XP)
# Fit Curve
Yfit = P(X)

import datetime
start = Xdate[0]
#start
end_dt = datetime.datetime.strptime(Xdate[len(Xdate)-1], "%m/%d/%y")

end_date = datetime.datetime.strptime(str(end_dt),'%Y-%m-%d %H:%M:%S').date()

end_forecast_dt= end_dt + datetime.timedelta(days=7)

end_forecast =  datetime.datetime.strptime(str(end_forecast_dt),'%Y-%m-%d %H:%M:%S').date()
end_forecast
#
mydates = pd.date_range(start, end_forecast).to_list()
mydates_df = pd.DataFrame(mydates,columns =['Date']) 
mydates_df  = mydates_df.set_index('Date')
mydates_df['Date'] = mydates_df.index
X_FC = mydates_df['Date']

fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.plot(X, Y, '--',label='Actual Confirmed')
ax.plot(XP, YP, 'o',label='Predicted Fit using 4th degree polynomial')
plt.title('COVID RISE IN India Current Vs Predictions till 26th April 2020')
ax.legend()
ax.set_ylim(0,25000)
ax.grid(True)
plt.show()

# Define new figure 
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(X_FC,YP,'--')
ax.tick_params(direction='out', length=10, width=10, colors='r')
ax.set_xlabel('Date',fontsize=25)
ax.set_ylabel('Predicted Cases',fontsize=25)
ax.set_ylim(0,30000)
ax.set_title('COVID 19 PREDICTION for India till 26th April 2020',fontsize=25)
fig.autofmt_xdate()

ax.grid(True)
fig.tight_layout()

plt.show()

# Create a dataframe from Predicted data 
dict1 = {'Date':X_FC,'Predicted_Cases':(YP)}
pred_df = pd.DataFrame.from_dict(dict1)
pred_df = pred_df[['Predicted_Cases']]
pred_df.Predicted_Cases = pred_df.Predicted_Cases.astype(int)
pred_df1 = pred_df.tail(n=8)
pred_df1.style.background_gradient()



