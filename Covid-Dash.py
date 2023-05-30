# import
import datetime
import pandas as pd
import numpy as np
#dash and plotly
from http import server
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
# Models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing
from pmdarima.arima import auto_arima

# Variables
colors = {
    'background': '#171E30',
    'text': '#D0D2D6',
    'cahrtBackground': '#283046',
    'chartPaper': '#283046',
    'clViolet': '#675ED6',
    'clBlue': '#06B7CF',
    'clOrange': '#DFB66F',
    'clGreen': '#65FA9E',
    'clRed': '#D84F51',
    'clh1b':'#283035'

}

mseList = np.array(0)
modelNames = np.array(0)

# read database
df = pd.read_csv("owid-covid-data.csv")
apple = pd.read_csv("applemobilitytrends.csv")

# add days since to datbase
basedate = pd.Timestamp('2020-01-01')
df["days_since"] = (pd.to_datetime(df['date']) - basedate).dt.days
# Make different dataframes

# remove null continents (to get countries only)
df1 = df.dropna(subset=['continent'])

# make last day cases by country
last_day_total_cases = df1.groupby(['iso_code'], as_index=False)[
    'new_cases'].sum()


# Worldwide all days
# Make worldwide datbase
worldwide_total = df1.groupby(['date'], as_index=False)['total_cases'].sum()
worldwide_daily_new = df1.groupby(['date'], as_index=False)['new_cases'].sum()
worldwide_daily_deaths = df1.groupby(['date'], as_index=False)[
    'new_deaths'].sum()
worldwide_daily_vaccine = df1.groupby(['date'], as_index=False)[
    'new_vaccinations'].sum()

# add days since to worldwide
basedate = pd.Timestamp('2020-01-01')
worldwide_total["days_since"] = (pd.to_datetime(
    worldwide_total['date']) - basedate).dt.days

# Filter apple dataset
apple1 = apple.drop(
    columns=['geo_type', 'alternative_name', 'sub-region', 'country'])
apple_us = (apple1[apple1['region'] == 'United States'])
apple_us_drive = apple_us[apple_us['transportation_type'] == 'driving']
apple_us_drive = (apple_us_drive.drop(
    columns=['transportation_type', 'region'])).T
apple_us_drive['date'] = apple_us_drive.index

apple_us_walk = apple_us[apple_us['transportation_type'] == 'walking']
apple_us_walk = (apple_us_walk.drop(
    columns=['transportation_type', 'region'])).T
apple_us_walk['date'] = apple_us_walk.index

apple_us_transit = apple_us[apple_us['transportation_type'] == 'transit']
apple_us_transit = (apple_us_transit.drop(
    columns=['transportation_type', 'region'])).T
apple_us_transit['date'] = apple_us_transit.index


apple_edited = pd.DataFrame()
apple_edited['date'] = apple_us_drive['date']
apple_edited['drive'] = apple_us_drive.iloc[:, 0]
apple_edited['walk'] = apple_us_walk.iloc[:, 0]
apple_edited['transit'] = apple_us_transit.iloc[:, 0]
apple_edited = apple_edited[apple_edited["date"] > "2020-01-21"]

# filter covid datset for US
us_covid = df1[df1["location"] == "United States"]

# Apple driving us
apple_drive = go.Figure()
apple_drive.add_trace(go.Bar(
    x=apple_edited['date'],
    y=apple_edited['drive'],
    name='Apple Drive',
    marker_color=colors['clViolet']
))
apple_drive.update_layout(title="Apple mobility trends (driving) in the USA",
                          xaxis_title="Date", yaxis_title="Relative driving volume")
apple_drive.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Apple walking us
apple_walk = go.Figure()
apple_walk.add_trace(go.Bar(
    x=apple_edited['date'],
    y=apple_edited['walk'],
    name='Apple Walk',
    marker_color=colors['clBlue']
))
apple_walk.update_layout(title="Apple mobility trends (walking) in the USA",
                         xaxis_title="Date", yaxis_title="Relative walking volume")
apple_walk.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Apple transit us
apple_transit = go.Figure()
apple_transit.add_trace(go.Bar(
    x=apple_edited['date'],
    y=apple_edited['transit'],
    name='Apple transit',
    marker_color=colors['clOrange']
))
apple_transit.update_layout(title="Apple mobility trends (transit) in the USA",
                            xaxis_title="Date", yaxis_title="Relative transit volume")
apple_transit.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Us Daily Cases
us_daily = go.Figure()
us_daily.add_trace(go.Bar(
    x=us_covid['date'],
    y=us_covid['new_cases'],
    name='US Daily',
    marker_color=colors['clGreen']
))
us_daily.update_layout(title="Daily new covid cases in the USA",
                       xaxis_title="Date", yaxis_title="Daily new cases")
us_daily.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Worldwide graphs

# Worldwide total

ww_t_c = go.Figure()
ww_t_c.add_trace(go.Scatter(x=worldwide_total["date"], y=worldwide_total["total_cases"],
                            mode='lines+markers', name="Total cases",
                            line=dict(color=colors['clBlue'], width=4)))
ww_t_c.update_layout(title="Total covid cases in the world",
                     xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
ww_t_c.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Worldwide daily
ww_d_c = go.Figure()
ww_d_c.add_trace(go.Bar(
    x=worldwide_daily_new["date"],
    y=worldwide_daily_new["new_cases"],
    name='New Cases',
    marker_color=colors['clOrange']
))
ww_d_c.update_layout(title="Daily covid cases in the world",
                     xaxis_title="Date", yaxis_title="Daily covid cases")
ww_d_c.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Worldwide daily deaths
ww_d_d = go.Figure()
ww_d_d.add_trace(go.Bar(x=worldwide_daily_deaths["date"], y=worldwide_daily_deaths["new_deaths"],
                        name='New Deaths', marker_color=colors['clRed']
                        ))
ww_d_d.update_layout(title="Daily covid deaths in the world",
                     xaxis_title="Date", yaxis_title="Daily covid deaths")
ww_d_d.update_layout(
    paper_bgcolor=colors['cahrtBackground'], plot_bgcolor=colors['chartPaper'], font_color=colors['text'],
    title_font_color=colors['text'], legend_title_font_color=colors['text'])

# Worldwide total Vaccinces
ww_t_v = go.Figure()
ww_t_v.add_trace(go.Bar(
    x=worldwide_daily_vaccine["date"],
    y=worldwide_daily_vaccine["new_vaccinations"],
    name='Vaccination',
    marker_color=colors['clGreen']
))
ww_t_v.update_layout(title="Daily covid vaccinations in the world",
                     xaxis_title="Date", yaxis_title="Daily vaccination")
ww_t_v.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Continent cases
continent = df.groupby(['continent'], as_index=False)['new_cases'].sum()
continent_pop = [1373486472, 4678444992,
                 748962983, 596581283, 43219954, 434260137]
continent['pop'] = continent_pop
continent['cases_per_pop'] = continent['new_cases']/continent['pop']


# pie chart

# worldwide
pie_ww_fig = px.pie(continent, values="cases_per_pop", names="continent",
                    title="Total covid cases by continent per popluation",
                    labels={'cases_per_pop': 'Total cases per population', 'continent': 'Continents'})
pie_ww_fig.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# scatter geo
ww_geo_fig = px.scatter_geo(last_day_total_cases, locations="iso_code", size="new_cases", color="iso_code",

                            labels={"date": "Country", "new_cases": "Total cases"})
ww_geo_fig.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Over the time
df2 = df1
df2['date'] = pd.to_datetime(df1['date'], errors='coerce')
fig = px.choropleth(df2, locations="location",
                    color=np.log(df2["total_cases"]),
                    locationmode='country names', hover_name="location",
                    animation_frame=df2["date"].dt.strftime('%Y-%m-%d'),
                    title='Cases over time', color_continuous_scale=px.colors.sequential.matter)
fig.update(layout_coloraxis_showscale=False)

# Correalation between parameters
dfCorr = df1[['total_cases', 'total_deaths', 'total_tests', 'total_vaccinations',
              'handwashing_facilities', 'life_expectancy', 'human_development_index']]
cor = dfCorr.iloc[:, :].corr().style.background_gradient(
    cmap='Reds').format("{:.3f}")
cor2 = dfCorr.iloc[:, :].corr()

corr2Array = cor2.to_numpy().round(decimals=2, out=None)
#labels = ['Total cases', 'Total deaths', 'Total tests','Total vaccinations','Handwashing facilities','Life Expectany','HDI']
labels = ['HDI', 'Life Expectany', 'Handwashing facilities',
          'Total vaccinations', 'Total tests', 'Total deaths', 'Total cases']
fig_cor = ff.create_annotated_heatmap(
    corr2Array, x=labels, y=labels, colorscale='Hot')
fig_cor.update_layout(
    plot_bgcolor=colors['cahrtBackground'],
    paper_bgcolor=colors['cahrtBackground'],
    font_color=colors['text'])


# data for models
# regression
x_total = np.array(worldwide_total['days_since']).reshape(-1, 1)
y_total = np.array(worldwide_total['total_cases'])
# time series
train_data = worldwide_total.iloc[:int(worldwide_total.shape[0]*0.95)]
valid_data = worldwide_total.iloc[int(worldwide_total.shape[0]*0.95):]

x_train = np.array(train_data['days_since']).reshape(-1, 1)
y_train = np.array(train_data['total_cases'])

x_valid = np.array(valid_data['days_since']).reshape(-1, 1)
y_valid = np.array(valid_data['total_cases'])

# Liner regression model

# Fit model
linear_reg_WW = LinearRegression().fit(x_train, y_train)
valid_data['linear_pred'] = linear_reg_WW.predict(x_valid)
train_data['linear_pred'] = linear_reg_WW.predict(x_train)
# score
linear_score = np.sqrt(mean_squared_error(
    valid_data['total_cases'], valid_data['linear_pred']))
mseList = np.append(mseList, linear_score)
modelNames = np.append(modelNames, 'Linear Regression')

# Draw model and data
# make the graph
fig_ww_line_model_data = go.Figure()
fig_ww_line_model_data.add_trace(go.Scatter(x=train_data["date"], y=train_data["total_cases"],
                                            mode='lines+markers', name="Train Data"))
fig_ww_line_model_data.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                                            mode='lines+markers', name="Validation",))
fig_ww_line_model_data.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["linear_pred"],
                                            mode='lines+markers', name="Prediction",))
fig_ww_line_model_data.add_trace(go.Scatter(x=train_data["date"], y=train_data["linear_pred"],
                                            mode='lines+markers', name="Model fit",))
fig_ww_line_model_data.update_layout(title="Prediction of Worlds Total Cases by Linear Regression",
                                     xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
fig_ww_line_model_data.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Polynomial

# making anf fitting
degree = 5
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(x_train, y_train)

# prediction
valid_data['poly_pred'] = polyreg.predict(x_valid)
train_data['poly_pred'] = polyreg.predict(x_train)

# score
poly_score = np.sqrt(mean_squared_error(
    valid_data['total_cases'], valid_data['poly_pred']))
mseList = np.append(mseList, poly_score)
modelNames = np.append(modelNames, 'Polynomial Regression')


# Draw model and data
# make the graph
fig_ww_poly_model_data = go.Figure()
fig_ww_poly_model_data.add_trace(go.Scatter(x=train_data["date"], y=train_data["total_cases"],
                                            mode='lines+markers', name="Train Data"))
fig_ww_poly_model_data.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                                            mode='lines+markers', name="Validation",))
fig_ww_poly_model_data.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["poly_pred"],
                                            mode='lines+markers', name="Prediction",))
fig_ww_poly_model_data.add_trace(go.Scatter(x=train_data["date"], y=train_data["poly_pred"],
                                            mode='lines+markers', name="Model fit",))
fig_ww_poly_model_data.update_layout(title="Modeling of Worlds Total Cases by Polynomial Regression",
                                     xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
fig_ww_poly_model_data.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# SVM

# making and fitting the mopdel
svm_degree = 12
svm = SVR(C=1, degree=svm_degree, kernel='poly', epsilon=0.01)
svm.fit(x_train, y_train)

# prediction
valid_data['svm_pred'] = svm.predict(x_valid)
train_data['svm_pred'] = svm.predict(x_train)

# Score
svm_score = np.sqrt(mean_squared_error(
    valid_data['total_cases'], valid_data['svm_pred']))
#mseList =np.append(mseList, svm_score)
#modelNames = np.append(modelNames, 'SVM')

# Draw model and data
# make the graph
fig_ww_svm = go.Figure()
fig_ww_svm.add_trace(go.Scatter(x=train_data["date"], y=train_data["total_cases"],
                                mode='lines+markers', name="Train Data"))
fig_ww_svm.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                                mode='lines+markers', name="Validation",))
fig_ww_svm.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["svm_pred"],
                                mode='lines+markers', name="Prediction",))
fig_ww_svm.add_trace(go.Scatter(x=train_data["date"], y=train_data["svm_pred"],
                                mode='lines+markers', name="Model fit",))
fig_ww_svm.update_layout(title="Modeling of Worlds Total Cases by SVM",
                         xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
fig_ww_svm.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])


# Holt Model

#Making and fitting
holt = Holt(np.asarray(train_data['total_cases'])).fit(
    smoothing_level=0.4, smoothing_trend=0.4, optimized=False)
# prediction
valid_data['holt_pred'] = holt.forecast(len(valid_data))


# Score
holt_score = np.sqrt(mean_squared_error(
    valid_data['total_cases'], valid_data['holt_pred']))
mseList = np.append(mseList, holt_score)
modelNames = np.append(modelNames, 'Holt')

# graph

fig_holt = go.Figure()
fig_holt.add_trace(go.Scatter(x=train_data["date"], y=train_data["total_cases"],
                              mode='lines+markers', name="Train"))
fig_holt.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                              mode='lines+markers', name="Validation",))
fig_holt.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["holt_pred"],
                              mode='lines+markers', name="Prediction",))
fig_holt.update_layout(title="Prediction of Worlds Total Cases by Holt's Linear Model",
                       xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
fig_holt.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# AR Model (using AUTO ARIMA)

# making and fitting the mopdel
model_ar = auto_arima(train_data['total_cases'], trace=True, error_action='ignore', start_p=0, start_q=0, max_p=4, max_q=0,
                      suppress_warnings=True, stepwise=False, seasonal=False)
model_ar.fit(train_data['total_cases'])

# prediction
valid_data['ar_pred'] = model_ar.predict(len(valid_data))

# Score
ar_score = np.sqrt(mean_squared_error(
    valid_data['total_cases'], valid_data['ar_pred']))
mseList = np.append(mseList, ar_score)
modelNames = np.append(modelNames, 'AR')

# graph

fig_ar = go.Figure()
fig_ar.add_trace(go.Scatter(x=train_data["date"], y=train_data["total_cases"],
                            mode='lines+markers', name="Train"))
fig_ar.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                            mode='lines+markers', name="Validation",))
fig_ar.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["ar_pred"],
                            mode='lines+markers', name="Prediction",))
fig_ar.update_layout(title="Prediction of Worlds Total Cases by AR Model",
                     xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
fig_ar.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Arima Model

# making anf fitting
model_arima = auto_arima(train_data['total_cases'], trace=True, error_action='ignore', start_p=1, start_q=1, max_p=3, max_q=3,
                         suppress_warnings=True, stepwise=False, seasonal=False)
model_arima.fit(train_data['total_cases'])


# prediction
valid_data['arima_pred'] = model_arima.predict(len(valid_data))

# Score
arima_score = np.sqrt(mean_squared_error(
    valid_data['total_cases'], valid_data['arima_pred']))
mseList = np.append(mseList, arima_score)
modelNames = np.append(modelNames, 'ARIMA')

# graph

fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(x=train_data["date"], y=train_data["total_cases"],
                    mode='lines+markers', name="Train"))
fig_arima.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                    mode='lines+markers', name="Validation",))
fig_arima.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["arima_pred"],
                    mode='lines+markers', name="Prediction",))
fig_arima.update_layout(title="Prediction of Worlds Total Cases by ARIMA Model",
                        xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
fig_arima.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])


start = datetime.datetime.strptime("2021-11-03", "%Y-%m-%d")
end = datetime.datetime.strptime("2022-01-31", "%Y-%m-%d")
date_generated = [
    start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
predict = pd.DataFrame()
predict['date'] = date_generated
predict['arima_pred'] = model_arima.predict(len(predict))

fig_arima_pred = go.Figure()
fig_arima_pred.add_trace(go.Scatter(x=predict["date"], y=predict["arima_pred"],
                                    mode='lines+markers', name="Train"))
fig_arima_pred.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                                    mode='lines+markers', name="Validation",))
fig_arima_pred.update_layout(title="Prediction of Worlds Total Cases by ARIMA Model To The End of Junuary 2022",
                             xaxis_title="Date", yaxis_title="Total Cases")
fig_arima_pred.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# All prediction validation in one chart
fig_all_pred = go.Figure()
fig_all_pred.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["total_cases"],
                                  mode='lines', name="Validation",))
fig_all_pred.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["linear_pred"],
                                  mode='lines+markers', name="Linear Regression",))
fig_all_pred.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["poly_pred"],
                                  mode='lines+markers', name="Polynomial Regression",))
fig_all_pred.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["ar_pred"],
                                  mode='lines+markers', name="AR",))
fig_all_pred.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["holt_pred"],
                                  mode='lines+markers', name="Holt",))
fig_all_pred.add_trace(go.Scatter(x=valid_data["date"], y=valid_data["arima_pred"],
                                  mode='lines+markers', name="Arima",))
fig_all_pred.update_layout(title="Comparison of prediction models",
                           xaxis_title="Date", yaxis_title="Total Cases")
fig_all_pred.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# All prediction comparison Bar
# Worldwide total Vaccinces
all_pred_bar = go.Figure()
all_pred_bar.add_trace(go.Bar(
    x=modelNames,
    y=mseList,
    name='Models',
    marker_color=colors['clGreen']
))
all_pred_bar.update_layout(title="Model Comparison",
                           xaxis_title="Model Name", yaxis_title="MSE")
all_pred_bar.update_layout(
    paper_bgcolor=colors['cahrtBackground'],
    plot_bgcolor=colors['chartPaper'],
    font_color=colors['text'],
    title_font_color=colors['text'],
    legend_title_font_color=colors['text'])

# Making the dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Covid 19 Visualization and prediction"
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Div([
        html.H1(children='Covid statistics Worldwide', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
        html.Div([
            dcc.Graph(id='ww_total_graph', figure=ww_t_c, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      ),
            dcc.Graph(id='ww_total_Vaccine', figure=ww_t_v, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      )]),
        html.Div([
            dcc.Graph(id='ww_daily_graph', figure=ww_d_c, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      ),
            dcc.Graph(id='ww_daily_deaths_graph', figure=ww_d_d, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      )]),

    ]),
    html.Div([
        html.H1(children='Total Cases on the World Map', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
        dcc.Graph(id='ww_geo', figure=ww_geo_fig, clickData=None, hoverData=None,
                  config={
                     'staticPlot': False,     # True, False
                      'scrollZoom': True,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                  },
                  className='row'
                  )
    ]),
    html.Div([
        html.H1(children='Covid statistics by continent', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
        dcc.Graph(id='continent_total_graph', figure=pie_ww_fig, clickData=None, hoverData=None,
                  config={
                      'staticPlot': False,     # True, False
                      'scrollZoom': True,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                  },
                  className='row'
                  )
    ]),
    html.Div([
        html.H1(children='Covid statistics by country', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),

        html.H4(children='Select country', style={
                'textAlign': 'left', 'color': colors['text'], 'backgroundColor': colors['chartPaper']}),
        dcc.Dropdown(id='dpdn1', value=['Norway'], multi=False,
                     options=[{'label': x, 'value': x} for x in
                              df1.location.unique()],
                     style={'color': colors['chartPaper'], 'background-color': colors['chartPaper']}),
        html.Div([
            dcc.Graph(id='country_total_graph', figure={}, clickData=None, hoverData=None,
                      config={
                      'staticPlot': False,     # True, False
                      'scrollZoom': True,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      ),
            dcc.Graph(id='country_total_Vaccine', figure={}, clickData=None, hoverData=None,
                      config={
                'staticPlot': False,     # True, False
                'scrollZoom': True,      # True, False
                'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                'showTips': False,       # True, False
                'displayModeBar': True,  # True, False, 'hover'
                'watermark': True,
                # 'modeBarButtonsToAdd': ['pan2d','select2d'],
            },
                className='six columns'
            )]),
        html.Div([
            dcc.Graph(id='country_daily_graph', figure={}, clickData=None, hoverData=None,
                      config={
                'staticPlot': False,     # True, False
                      'scrollZoom': True,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      ),
            dcc.Graph(id='country_daily_deaths_graph', figure={}, clickData=None, hoverData=None,
                      config={
                'staticPlot': False,     # True, False
                'scrollZoom': True,      # True, False
                'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                'showTips': False,       # True, False
                'displayModeBar': True,  # True, False, 'hover'
                'watermark': True,
                # 'modeBarButtonsToAdd': ['pan2d','select2d'],
            },
                className='six columns'
            )])
    ]),

    # Prediction charts
    html.Div([
        html.H1(children='Total covid case predictions', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
        html.Div([
            dcc.Graph(id='linear_pred', figure=fig_ww_line_model_data, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='four columns'
                      ),
            dcc.Graph(id='poly_pred', figure=fig_ww_poly_model_data, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='four columns'
                      ),

            dcc.Graph(id='svm_pred', figure=fig_ww_svm, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='four columns'
                      )
        ],
            className='row'),
        html.Div([], className='row'),
        html.Div([
            dcc.Graph(id='holt_pred', figure=fig_holt, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='four columns'
                      ),
            dcc.Graph(id='ar_pred', figure=fig_ar, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='four columns'
                      ),
            dcc.Graph(id='arima_pred', figure=fig_arima, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='four columns'
                      )
        ], className='row'),
        html.Div([
            html.H1(children='Comparison of prediction models', style={
                    'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
            html.Div([
                dcc.Graph(id='pred_compare', figure=fig_all_pred, clickData=None, hoverData=None,
                          config={
                              'staticPlot': False,     # True, False
                              'scrollZoom': True,      # True, False
                              'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                              'showTips': False,       # True, False
                              'displayModeBar': True,  # True, False, 'hover'
                              'watermark': True,
                              # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                          },
                          className='six columns'
                          ),
                dcc.Graph(id='pred_compare_bar', figure=all_pred_bar, clickData=None, hoverData=None,
                          config={
                              'staticPlot': False,     # True, False
                              'scrollZoom': True,      # True, False
                              'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                              'showTips': False,       # True, False
                              'displayModeBar': True,  # True, False, 'hover'
                              'watermark': True,
                              # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                          },
                          className='six columns'
                          )
            ])
        ])
    ]),
    html.Div([
        html.H1(children='Future Covid Prediction', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
        html.Div([
            dcc.Graph(id='future_pred', figure=fig_arima_pred, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='row'
                      )
        ])
    ]),
    # Correlations
    html.Div([
        html.H1(children='Correlation Between parameters', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
        html.Div([
            dcc.Graph(id='correlation', figure=fig_cor, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='row'
                      )
        ])
    ]),
    # Apple Mobility
    html.Div([
        html.H1(children='Reviewing Apples Moblility Datbase', style={
                'textAlign': 'center', 'color': colors['text'], 'backgroundColor': colors['clh1b']}),
        html.Div([
            dcc.Graph(id='us_daily', figure=us_daily, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      ),
            dcc.Graph(id='apple_walk', figure=apple_walk, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      )]),
        html.Div([
            dcc.Graph(id='apple_transit', figure=apple_transit, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      ),
            dcc.Graph(id='apple_drive', figure=apple_drive, clickData=None, hoverData=None,
                      config={
                          'staticPlot': False,     # True, False
                          'scrollZoom': True,      # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,       # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToAdd': ['pan2d','select2d'],
                      },
                      className='six columns'
                      )]),

    ]),
    html.Div([
        html.H1(children='----', style={'textAlign': 'center',
                'color': colors['text'], 'backgroundColor': colors['clh1b']})
    ])

])

# Country graphs Dash

# country total


@app.callback(
    Output(component_id='country_total_graph', component_property='figure'),
    Input(component_id='dpdn1', component_property='value'),
)
def update_graph(country_chosen):
    dff = df1[df1["location"] == country_chosen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["date"], y=dff["total_cases"],
                             mode='lines+markers', name="Total cases",
                             line=dict(color=colors['clBlue'], width=4)))
    fig.update_layout(title="Total covid cases in "+country_chosen,
                      xaxis_title="Date", yaxis_title="Total Cases", legend=dict(x=0, y=1, traceorder="normal"))
    fig.update_layout(
        paper_bgcolor=colors['cahrtBackground'],
        plot_bgcolor=colors['chartPaper'],
        font_color=colors['text'],
        title_font_color=colors['text'],
        legend_title_font_color=colors['text'])

    return fig

# country daily


@app.callback(
    Output(component_id='country_daily_graph', component_property='figure'),
    Input(component_id='dpdn1', component_property='value'),
)
def update_graph(country_chosen):
    dff = df1[df1["location"] == country_chosen]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dff["date"],
        y=dff["new_cases"],
        name='New Cases',
        marker_color=colors['clOrange']
    ))
    fig.update_layout(title="Daily new Covid cases in "+country_chosen,
                      xaxis_title="Date", yaxis_title="Daily cases")
    fig.update_layout(
        paper_bgcolor=colors['cahrtBackground'],
        plot_bgcolor=colors['chartPaper'],
        font_color=colors['text'],
        title_font_color=colors['text'],
        legend_title_font_color=colors['text'])
    return fig

# country daily deaths


@app.callback(
    Output(component_id='country_daily_deaths_graph',
           component_property='figure'),
    Input(component_id='dpdn1', component_property='value'),
)
def update_graph(country_chosen):
    dff = df1[df1["location"] == country_chosen]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dff["date"],
        y=dff["new_deaths"],
        name='New deaths',
        marker_color=colors['clRed']
    ))
    fig.update_layout(title="Daily new Covid deaths in " + country_chosen,
                      xaxis_title="Date", yaxis_title="Daily deaths")
    fig.update_layout(
        paper_bgcolor=colors['cahrtBackground'],
        plot_bgcolor=colors['chartPaper'],
        font_color=colors['text'],
        title_font_color=colors['text'],
        legend_title_font_color=colors['text'])
    return fig
 # country total Vaccinces


@app.callback(
    Output(component_id='country_total_Vaccine', component_property='figure'),
    Input(component_id='dpdn1', component_property='value'),
)
def update_graph(country_chosen):
    dff = df1[df1["location"] == country_chosen]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dff["date"],
        y=dff["total_vaccinations"],
        name='Total vaccine',
        marker_color=colors['clGreen']
    ))
    fig.update_layout(title="Total vaccination in " + country_chosen,
                      xaxis_title="Date", yaxis_title="Daily deaths")
    fig.update_layout(
        paper_bgcolor=colors['cahrtBackground'],
        plot_bgcolor=colors['chartPaper'],
        font_color=colors['text'],
        title_font_color=colors['text'],
        legend_title_font_color=colors['text'])
    return fig


# Run Dash App
if __name__ == '__main__':
    app.run_server(debug=False)
