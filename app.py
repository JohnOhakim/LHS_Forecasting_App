import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas_gbq

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.simplefilter(action="ignore")

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from jupyterlab_dash import AppViewer
viewer = AppViewer()

import fbprophet

project_id = "cp-gaa-visualization-dev"
host = '0.0.0.0'



def query_table(sql_query):
    
    return pandas_gbq.read_gbq(sql_query, project_id=project_id)


def prep_data_for_tsmodel(data):
    
    data = data.groupby(['product_title', 'start_date'], 
                   as_index=False)[['shipped_units']].sum()
    data.sort_values('start_date', ascending=True, inplace=True)
    data.set_index('start_date', drop=False, inplace=True)
    
    data = data.loc[:, ['start_date', 'shipped_units']]
    data = data.resample('D').sum()
    data.reset_index(inplace=True)
    
    return data.rename(columns={'start_date': 'ds', 'shipped_units': 'y'})



sql_query = """
SELECT 
  asin,
  product_title,
  category,
  subcategory,
  shipped_units,
  rep_oos,
  subcategory_sales_rank,
  change_in_gv_prior_period,
  start_date,
  end_date, 
  country
    
FROM 
  `cp-gaa-visualization-dev.arap.sales_diagnostic_detail_dly`
  
WHERE
  product_title LIKE "%Softsoap%" AND product_title LIKE "%Liquid Hand Soap%" 
  AND
  country = 'USA'
  AND 
  start_date >= '2019-10-01' 
"""
# AND end_date < '2020-05-27')

sales_diag = query_table(sql_query)


lhs_daily = prep_data_for_tsmodel(sales_diag)


ecomm_days = {'2020-07-06': 'Prime Day', '2019-11-29': 'Black Friday', '2019-12-02': 'Cyber Monday'}
year = [2019, 2020]


# covid_df_1 = pd.read_csv('./data/new-covid-cases-per-million (1).csv')
# covid_df_2 = covid_df_1[covid_df_1['Entity'] == 'United States']

# covid_df_2.rename(columns={'Entity': 'country', 'Code': 'code', 'Date': 'ds', 
#                   'Daily new confirmed cases of COVID-19 per million people (cases)': 'infection_rate'}, inplace=True)

# covid_df_2['ds'] = pd.to_datetime(covid_df_2.ds)
# covid_df_2 = covid_df_2.loc[:, ['ds', 'infection_rate']]
# covid_df_2.reset_index(inplace=True, drop=True)

# lhs_daily_added_reg = pd.merge(lhs_daily, covid_df_2, on='ds', how='outer')
# lhs_daily_added_reg.fillna(0, inplace=True)

# lhs_len = int(round(lhs_daily_added_reg.shape[0] * (0.90), 0))

# lhs_train = lhs_daily_added_reg.iloc[0:lhs_len]

# lhs_test = lhs_daily_added_reg[lhs_len:]

# len_of_train = len(lhs_train)
# len_of_test = len(lhs_test)

############
df_mobility = pd.read_csv('./data/Global_Mobility_Report.csv')
df_mobility = df_mobility[df_mobility['country_region'] == 'United States']
df_mobility_filtered = df_mobility.groupby(['date'], 
                   as_index=False)[['retail_and_recreation_percent_change_from_baseline']].median()
df_mobility_filtered.rename(columns={'date': 'ds', 
                  'retail_and_recreation_percent_change_from_baseline': 'extra_var'}, inplace=True)

df_mobility_filtered['ds'] = pd.to_datetime(df_mobility_filtered.ds)
df_mobility_filtered = df_mobility_filtered.loc[:, ['ds', 'extra_var']]
df_mobility_filtered.reset_index(inplace=True, drop=True)

lhs_daily_added_regs = pd.merge(lhs_daily, df_mobility_filtered, on='ds', how='outer')
lhs_daily_added_regs.fillna(0, inplace=True)

lhs_len = int(round(lhs_daily_added_regs.shape[0] * (0.90), 0))

lhs_train = lhs_daily_added_regs.iloc[0:lhs_len]

lhs_test = lhs_daily_added_regs[lhs_len:]

len_of_train = len(lhs_train)
len_of_test = len(lhs_test)




external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
   
    
def return_model_accuracy(predictions_df, history_df):
   
    #metric_df = predictions_df.set_index('ds')[['yhat']].join(history_df.set_index('ds').y).reset_index()
    
    a = predictions_df.set_index('ds')[['yhat']]
    b = history_df.set_index('ds').y
    metric_df = a.join(b).reset_index()
    metric_df = metric_df.dropna()
    
    r2 = r2_score(metric_df.y, metric_df.yhat)
    mse = mean_squared_error(metric_df.y, metric_df.yhat)
    mae = mean_absolute_error(metric_df.y, metric_df.yhat)
    rsme = np.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
    
    return [rsme, r2, mae, mse]

def suppress_sci_not(y):
    return round(y, 2)

def get_holidays(custom_days, year):
    import holidays 
    holiday_dict = holidays.US(years=year)
    custom_holiday_dict = holidays.HolidayBase()
    custom_holiday_dict.append(custom_days)

    holiday_list = []
    date_list = []
    for i in holiday_dict.items():
        date_list.append(i[0])
        holiday_list.append(i[1])
    df_a = pd.DataFrame({'ds': date_list, 'holiday': holiday_list})

    custom_holiday_list = []
    custom_date_list = []   
    for i in custom_holiday_dict.items():
        custom_date_list.append(i[0])
        custom_holiday_list.append(i[1])
    df_b = pd.DataFrame({'ds': custom_date_list, 'holiday': custom_holiday_list})

    holiday_df = pd.concat([df_a, df_b])
    holiday_df.reset_index(drop=True)
    return holiday_df.sort_values(by='ds')






controls = dbc.Card(
    [
        html.H5("Graphical Widgets"),
        dbc.FormGroup(
            [
                dbc.Label("Growth", color='primary'),
                dbc.RadioItems(
                    options=[{'label': i, 'value': i} for i in ['linear', 'logistic']],
                    value='linear',
                    id="model-selector"
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Add Feature", color='primary'),
                dbc.Checklist(
                    id="add-feature",
                    options=[
                        {'label': 'Add Regressor', 'value': 'added_regressor'}
                    ],
                    value=['added_regressor'],
                    switch=True
                ),
            ]
        ),
        dbc.FormGroup(
        [
            dbc.Label("Changepoints", html_for="slider", color='primary'),
            dcc.Slider(
                id='prior-scale-tuner',
                min=0,
                max=2,
                step=0.001,
                marks={
                    0.05: {'label': '0.05'},
                    0.25: {'label': '0.25'},
                    0.8: {'label': '0.8'},
                    1: {'label': '1'},
                    2: {'label': '2'}
                },
                value=0.8,
                included=False
            )
        ]
    ),
    dcc.Dropdown(
        id='forecast-period',
        options=[
            {'label': '45 days', 'value': 45},
            {'label': '60 days', 'value': 60},
            {'label': '90 days', 'value': 90}
        ],
        placeholder='Number of Days',
        value=45
    ),
    ],
    body=True,
    
)

table = dbc.FormGroup(
    [
        dbc.Label("Forecasted Data", size='lg'),
        dbc.Table.from_dataframe(pd.DataFrame(columns=['A', 'B'], index=range(3)).reset_index(), 
                                 id='forecast-table', 
                                 striped=True, 
                                 bordered=True, 
                                 hover=True
                                )
    ]

)


app.layout = dbc.Container(
    [
        html.Img(src=app.get_asset_url('colgate-logo-2.jpeg'), style={'height':'5%', 'width':'10%'}),
        html.H3("Forecast Model"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, xs=4),
                dbc.Col(dcc.Graph(id='forecast-graph',
                                  config={'displayModeBar': False}
                                 ), 
                        align='stretch'),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(table),
                
            ]
        )
    ],
    fluid=True,
)


@app.callback(Output('forecast-graph', 'figure'),
              [Input('model-selector', 'value'), 
               Input('prior-scale-tuner', 'value'), 
               Input('forecast-period', 'value'), 
               Input('add-feature', 'value')])
def update_graph(model_selected, cp_scale, num_of_days, added_feature):

    holidays_df = get_holidays(ecomm_days, year)


    #dates = ['2020-02-29', '2020-03-29']
    dates = ['2020-02-29', '2020-04-01']

    name_of_regressor = 'extra_var'
    
    def forecast_data(data, holidays_df, dates, cap=21000, 
                  cp_prior_scale=cp_scale, fourier_order=15, periods=num_of_days, 
                  added_regressor = added_feature, added_reg_name='added_reg'):
    
        if added_regressor == added_feature:
            data['cap'] = cap
            data['covid_period'] = np.where((data['ds'] > dates[0]) & (data['ds'] < dates[1]), True, False)

            model = fbprophet.Prophet(changepoint_prior_scale=cp_prior_scale, yearly_seasonality=False, 
                                           holidays=holidays_df, growth=model_selected
                                          )

            model.add_country_holidays(country_name='US')
            model.add_regressor(added_reg_name)
            model.add_seasonality(name='yearly', period=30, fourier_order=fourier_order, condition_name='covid_period')
            model.fit(data)

            preds_df = model.make_future_dataframe(periods=periods, freq='D')
            preds_df['cap'] = cap
            preds_df['covid_period'] = np.where((preds_df['ds'] > dates[0]) & (preds_df['ds'] < dates[1]), True, False)
            preds_df[added_reg_name] = data[added_reg_name]
            preds_df[added_reg_name].fillna(0, inplace=True)

            return model.predict(preds_df), model
            

        else:
            data['cap'] = cap
            data['covid_period'] = np.where((data['ds'] > dates[0]) & (data['ds'] < dates[1]), True, False)

            model = fbprophet.Prophet(changepoint_prior_scale=cp_prior_scale, yearly_seasonality=False, 
                                           holidays=holidays_df, growth=model_selected
                                          )

            model.add_country_holidays(country_name='US')
            model.add_seasonality(name='yearly', period=30, fourier_order=fourier_order, condition_name='covid_period')
            model.fit(data)

            preds_df = model.make_future_dataframe(periods=periods, freq='D')
            preds_df['cap'] = cap
            preds_df['covid_period'] = np.where((preds_df['ds'] > dates[0]) & (preds_df['ds'] < dates[1]), True, False)

            return model.predict(preds_df), model
                   
    preds, model = forecast_data(lhs_train, holidays_df, dates, added_reg_name=name_of_regressor)
    
    
    fig = go.Figure({
                          'data': [
                              {
                                  'x':lhs_train.ds, 
                                  'y':lhs_train.y,
                                  'mode':'markers',
                                  'marker':{'color':'Black',
                                            'opacity':0.6,
                                            'size':4.5
                                           },
                                  'name':'Actual'

                              },

                              {
                                  'x':lhs_test.ds, 
                                  'y':lhs_test.y,
                                  'mode':'markers',
                                  'marker':{'color':'red', 
                                            'opacity':0.8,
                                            'size':4.5
                                           },
                                  'name':'Holdout'

                              },

                              {
                                  'x':preds.ds, 
                                  'y':preds.yhat_lower,
                                  'fillcolor':'rgba(63, 127, 191, 0.2)',
                                  'line':{'width':0},
                                  'name':'Lower Bound', 
                                  'showlegend':False

                                  },

                              {
                                  'x':preds.ds, 
                                  'y':preds.yhat_upper,
                                  'fill':'tonexty', 
                                  'fillcolor':'rgba(63, 127, 191, 0.2)',
                                  'line':{'width':0},
                                  'name':'Upper Bound',
                                  'showlegend':False

                              },


                              {
                                  'x':preds.ds, 
                                  'y':preds.yhat,
                                  'mode':'lines',
                                  'line_color':'#3F7FBF',
                                  'name':'Prediction'

                              }],
                        'layout':{#'height':700, 
                                  'yaxis':{'title_text':'Shipped Units'},
                                  'title':{'text':f'{num_of_days} Day Forecast',
                                           'font':{'size':25}
                                          },
                                  'xaxis_title':'Date'

                              }
        })

    fig.add_shape({

                                  'type': 'line',
                                  'yref': 'y', 'y0': 0, 'y1': lhs_train.y.max(),
                                  'xref': 'x', 'x0': preds.ds[len_of_train], 'x1': preds.ds[len_of_train],
                                  'line':{'color':"Red",
                                          'width':1
                                         }

                          })

    fig.add_shape({
        'type': 'line',
        'yref': 'y', 'y0': 0, 'y1': lhs_train.y.max(),
        'xref': 'x', 'x0': preds.ds[len_of_train], 'x1': preds.ds[len_of_train],
        'line':{'color':"Red",
                'width':1
               }
    }),
    fig.add_shape({
        'type': 'line',
        'yref': 'y', 'y0': 0, 'y1': lhs_train.y.max(),
        'xref': 'x', 'x0': lhs_test.ds.iloc[-1], 'x1': lhs_test.ds.iloc[-1],
        'line':{'color':"Green",
                'width':1
               }
            }) 
    
    return fig

@app.callback(Output('forecast-table', 'children'), 
             [Input('model-selector', 'value'), 
               Input('prior-scale-tuner', 'value'), 
               Input('forecast-period', 'value'), 
               Input('add-feature', 'value') 
               ]
             )
def generate_data_table(model_selected, cp_scale, num_of_days, added_feature):
    holidays_df = get_holidays(ecomm_days, year)

    #dates = ['2020-02-29', '2020-03-29']
    dates = ['2020-02-29', '2020-04-01']

    #name_of_regressor = 'infection_rate'
    name_of_regressor = 'extra_var'
    
    def forecast_data(data, holidays_df, dates, cap=21000, 
                  cp_prior_scale=cp_scale, fourier_order=15, periods=num_of_days, 
                  added_regressor = added_feature, added_reg_name='added_reg'):
    
        if added_regressor == added_feature:
            data['cap'] = cap
            data['covid_period'] = np.where((data['ds'] > dates[0]) & (data['ds'] < dates[1]), True, False)

            model = fbprophet.Prophet(changepoint_prior_scale=cp_prior_scale, yearly_seasonality=False, 
                                           holidays=holidays_df, growth=model_selected
                                          )

            model.add_country_holidays(country_name='US')
            model.add_regressor(added_reg_name)
            model.add_seasonality(name='yearly', period=30, fourier_order=fourier_order, condition_name='covid_period')
            model.fit(data)

            preds_df = model.make_future_dataframe(periods=periods, freq='D')
            preds_df['cap'] = cap
            preds_df['covid_period'] = np.where((preds_df['ds'] > dates[0]) & (preds_df['ds'] < dates[1]), True, False)
            preds_df[added_reg_name] = data[added_reg_name]
            preds_df[added_reg_name].fillna(0, inplace=True)

            return model.predict(preds_df), model
            

        else:
            data['cap'] = cap
            data['covid_period'] = np.where((data['ds'] > dates[0]) & (data['ds'] < dates[1]), True, False)

            model = fbprophet.Prophet(changepoint_prior_scale=cp_prior_scale, yearly_seasonality=False, 
                                           holidays=holidays_df, growth=model_selected
                                          )

            model.add_country_holidays(country_name='US')
            model.add_seasonality(name='yearly', period=30, fourier_order=fourier_order, condition_name='covid_period')
            model.fit(data)

            preds_df = model.make_future_dataframe(periods=periods, freq='D')
            preds_df['cap'] = cap
            preds_df['covid_period'] = np.where((preds_df['ds'] > dates[0]) & (preds_df['ds'] < dates[1]), True, False)

            return model.predict(preds_df), model
                  
    preds, model = forecast_data(lhs_train, holidays_df, dates, added_reg_name=name_of_regressor)
    
    model_forecast = preds.loc[len_of_train:,['ds', 'yhat']]
    forecast_ts = model_forecast.set_index('ds')
    monthly_forecast = forecast_ts.resample('M').sum()
    monthly_forecast['yhat'] = monthly_forecast.yhat.apply(suppress_sci_not)
    monthly_forecast.reset_index(inplace=True)
    monthly_forecast.rename(columns={'ds': 'Date', 'yhat': 'Shipped Units'}, inplace=True)
    return dbc.Table.from_dataframe(monthly_forecast, 
                                    striped=True, 
                                    bordered=True, 
                                    hover=True,
                                    date_format='%m/%Y'
                                   )
    

    
port = 8051
    
app.run_server(debug=True, host=host, port=port)