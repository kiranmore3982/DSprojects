import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import sys
import matplotlib.pyplot as plt
import altair as alt
st.set_page_config(layout="wide")
import time
import random
from datetime import datetime
global dfFinal,dfIndivudal
dfFinal = pd.DataFrame()
dfIndivudal = pd.DataFrame()
global col1, col2,col3
col1, col2,col3 = st.columns([3,1,1]) 


def download_stocks_history(stksybl):
    stock_final = pd.DataFrame()
    stock_symbol = stksybl 
    try:
        stock = []
        stock = yf.download(stock_symbol,period='5y')
        if len(stock) == 0:
            None
        else:
            stock_final = stock_final.append(stock,sort=False)
    except Exception:
        None
    return stock_final

def load_model():
    model = keras.models.load_model('finalized_model.h5')
    return model

def predict_future_data(next_prediction_days,stocks_data,compCode):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_reference_data = 60 
    next_future_days = next_prediction_days
    LSTM_test_data = np.log(stocks_data[int(len(stocks_data)*0.9):])
    
    
  
    x_input=LSTM_test_data[len(LSTM_test_data)-past_reference_data:].values.reshape(-1,1)
    
    x_temp_input = scaler.fit_transform(x_input)
    x_input = x_temp_input.reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    


    from numpy import array
    from datetime import timedelta
    from datetime import date
    
    import time
    LSTM_model = load_model()
    lst_output=[]
    n_steps=past_reference_data
    i=0

    while(i<next_future_days):
        
        if(len(temp_input)>past_reference_data):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            st.write(x_input)
            yhat = LSTM_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            st.write(x_input)
            yhat = LSTM_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    st.write(lst_output)

    day_new=np.arange(1,past_reference_data+1)
    day_pred=np.arange(past_reference_data+1,past_reference_data+1+next_future_days)

    # Predict Next Stock Price
    lstm_nxt30_stocks = np.exp(scaler.inverse_transform(lst_output))
    future_30days_stock_predictions = []
    future_30days_stock_predictions = pd.DataFrame(index=range(0,next_future_days),columns=['Date', compCode])
    Begindate = date.today()
    for i in range(0,next_future_days):
        future_30days_stock_predictions['Date'][i] = Begindate + timedelta(days=i+1)
        future_30days_stock_predictions[compCode][i] = lstm_nxt30_stocks[i][0]
        
    
    if dfFinal.empty:
        dfFinal['Date'] = future_30days_stock_predictions['Date']
        dfFinal[compCode] = future_30days_stock_predictions[compCode]
    else:
        dfFinal[compCode] = future_30days_stock_predictions[compCode]
    dfIndivudal = dfFinal
    dfIndivudal = dfIndivudal[['Date',compCode]]
     
    # Plot Next Stock Price
    with st.spinner('Wait for it...loading'):
        time.sleep(1) 
        
    return dfFinal,scaler,day_new,day_pred,x_temp_input,past_reference_data,next_prediction_days,future_30days_stock_predictions,dfIndivudal
 
def plot_future_prediction(scaler,day_new,day_pred,x_temp_input,past_reference_data,next_prediction_days,future_30days_stock_predictions):
    future_30days_stock_predictions.set_index('Date',inplace=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(day_new,np.exp(scaler.inverse_transform(x_temp_input[len(x_temp_input)-past_reference_data:])),color = 'blue',label='Past 60 days data')
    ax.plot(day_pred,future_30days_stock_predictions,color = 'orange',label='Next '+str(next_prediction_days)+' days Data')
    ax.legend()
    st.pyplot(fig)

def plot_combined_future_prediction(combined_data):

    df = combined_data
    df = df.melt('Date', var_name='Stoks', value_name='value')
    chart = alt.Chart(df).mark_line().encode(
    x=alt.X('yearmonthdate(Date):O',title='Day'),
    y=alt.Y('value:Q',title='Predicted Close',bin={'maxbins':60}),
    color=alt.Color("Stoks:N"),
    ).properties(title="Stock Prediction",width=400,height=500)
    st.altair_chart(chart, use_container_width=True)


def main():
    with col1:
        st.title("NSE Stock Price Prediction")
        stk_multiOptions = st.multiselect(
        'Select stocks for prediction',
        ['BAJFINANCE','HDFC','HDFCBANK','HINDUNILVR','ICICIBANK','INFY','RELIANCE','SBIN','TCS','WIPRO'])
        
        predicted_days_option = st.selectbox(
        'No Of Future Days?',
        (5,7,10,25,30))
        if st.button('Predict'):
            global dfFinal
            with st.spinner('Wait for it...loading'):
                time.sleep(1)        
            companies = stk_multiOptions
            
            next_prediction_days = predicted_days_option
        
            for companyCode in companies:
                stocks_data = download_stocks_history(companyCode+".NS") 
                dfFinal,scaler,day_new,day_pred,x_temp_input,past_reference_data,next_prediction_days,future_30days_stock_predictions,dfIndivudal = predict_future_data(next_prediction_days,stocks_data["Adj Close"],companyCode)
                with col3:
                    if dfIndivudal.index.name!='Date':
                        dfIndivudal.set_index('Date',inplace=True)
                        st.dataframe(dfIndivudal,width=300,height=200)
                with col2:
                    plot_future_prediction(scaler,day_new,day_pred,x_temp_input,past_reference_data,next_prediction_days,future_30days_stock_predictions)
            with st.spinner('Wait for it...loading'):
                time.sleep(2)  
            # if dfFinal.index.name!='Date':
            #     dfFinal.set_index('Date',inplace=True)
            with col1:
                st.write("==========  Result  ==========")
                st.dataframe(dfFinal,width=1000,height=2000)
                plot_combined_future_prediction(dfFinal)
        
    
             
    
if __name__=='__main__':
    main()



