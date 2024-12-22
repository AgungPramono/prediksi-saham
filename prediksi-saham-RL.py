from pyexpat import model
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import numpy as np

from sklearn.metrics import mean_squared_error
import  numpy as np


data = pd.read_csv('dataset\saham_new.csv')  # Ganti dengan nama filemu

# Mengatur pandas untuk menampilkan semua baris
pd.set_option('display.max_rows', None)  # Menampilkan semua baris
pd.set_option('display.max_columns', None)  # Menampilkan semua kolom (opsional)
pd.set_option('display.width', None)  # Menyesuaikan lebar tampilan

print(data.head())
print(data.info())
print(data.isnull().sum())

print(data.describe())

numeric_data = data.select_dtypes(include=['float64', 'int64'])
print(numeric_data.corr())
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
#plt.show()

#plt.figure(figsize=(10, 5))
#plt.plot(data['Date'], data['Close'], label='Close Price')
#plt.title('Close Price Over Time')
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.legend()
#plt.show()

#plt.figure(figsize=(10, 5))
#plt.bar(data['Date'], data['Volume'], color='orange', label='Volume')
#plt.title('Trading Volume Over Time')
#plt.xlabel('Date')
#plt.ylabel('Volume')
#plt.legend()
#plt.show()

#data['Date'] = pd.to_datetime(data['Date'])
#data.set_index('Date', inplace=True)

#result = seasonal_decompose(data['Close'], model='additive', period=5)
#result.plot()
#plt.show()

#plt.boxplot(data['Close'])
#plt.title('Boxplot of Close Prices')
#plt.show()

x = data[['Open','High','Low','Volume']]
y = data[['Close']]

# split data 90% training 10% testing
X_train,X_test, y_train,y_test = train_test_split(x,y, test_size=0.1, random_state=42, shuffle=False)

print("\nJumlah data training : ", len(X_train))
print("Data Training", X_train.head())


print("\nJumlah data testing : ", len(X_test))
print("Data Testing",X_test.head())


model = LinearRegression()

model.fit(X_train, y_train)

#prediksi
y_pred = model.predict(X_test)

#hasil prediksi
print("===================================================================================")
result_df = X_test.copy()
result_df['Date'] = data.loc[X_test.index,'Date']
result_df['Actual'] = y_test
result_df['Predicted'] = y_pred.round()
result_df['Index'] = range(1, len(result_df) + 1)
print("\nJumlah Data Prediksi :",len(result_df))
print("Hasil Prediksi:")
print("===================================================================================")
#print(result_df)
print(result_df[['Index','Date','Open','High','Low','Volume', 'Actual','Predicted']])
print("===================================================================================")

print("===================================================================================")
#evaluasi
rmse = np.sqrt(root_mean_squared_error(y_test,y_pred)) #RMSE
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
r2 = r2_score(y_test, y_pred)  # R-squared
print('\nRMSE', round(rmse, 2))
print('MSE',round( mse, 2))
print('MAE',round( mae, 2))
print('R-Squared', r2)
print("===================================================================================")


#print("Rentang data :",result_df[(result_df['Date'] >= '2020-01-01') & (result_df['Date'] <= '2020-01-31')])


#tampilkan grafik actual dan prediksi
# Pastikan kolom Date diubah ke tipe datetime
result_df['Date'] = pd.to_datetime(result_df['Date'])

# Downsampling: Filter data mulai 1 Januari 2020 hingga 11 Desember 2024
""" filtered_df = result_df[
    (result_df['Date'] >= '2020-01-01') & 
    (result_df['Date'] <= '2024-12-12') & 
    (result_df['Date'].dt.month.isin([1, 7]))  # Hanya bulan Januari dan Juli
] """


# Buat plot interaktif
fig = go.Figure()

# Plot data aktual dan prediksi sebagai spline
fig.add_trace(go.Scatter(
    x=result_df['Date'], 
    y=result_df['Actual'],
    name='Close:', 
    line=dict(color='blue', shape='spline')  # Menggunakan spline
))
fig.add_trace(go.Scatter(
    x=result_df['Date'], 
    y=result_df['Predicted'], 
    mode='lines', 
    name='Prediction(Close):', 
    line=dict(color='green', shape='spline')  # Menggunakan spline
))

# Layout dan penataan grafik
fig.update_layout(
    title='Actual vs Predicted Stock Prices',
    xaxis_title='Date',
    yaxis_title='Close,Prediction Close',
    showlegend=True,
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=1.5 
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=1.5
    )
)

# Menampilkan grafik
fig.show()