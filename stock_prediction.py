import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def load_data(file_path):
    try:
        data = pd.read_csv('dataset/saham_new.csv')  # Ganti dengan nama filemu
        print("Data berhasil di load")
        return data
    except FileNotFoundError as e:
        print("File tidak ditemukan", e)
        return None

def explore_data(data,name="Dataset"):
    # Mengatur pandas untuk menampilkan semua baris
    pd.set_option('display.max_rows', None)  # Menampilkan semua baris
    pd.set_option('display.max_columns', None)  # Menampilkan semua kolom (opsional)
    pd.set_option('display.width', None)  # Menyesuaikan lebar tampilan

    print(data.head())
    print(f"\nInformasi {name}:")
    print(data.info())
    print("\nStatistik Deskriptif:")
    print(data.describe())
    print("\nPengecekan Missing Values:")
    print(data.isnull().sum())

def plot_correlation_matrix(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    print(numeric_data.corr())
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

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

def split_data_training_testign(data):
    x = data[['Open','High','Low','Volume']]
    y = data[['Close']]

    # split data 90% training 10% testing
    data_training,data_testing, y_train,y_test = train_test_split(x,y, test_size=0.1, random_state=42, shuffle=False)

    print("\nJumlah data training : ", len(data_training))
    print("Data Training", data_training.head())


    print("\nJumlah data testing : ", len(data_testing))
    print("Data Testing",data_testing.head())

    return data_training,data_testing,y_test,y_train

def training_model(data_training,y_train):
    model = LinearRegression()

    model.fit(data_training, y_train)

    return model

def testing(data,data_testing,y_test,model):
    y_prediction = model.predict(data_testing)
    # hasil prediksi
    print("===================================================================================")
    result_df = data_testing.copy()
    result_df['Date'] = data.loc[data_testing.index, 'Date']
    result_df['Actual'] = y_test
    result_df['Predicted'] = y_prediction.round()
    result_df['Index'] = range(1, len(result_df) + 1)
    print("\nJumlah Data Prediksi :", len(result_df))
    print("Hasil Prediksi:")
    print("===================================================================================")
    # print(result_df)
    print(result_df[['Index', 'Date', 'Open', 'High', 'Low', 'Volume', 'Actual', 'Predicted']])
    print("===================================================================================")

    print("===================================================================================")
    return y_prediction,result_df

def model_evaluation(y_test,y_prediction):
    rmse = np.sqrt(root_mean_squared_error(y_test,y_prediction)) #RMSE
    mse = mean_squared_error(y_test, y_prediction)  # Mean Squared Error
    mae = mean_absolute_error(y_test, y_prediction)  # Mean Absolute Error
    r2 = r2_score(y_test, y_prediction)  # R-squared
    print('\nRMSE', round(rmse, 2))
    print('MSE',round( mse, 2))
    print('MAE',round( mae, 2))
    print('R-Squared', r2)
    print("===================================================================================")


#print("Rentang data :",result_df[(result_df['Date'] >= '2020-01-01') & (result_df['Date'] <= '2020-01-31')])



def plot_grafik(result_df=None):
    # tampilkan grafik actual dan prediksi
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

def main():
    data = load_data('dataset/saham_new.csv')
    if data is None:
        return
    else:
        explore_data(data)
        data_training, data_testing, y_test,y_train = split_data_training_testign(data)
        model = training_model(data_training, y_train)
        y_prediction, result_df = testing(data, data_testing, y_test, model)
        model_evaluation(y_test, y_prediction)
        plot_grafik(result_df)
        plot_correlation_matrix(data)

if __name__ == '__main__':
    main()