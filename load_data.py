


import pandas as pd
from sklearn.datasets import fetch_california_housing #Veri setini import ediyoruz

def load_housing_data():
    """California Housing veri setini yükleyip Pandas DataFrame olarak döndürür."""
    
    # California konut fiyatları veri setini yükleyelim
    data = fetch_california_housing()
   
    # Veriyi Pandas DataFrame formatına çevirelim
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Hedef değişkeni ekleyelim (Ev fiyatları)
    df['Target'] = data.target 
    
    return df

if __name__ == "__main__":
    df = load_housing_data()
    print(df.head())  # İlk 5 satırı ekrana yazdır
