import os
import sys

# Proje kök dizinini ekleyelim
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.load_data import load_housing_data

def preprocess_data(df):
    """Veriyi eğitim ve test setlerine ayırma, normalizasyon."""
    
    # Özellikler (X) ve hedef değişkeni (y) ayırma
    X = df.drop('Target', axis=1)  # 'Target' dışındaki tüm özellikler
    y = df['Target']  # 'Target' olan hedef değişken

    # Veriyi eğitim (%80) ve test (%20) olarak ayıralım
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Veriyi normalize edelim (özellikleri aynı ölçeğe getirelim)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    df = load_housing_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Eğitim verisi boyutu:", X_train.shape)
    print("Test verisi boyutu:", X_test.shape)
