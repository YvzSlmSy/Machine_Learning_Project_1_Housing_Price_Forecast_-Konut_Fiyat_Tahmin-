import os
import sys

# Proje kök dizinini ekleyelim
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.preprocess import preprocess_data
import numpy as np

def train_model(X_train, y_train):
    """Modeli eğitim verisiyle eğitme."""
    model = LinearRegression()
    model.fit(X_train, y_train)  # Modeli eğitme
    return model

def evaluate_model(model, X_test, y_test):
    """Modeli test verisiyle değerlendirme."""
    y_pred = model.predict(X_test)  # Tahminler
    mse = mean_squared_error(y_test, y_pred)  # Ortalama kare hata
    rmse = np.sqrt(mse)  # Kök ortalama kare hata (RMSE)
    print("Model Değerlendirmesi:")
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    # Veriyi yükle ve işle
    from src.load_data import load_housing_data
    df = load_housing_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Modeli eğit
    model = train_model(X_train, y_train)

    # Modeli değerlendir
    evaluate_model(model, X_test, y_test)
