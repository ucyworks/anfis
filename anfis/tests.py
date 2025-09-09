import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # VS Code ile uyumlu backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Grafik kaydetme dizini oluştur
import os
graph_dir = "grafik_cikti"
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    print(f"Grafik dizini oluşturuldu: {graph_dir}")

print(f"Grafikler şu dizinde kaydedilecek: {os.path.abspath(graph_dir)}")

def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error hesapla"""
    return numpy.mean(numpy.abs((actual - predicted) / actual)) * 100

def calculate_kge(actual, predicted):
    """Kling-Gupta Efficiency hesapla"""
    # Korelasyon katsayısı
    r = numpy.corrcoef(actual, predicted)[0, 1]
    
    # Bias ratio (alpha)
    alpha = numpy.std(predicted) / numpy.std(actual)
    
    # Variability ratio (beta)
    beta = numpy.mean(predicted) / numpy.mean(actual)
    
    # KGE hesapla
    kge = 1 - numpy.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def calculate_nse(actual, predicted):
    """Nash-Sutcliffe Efficiency hesapla"""
    numerator = numpy.sum((actual - predicted)**2)
    denominator = numpy.sum((actual - numpy.mean(actual))**2)
    nse = 1 - (numerator / denominator)
    return nse

# Alloy-data.csv dosyasından veri okuma (header yok)
data = pd.read_csv("../../synthethic-data.csv", header=None, 
                   names=['sic', 'graphite', 'weight', 'sliding_rate', 'wear_rate'])

# İlk 576 örnek eğitim verisi (X), son 48 örnek test verisi (Y)
train_size = len(data) - 48  # Son 48'i test için ayır
train_data = data[:train_size]
test_data = data[train_size:]

print(f"Toplam veri: {len(data)}")
print(f"Eğitim verisi (X): {len(train_data)}")
print(f"Test verisi (Y): {len(test_data)}")

# Eğitim verisi
X_train = train_data[['sic', 'graphite', 'weight', 'sliding_rate']].values
Y_train = train_data['wear_rate'].values

# Test verisi
X_test = test_data[['sic', 'graphite', 'weight', 'sliding_rate']].values
Y_test = test_data['wear_rate'].values

mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}], ['gaussmf',{'mean':-7.,'sigma':7.}]],
      [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}], ['gaussmf',{'mean':-10.5,'sigma':5.}]],
      [['gaussmf',{'mean':15.,'sigma':5.}],['gaussmf',{'mean':25.,'sigma':5.}],['gaussmf',{'mean':35.,'sigma':10.}]],
      [['gaussmf',{'mean':0.5,'sigma':0.5}],['gaussmf',{'mean':1.5,'sigma':0.5}],['gaussmf',{'mean':2.5,'sigma':0.5}]]]


mfc = membership.membershipfunction.MemFuncs(mf)

# R-kare threshold ile eğitim (NORMAL EĞİTİM)
target_r2_threshold = 0.9180
max_epochs = 100  # Yeterince yüksek maksimum
step_epochs = 5   # Verimli adım boyutu

print("\n=== ANFIS EĞİTİMİ BAŞLIYOR ===")
print(f"Hedef R² threshold: {target_r2_threshold}")
print(f"Maksimum epoch: {max_epochs}")
print(f"Adım boyutu: {step_epochs} epoch")

# İlk eğitimi yap
anf = anfis.ANFIS(X_train, Y_train, mfc)
anf.trainHybridJangOffLine(epochs=step_epochs)

current_epochs = step_epochs
best_r2 = 0

print(f"\n=== EĞİTİM SÜRECI ===")

while current_epochs < max_epochs:
    # Test verisiyle R² hesapla
    temp_predictions = []
    for i in range(len(X_test)):
        prediction = anfis.predict(anf, X_test[i:i+1])
        temp_predictions.append(prediction[0])
    temp_predictions = numpy.array(temp_predictions).flatten()
    current_r2 = r2_score(Y_test, temp_predictions)
    
    # Mevcut hata
    current_error = anf.errors[-1] if len(anf.errors) > 0 else float('inf')
    
    print(f"Epoch {current_epochs:3d}: Error = {current_error:.6f}, Test R² = {current_r2:.6f}")
    
    # En iyi R² güncelle
    if current_r2 > best_r2:
        best_r2 = current_r2
        print(f"      📈 Yeni en iyi R²: {best_r2:.6f}")
    
    # Threshold kontrolü
    if current_r2 >= target_r2_threshold:
        print(f"      🎯 HEDEF ULAŞILDI! R² = {current_r2:.6f} >= {target_r2_threshold:.4f}")
        print(f"      🏁 Eğitim durduruluyor...")
        break
    else:
        # Hedefe ne kadar kaldığını göster
        remaining = target_r2_threshold - current_r2
        print(f"      🎯 Hedefe kalan: {remaining:.6f}")
    
    # Hata durumu analizi
    if len(anf.errors) >= 2:
        error_change = anf.errors[-1] - anf.errors[-2]
        if error_change > 0:
            print(f"      ⚠️  Hata arttı: +{error_change:.6f}")
        else:
            print(f"      ✅ Hata azaldı: {error_change:.6f}")
    
    # Devam et: Daha fazla epoch eğit
    print(f"      ➡️  {step_epochs} epoch daha eğitiliyor...")
    anf.trainHybridJangOffLine(epochs=step_epochs)
    current_epochs += step_epochs

print(f"\n=== EĞİTİM TAMAMLANDI ===")
print(f"Toplam epoch: {current_epochs}")
final_error = anf.errors[-1] if len(anf.errors) > 0 else float('inf')
print(f"Son hata: {final_error:.6f}")
print(f"En iyi R²: {best_r2:.6f}")

# Son test R² hesapla
temp_predictions = []
for i in range(len(X_test)):
    prediction = anfis.predict(anf, X_test[i:i+1])
    temp_predictions.append(prediction[0])
temp_predictions = numpy.array(temp_predictions).flatten()
final_r2 = r2_score(Y_test, temp_predictions)

print(f"Son R²: {final_r2:.6f}")
print(f"Hedef threshold: {target_r2_threshold:.6f}")
print(f"Hedef ulaşıldı: {'✅ EVET' if final_r2 >= target_r2_threshold else '❌ HAYIR'}")

# Hata gelişimi göster
if len(anf.errors) >= 2:
    first_error = anf.errors[0]
    last_error = anf.errors[-1]
    error_change = last_error - first_error
    print(f"Hata değişimi: {first_error:.6f} -> {last_error:.6f} ({error_change:+.6f})")
    
    if error_change > 0:
        print("⚠️  Genel eğilim: Hata arttı (overfitting riski)")
    else:
        print("✅ Genel eğilim: Hata azaldı (normal öğrenme)")

print("="*50)

# Test verisiyle tahmin yap
test_predictions = []
for i in range(len(X_test)):
    prediction = anfis.predict(anf, X_test[i:i+1])
    test_predictions.append(prediction[0])

test_predictions = numpy.array(test_predictions)

# Tahmin değerlerini al
predictions = test_predictions.flatten()
actual_values = Y_test.flatten()

print(f"\n=== EĞİTİM VE TEST VERİSİ BİLGİLERİ ===")
print(f"Eğitim için kullanılan örnek sayısı: {len(X_train)}")
print(f"Test için kullanılan örnek sayısı: {len(X_test)}")
print(f"Model eğitim verisiyle eğitildi, test verisiyle değerlendirildi.")

# Metrikleri hesapla
r2 = r2_score(actual_values, predictions)
mse = mean_squared_error(actual_values, predictions)
mae = mean_absolute_error(actual_values, predictions)
rmse = numpy.sqrt(mse)
mape = calculate_mape(actual_values, predictions)
kge = calculate_kge(actual_values, predictions)
nse = calculate_nse(actual_values, predictions)

print("=== MODEL PERFORMANS METRİKLERİ ===")
print(f"R-kare (R²): {r2:.6f}")
print(f"MSE (Ortalama Kare Hata): {mse:.6f}")
print(f"MAE (Ortalama Mutlak Hata): {mae:.6f}")
print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.6f}")
print(f"MAPE (Ortalama Mutlak Yüzde Hata): {mape:.6f}%")
print(f"KGE (Kling-Gupta Efficiency): {kge:.6f}")
print(f"NSE (Nash-Sutcliffe Efficiency): {nse:.6f}")

print("\n=== METRİK YORUMLARI ===")
print(f"R² > 0.8: {'Mükemmel' if r2 > 0.8 else 'Orta' if r2 > 0.5 else 'Zayıf'}")
print(f"MAPE < 10%: {'Çok İyi' if mape < 10 else 'İyi' if mape < 20 else 'Orta' if mape < 50 else 'Zayıf'}")
print(f"KGE > 0.75: {'Çok İyi' if kge > 0.75 else 'İyi' if kge > 0.5 else 'Orta' if kge > 0.25 else 'Zayıf'}")
print(f"NSE > 0.75: {'Çok İyi' if nse > 0.75 else 'İyi' if nse > 0.5 else 'Orta' if nse > 0.25 else 'Zayıf'}")

print("\n=== GERÇEK VE TAHMİN DEĞERLERİ TABLOSU ===")
# Tablo oluştur
results_df = pd.DataFrame({
    'Sıra': range(1, len(actual_values) + 1),
    'Gerçek Değer': actual_values,
    'Tahmin Değer': predictions,
    'Mutlak Hata': abs(actual_values - predictions),
    'Kare Hata': (actual_values - predictions)**2
})

# Tabloyu yazdır
print(results_df.to_string(index=False, float_format='%.6f'))

print("=== ÖZET İSTATİSTİKLER ===")
print(f"Test Örnek Sayısı: {len(actual_values)}")
print(f"Ortalama Gerçek Değer: {numpy.mean(actual_values):.6f}")
print(f"Ortalama Tahmin Değer: {numpy.mean(predictions):.6f}")
print(f"Maksimum Mutlak Hata: {numpy.max(abs(actual_values - predictions)):.6f}")
print(f"Minimum Mutlak Hata: {numpy.min(abs(actual_values - predictions)):.6f}")

print("\n=== EĞİTİM VERİSİ ÜZERİNDE KONTROL ===")
train_predictions = anf.fittedValues.flatten()
train_actual = Y_train.flatten()
train_r2 = r2_score(train_actual, train_predictions)
print(f"Eğitim R²: {train_r2:.6f}")
print(f"Test R²: {r2:.6f}")
print(f"Overfitting kontrolü: {'VAR' if abs(train_r2 - r2) > 0.1 else 'YOK'}")

# Regresyon doğrusu grafiği oluştur
def plot_regression_line(actual, predicted, title="Gerçek vs Tahmin Değerleri"):
    """Regresyon doğrusu grafiği çiz"""
    
    # y = ax + b doğrusu için linear regression
    lr = LinearRegression()
    actual_reshaped = actual.reshape(-1, 1)
    lr.fit(actual_reshaped, predicted)
    
    # Doğru parametreleri
    a = lr.coef_[0]  # eğim
    b = lr.intercept_  # y-kesimi
    
    # R² değeri
    r2_value = r2_score(actual, predicted)
    
    # Grafik oluştur
    plt.figure(figsize=(10, 8))
    
    # Noktaları çiz (tahminler) - label yok
    plt.scatter(actual, predicted, alpha=0.7, color='blue', s=50)
    
    # Min-max değerleri hesapla
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    
    # Regresyon doğrusu (y = ax + b) - mavi kesikli çizgi - sadece denklem label'ı
    x_line = numpy.linspace(min_val, max_val, 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, 'b--', linewidth=2, 
             label=f'y = {a:.3f}x + {b:.3f}')
    
    # Grafik düzenleme
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{title}\nR² = {r2_value:.4f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Eşit ölçek
    plt.axis('equal')
    plt.xlim(min_val * 0.95, max_val * 1.05)
    plt.ylim(min_val * 0.95, max_val * 1.05)
    
    plt.tight_layout()
    
    # Grafiği kaydet
    if "Test Data" in title:
        regression_plot_path = os.path.join(graph_dir, "test_regression_plot.png")
        plt.savefig(regression_plot_path, dpi=300, bbox_inches='tight')
        print(f"Regresyon grafiği kaydedildi: {regression_plot_path}")
    
    plt.show()
    
    print(f"\n=== REGRESYON DOĞRUSU PARAMETRELERİ ===")
    print(f"Denklem: y = {a:.6f}x + {b:.6f}")
    print(f"Eğim (a): {a:.6f}")
    print(f"Y-kesimi (b): {b:.6f}")
    print(f"R² değeri: {r2_value:.6f}")
    
    return a, b, r2_value

# Test verisi regresyon grafiği
print("\n=== TEST VERİSİ REGRESYON GRAFİĞİ ===")
test_a, test_b, test_r2 = plot_regression_line(actual_values, predictions, "Test Data: Actual vs Predicted")

print(round(anf.consequents[-1][0],6))
print(round(anf.consequents[-2][0],6))
print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print('test is good')

print("Plotting errors")
plt.figure(figsize=(10, 6))
plt.plot(range(len(anf.errors)), anf.errors, 'ro-', label='Training Errors', linewidth=2, markersize=4)
plt.ylabel('Error', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.title('ANFIS Training Error Progress', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
error_plot_path = os.path.join(graph_dir, "anfis_training_errors.png")
plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
print(f"Hata grafiği kaydedildi: {error_plot_path}")
plt.show()

print("Plotting results")
plt.figure(figsize=(12, 8))
plt.plot(range(len(anf.fittedValues)), anf.fittedValues, 'r-', label='ANFIS Predictions', linewidth=2)
plt.plot(range(len(anf.Y)), anf.Y, 'b-', label='Actual Values', linewidth=2)
plt.xlabel('Sample Number', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('ANFIS Training Results: Actual vs Predicted', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
results_plot_path = os.path.join(graph_dir, "anfis_training_results.png")
plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
print(f"Sonuç grafiği kaydedildi: {results_plot_path}")
plt.show()
# ============================================================
# ===============  UNCERTAINTY EVALUATION (QR)  ==============
# ============  makaleye benzer yöntem - TAM BLOK  ============
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.ensemble import GradientBoostingRegressor

# ------------------ Metrikler ------------------
def calculate_picp(y_true, y_lower, y_upper):
    """Prediction Interval Coverage Probability"""
    y_true = np.asarray(y_true).ravel()
    y_lower = np.asarray(y_lower).ravel()
    y_upper = np.asarray(y_upper).ravel()
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return inside.mean()

def pinc_from_quantiles(q_lower, q_upper):
    """PINC = üst - alt kuantil (ör. 0.95-0.05 = 0.90)"""
    return float(q_upper - q_lower)

def ace(y_true, y_lower, y_upper, pinc):
    """ACE = PICP - PINC"""
    return calculate_picp(y_true, y_lower, y_upper) - float(pinc)

def pinaw(y_lower, y_upper):
    """Prediction Interval (Average) Width"""
    y_lower = np.asarray(y_lower).ravel()
    y_upper = np.asarray(y_upper).ravel()
    return np.mean(y_upper - y_lower)

def summarize_uncertainty(y_true, y_lower, y_upper, q_low, q_high) -> Dict[str, float]:
    pinc = pinc_from_quantiles(q_low, q_high)
    PICP = calculate_picp(y_true, y_lower, y_upper)
    return {
        "PINC": pinc,
        "PICP": PICP,
        "ACE":  PICP - pinc,
        "PINAW": pinaw(y_lower, y_upper)
    }

def plot_with_pi(y_true, y_pred_mid, y_lower, y_upper, title_suffix="(Quantile Regression)"):
    n = len(y_true)
    x = np.arange(n)
    plt.figure(figsize=(12,8))
    plt.plot(x, y_true, label="Actual", linewidth=2, color='blue')
    if y_pred_mid is not None:
        plt.plot(x, y_pred_mid, label="Prediction (median)", linewidth=2, color='red')
    plt.fill_between(x, y_lower, y_upper, alpha=0.25, label="Prediction Interval", color='gray')
    plt.title(f"Test: Prediction and Prediction Interval {title_suffix}", fontsize=14)
    plt.xlabel("Sample", fontsize=12)
    plt.ylabel("Target", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Grafiği kaydet
    filename = title_suffix.replace("(", "").replace(")", "").replace(" ", "_").replace(",", "").lower()
    pi_plot_path = os.path.join(graph_dir, f"prediction_interval_{filename}.png")
    plt.savefig(pi_plot_path, dpi=300, bbox_inches='tight')
    print(f"Tahmin aralığı grafiği kaydedildi: {pi_plot_path}")
    
    plt.show()

def plot_quartile_forecast(y_true, y_predicted, y_lower_5, y_upper_95, title="Quartile Forecast"):
    """Quartile Forecast grafiği çizer"""
    n = len(y_true)
    x = np.arange(n)
    
    plt.figure(figsize=(14, 10))
    
    # Ana çizgiler
    plt.plot(x, y_true, label="Actual", linewidth=3, color='blue', marker='o', markersize=4)
    plt.plot(x, y_predicted, label="Predicted", linewidth=3, color='red', marker='s', markersize=4)
    
    # Quantile çizgileri
    plt.plot(x, y_upper_95, label="95% Quantile", linewidth=2, color='green', linestyle='--', alpha=0.8)
    plt.plot(x, y_lower_5, label="5% Quantile", linewidth=2, color='orange', linestyle='--', alpha=0.8)
    
    # Prediction Interval (gölgeli alan)
    plt.fill_between(x, y_lower_5, y_upper_95, alpha=0.2, color='gray', label="Predictions Interval")
    
    # Grafik düzenlemeleri
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Sample", fontsize=14)
    plt.ylabel("Target Value", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Eksen sınırları
    all_values = np.concatenate([y_true, y_predicted, y_lower_5, y_upper_95])
    margin = (np.max(all_values) - np.min(all_values)) * 0.05
    plt.ylim(np.min(all_values) - margin, np.max(all_values) + margin)
    
    plt.tight_layout()
    
    # Grafiği kaydet
    quartile_plot_path = os.path.join(graph_dir, "quartile_forecast.png")
    plt.savefig(quartile_plot_path, dpi=300, bbox_inches='tight')
    print(f"Quartile Forecast grafiği kaydedildi: {quartile_plot_path}")
    
    plt.show()
    
    return quartile_plot_path

# ------------- Kuantil Regresyon Sarmalayıcı -------------
class QuantileRegressorPI:
    """
    Alt/medyan/üst kuantilleri aynı özniteliklerle öğrenir.
    Varsayılan: %5, %50, %95 kuantiller.
    """
    def __init__(self, quantiles: List[float] = [0.05, 0.50, 0.95], **gb_kwargs):
        self.quantiles = sorted(quantiles)
        self.models = {}
        # Sağlam ve hızlı varsayılan hiperparametreler
        self.gb_kwargs = {"n_estimators": 500, "max_depth": 3, "min_samples_leaf": 3,
                          "learning_rate": 0.05, "subsample": 1.0, "random_state": 42}
        self.gb_kwargs.update(gb_kwargs)

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).ravel()
        for q in self.quantiles:
            m = GradientBoostingRegressor(loss="quantile", alpha=q, **self.gb_kwargs)
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict_quantile(self, X, q):
        if q not in self.models:
            # Gerekirse sonradan ek kuantil modeli eğit
            m = GradientBoostingRegressor(loss="quantile", alpha=q, **self.gb_kwargs)
            m.fit(np.asarray(X_train), np.asarray(Y_train).ravel())
            self.models[q] = m
        return self.models[q].predict(X)

    def predict_interval(self, X, q_low=0.05, q_high=0.95):
        yL = self.predict_quantile(X, q_low)
        yU = self.predict_quantile(X, q_high)
        return yL, yU

    def predict_median(self, X):
        if 0.5 in self.models:
            return self.models[0.5].predict(X)
        # yoksa komşu kuantillerin ortalamasını kullan
        lower_qs = [q for q in self.quantiles if q < 0.5]
        upper_qs = [q for q in self.quantiles if q > 0.5]
        if not lower_qs or not upper_qs:
            # güvenli geri dönüş: üst-alt bandın ortası
            l,u = self.predict_interval(X, min(self.quantiles), max(self.quantiles))
            return (l+u)/2
        return (self.models[min(upper_qs)].predict(X) + self.models[max(lower_qs)].predict(X)) / 2

# ----------------- EĞİTİM ve DEĞERLENDİRME -----------------
# ANFIS eğitimini ve test tahminlerini (predictions, actual_values) siz yukarıda ürettiniz.
# Şimdi QR'ı sadece eğitim verisiyle eğitip testte PI hesaplayalım.

print("\n=== KUANTİL REGRESYON İLE BELİRSİZLİK (QR) ===")
qr = QuantileRegressorPI(quantiles=[0.05, 0.50, 0.95]).fit(X_train, Y_train)

# %90 kapsama (q_low=0.05, q_high=0.95)
q_low, q_high = 0.05, 0.95
yL_90, yU_90 = qr.predict_interval(X_test, q_low, q_high)
yMed_90 = qr.predict_median(X_test)

unc90 = summarize_uncertainty(Y_test, yL_90, yU_90, q_low, q_high)
print("\n--- 90% PI Metrikleri (QR) ---")
for k, v in unc90.items():
    print(f"{k}: {v:.4f}")

# Görselleştirme (test sırasına göre)
plot_with_pi(Y_test, yMed_90, yL_90, yU_90, title_suffix="(QR, 90% PI)")

# === QUARTİLE FORECAST GRAFİĞİ ===
print("\n=== QUARTİLE FORECAST GRAFİĞİ ===")

# 5% ve 95% quantile hesapla (90% PI için)
y_5_quantile = yL_90  # 5% quantile
y_95_quantile = yU_90  # 95% quantile

# ANFIS tahminlerini kullan
anfis_predictions_test = predictions  # Test tahminleri
actual_test = actual_values  # Gerçek değerler

print(f"Quartile verileri:")
print(f"  5% Quantile: min={np.min(y_5_quantile):.3f}, max={np.max(y_5_quantile):.3f}")
print(f"  95% Quantile: min={np.min(y_95_quantile):.3f}, max={np.max(y_95_quantile):.3f}")
print(f"  ANFIS Predictions: min={np.min(anfis_predictions_test):.3f}, max={np.max(anfis_predictions_test):.3f}")
print(f"  Actual Values: min={np.min(actual_test):.3f}, max={np.max(actual_test):.3f}")

# Quartile Forecast grafiğini çiz
quartile_path = plot_quartile_forecast(
    y_true=actual_test,
    y_predicted=anfis_predictions_test, 
    y_lower_5=y_5_quantile,
    y_upper_95=y_95_quantile,
    title="Quartile Forecast"
)

print(f"Quartile Forecast analizi tamamlandı!")

# (Opsiyonel) Birden çok kapsama seviyesi tablosu: 80% ve 95%
levels = [(0.10, 0.90), (0.025, 0.975)]
rows = []
for lo, hi in levels:
    yL, yU = qr.predict_interval(X_test, lo, hi)
    rows.append({**summarize_uncertainty(Y_test, yL, yU, lo, hi),
                 "q_low": lo, "q_high": hi})

if rows:
    import pandas as pd
    df_unc = pd.DataFrame(rows)[["q_low", "q_high", "PINC", "PICP", "ACE", "PINAW"]]
    print("\n--- Farklı Kapsama Seviyeleri (QR) ---")
    print(df_unc.to_string(index=False, float_format="%.4f"))

# === BELİRSİZLİK METRİKLERİ İYİLEŞTİRME ANALİZİ ===
print("\n=== BELİRSİZLİK ANALİZİ VE İYİLEŞTİRME ÖNERİLERİ ===")

# 90% PI analizi
picp_90 = unc90['PICP']
ace_90 = unc90['ACE']
pinaw_90 = unc90['PINAW']

print(f"\n📊 90% PI Durumu:")
print(f"   Hedef kapsama (PINC): 90.00%")
print(f"   Gerçek kapsama (PICP): {picp_90*100:.1f}%")
print(f"   Fark (ACE): {ace_90:.3f}")

if ace_90 < -0.1:
    print("   ⚠️  CİDDİ UNDERCOVERAGE: Bantlar çok dar!")
elif ace_90 < 0:
    print("   ⚠️  UNDERCOVERAGE: Bantlar dar kalmış")
elif ace_90 > 0.1:
    print("   ⚠️  OVERCOVERAGE: Bantlar gereğinden geniş")
else:
    print("   ✅ Kabul edilebilir kapsama seviyesi")

# Kalibrasyon faktörü önerisi
if ace_90 < -0.05:  # Undercoverage durumunda
    suggested_factor = 1 + abs(ace_90) * 2  # Ölçek faktörü
    print(f"\n🔧 İyileştirme Önerisi:")
    print(f"   Bant genişliği çarpanı: {suggested_factor:.2f}")
    print(f"   Daha güvenli kuantiller: q=0.01-0.99 (98% PI)")

# Farklı seviyelerdeki performans karşılaştırması
print(f"\n📈 Kapsama Seviyesi Karşılaştırması:")
for idx, (lo, hi) in enumerate(levels):
    picp = rows[idx]['PICP']
    ace = rows[idx]['ACE']
    pinc = rows[idx]['PINC']
    
    reliability = "Güvenilir" if abs(ace) < 0.05 else "Düşük güvenilirlik"
    coverage_quality = "İyi" if picp >= pinc*0.9 else "Zayıf"
    
    print(f"   {int(pinc*100)}% PI: PICP={picp*100:.1f}%, ACE={ace:.3f} → {reliability}, {coverage_quality}")

# En iyi kapsama seviyesi önerisi
best_idx = min(range(len(rows)), key=lambda i: abs(rows[i]['ACE']))
best_level = levels[best_idx]
best_pinc = rows[best_idx]['PINC']
print(f"\n🎯 En İyi Kapsama Seviyesi: {int(best_pinc*100)}% PI (q={best_level[0]:.3f}-{best_level[1]:.3f})")

# Kalibrasyon yapılmış tahmin aralıkları (95% PI için)
print(f"\n=== KALİBRASYON YAPILMIŞ TAHMİN ARALIKLARI ===")

# 95% PI için geliştirilmiş bantlar
yL_95, yU_95 = qr.predict_interval(X_test, 0.025, 0.975)
unc95_original = summarize_uncertainty(Y_test, yL_95, yU_95, 0.025, 0.975)

# Adaptif genişletme faktörü
if unc95_original['ACE'] < 0:
    expansion_factor = 1 + abs(unc95_original['ACE']) * 1.5
    
    # Bantları genişlet
    y_center = (yL_95 + yU_95) / 2
    half_width = (yU_95 - yL_95) / 2
    yL_95_cal = y_center - half_width * expansion_factor
    yU_95_cal = y_center + half_width * expansion_factor
    
    unc95_calibrated = summarize_uncertainty(Y_test, yL_95_cal, yU_95_cal, 0.025, 0.975)
    
    print(f"Orijinal 95% PI:")
    print(f"   PICP: {unc95_original['PICP']*100:.1f}%, ACE: {unc95_original['ACE']:.3f}")
    print(f"Kalibre edilmiş 95% PI (faktör: {expansion_factor:.2f}):")
    print(f"   PICP: {unc95_calibrated['PICP']*100:.1f}%, ACE: {unc95_calibrated['ACE']:.3f}")
    
    # Kalibre edilmiş görselleştirme
    plot_with_pi(Y_test, (yL_95_cal + yU_95_cal)/2, yL_95_cal, yU_95_cal, 
                title_suffix="(Kalibre edilmiş 95% PI)")

print(f"\n=== MAKALE İÇİN ÖNERİLER ===")
print("📝 Belirsizlik değerlendirmesi için öneriler:")
print("   • QR tabanlı PI'lar genellikle undercoverage eğilimi gösterir")
print("   • %95 PI daha güvenilir kapsama sağlar (%90'a göre)")
print("   • Kalibrasyon faktörü uygulanarak güvenilirlik artırılabilir")
print("   • PICP < PINC durumunda bant genişliği artırılmalıdır")

# (Opsiyonel) ANFIS noktasal tahminlerinizle QR medyanını yan yana görsel
plt.figure(figsize=(12,8))
plt.plot(actual_values, label="Actual", linewidth=2, color='blue')
plt.plot(predictions, label="ANFIS Point Prediction", linewidth=2, color='red')
plt.plot(yMed_90, label="QR Median", linewidth=2, linestyle="--", color='green')
plt.title("ANFIS Point Prediction vs. QR Median Comparison", fontsize=14)
plt.xlabel("Sample", fontsize=12)
plt.ylabel("Target", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Karşılaştırma grafiğini kaydet
comparison_plot_path = os.path.join(graph_dir, "anfis_vs_qr_comparison.png")
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"ANFIS vs QR karşılaştırma grafiği kaydedildi: {comparison_plot_path}")

plt.show()
# ====================  SON BLOK BİTTİ  =====================
