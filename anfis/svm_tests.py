import numpy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # VS Code ile uyumlu backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

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

def plot_3d_surface_modeling(X_train, Y_train, svm_model, scaler, title="SVM 3D Surface Modeling", surface_type="surface"):
    """3D Surface Modeling grafiği çiz - iki input bir output
    
    surface_type seçenekleri:
    - 'surface': Düz yüzey modeli (varsayılan)
    - 'wireframe': Tel kafes modeli
    - 'contour3d': 3D contour
    - 'trisurf': Triangulated surface
    - 'scatter3d': 3D scatter plot
    - 'combo': Birleşik model (surface + wireframe)
    """
    # İlk iki feature'ı kullan (sic ve graphite)
    feature_names = ['SiC (%)', 'Graphite (%)']
    
    # Grid oluştur
    x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
    x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()
    
    # Genişlet
    x1_range = x1_max - x1_min
    x2_range = x2_max - x2_min
    x1_min -= x1_range * 0.1
    x1_max += x1_range * 0.1
    x2_min -= x2_range * 0.1
    x2_max += x2_range * 0.1
    
    # Grid noktaları
    grid_density = 60 if surface_type in ['wireframe', 'trisurf'] else 50
    xx1, xx2 = numpy.meshgrid(
        numpy.linspace(x1_min, x1_max, grid_density),
        numpy.linspace(x2_min, x2_max, grid_density)
    )
    
    # Diğer feature'ları ortalama değerlerle doldur
    mean_weight = numpy.mean(X_train[:, 2])
    mean_sliding = numpy.mean(X_train[:, 3])
    
    # Tahmin için grid hazırla
    grid_points = numpy.c_[
        xx1.ravel(), 
        xx2.ravel(),
        numpy.full(xx1.ravel().shape, mean_weight),
        numpy.full(xx1.ravel().shape, mean_sliding)
    ]
    
    # Tahmin yap
    Z = svm_model.predict(grid_points)
    Z = Z.reshape(xx1.shape)
    
    # 3D plot
    fig = plt.figure(figsize=(16, 12))
    
    # İlk subplot: Ana surface model
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Surface model tipine göre çizim
    if surface_type == "surface":
        surf = ax1.plot_surface(xx1, xx2, Z, alpha=0.7, cmap='viridis', 
                               linewidth=0, antialiased=True, edgecolor='none')
        model_name = "Smooth Surface Model"
        
    elif surface_type == "wireframe":
        surf = ax1.plot_wireframe(xx1, xx2, Z, alpha=0.8, color='blue', linewidth=0.8)
        # Wireframe için ek surface ekle (şeffaf)
        ax1.plot_surface(xx1, xx2, Z, alpha=0.3, cmap='coolwarm', antialiased=True)
        model_name = "Wireframe Model"
        
    elif surface_type == "contour3d":
        # 3D contour lines
        surf = ax1.contour3D(xx1, xx2, Z, levels=15, cmap='viridis', alpha=0.8)
        # Arka plan surface ekle
        ax1.plot_surface(xx1, xx2, Z, alpha=0.2, cmap='viridis')
        model_name = "3D Contour Model"
        
    elif surface_type == "trisurf":
        # Triangulated surface
        surf = ax1.plot_trisurf(xx1.ravel(), xx2.ravel(), Z.ravel(), 
                               cmap='plasma', alpha=0.7, antialiased=True)
        model_name = "Triangulated Surface"
        
    elif surface_type == "scatter3d":
        # 3D scatter with prediction surface
        ax1.plot_surface(xx1, xx2, Z, alpha=0.4, cmap='viridis')
        surf = ax1.scatter(xx1.ravel()[::5], xx2.ravel()[::5], Z.ravel()[::5], 
                          c=Z.ravel()[::5], cmap='plasma', s=30, alpha=0.8)
        model_name = "Scatter Surface Model"
        
    elif surface_type == "combo":
        # Kombinasyon: Surface + Wireframe
        surf1 = ax1.plot_surface(xx1, xx2, Z, alpha=0.6, cmap='viridis', 
                                linewidth=0, antialiased=True)
        surf2 = ax1.plot_wireframe(xx1, xx2, Z, alpha=0.4, color='black', linewidth=0.5)
        surf = surf1  # Colorbar için
        model_name = "Hybrid Surface Model"
        
    else:
        # Varsayılan: normal surface
        surf = ax1.plot_surface(xx1, xx2, Z, alpha=0.7, cmap='viridis', 
                               linewidth=0, antialiased=True)
        model_name = "Default Surface Model"
    
    # Gerçek veri noktalarını ekle
    scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], Y_train, 
                         c=Y_train, cmap='plasma', s=80, alpha=0.9, 
                         edgecolors='black', linewidth=1)
    
    ax1.set_xlabel(feature_names[0], fontsize=12, labelpad=10)
    ax1.set_ylabel(feature_names[1], fontsize=12, labelpad=10)
    ax1.set_zlabel('Wear Rate', fontsize=12, labelpad=10)
    ax1.set_title(f'{title}\n{model_name}', fontsize=14, pad=20)
    
    # Viewing angle ayarla
    ax1.view_init(elev=20, azim=45)
    
    # Colorbar ekle (eğer surf objesi varsa)
    try:
        if hasattr(surf, 'get_array'):
            plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=20, label='Predicted Wear Rate')
    except:
        pass
    
    # İkinci subplot: Contour plot
    ax2 = fig.add_subplot(122)
    
    # Contour plot
    contour = ax2.contourf(xx1, xx2, Z, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = ax2.contour(xx1, xx2, Z, levels=20, colors='black', alpha=0.4, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Gerçek veri noktalarını ekle
    scatter2 = ax2.scatter(X_train[:, 0], X_train[:, 1], 
                          c=Y_train, cmap='plasma', s=60, alpha=0.9, 
                          edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel(feature_names[0], fontsize=12)
    ax2.set_ylabel(feature_names[1], fontsize=12)
    ax2.set_title(f'{title}\nContour Map', fontsize=14)
    
    # Colorbar ekle
    plt.colorbar(contour, ax=ax2, label='Predicted Wear Rate')
    
    plt.tight_layout()
    
    # Kaydet
    filename = f"{graph_dir}/svm_3d_surface_modeling.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"3D Surface Modeling grafiği kaydedildi: {filename}")
    
    return fig

def plot_feature_importance_surface(X_train, Y_train, svm_model, scaler):
    """Feature çiftleri için 3D yüzey grafikleri"""
    feature_names = ['SiC (%)', 'Graphite (%)', 'Weight (g)', 'Sliding Rate (m/s)']
    feature_pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    pair_names = [
        ('SiC', 'Graphite'),
        ('SiC', 'Weight'), 
        ('Graphite', 'Weight'),
        ('Weight', 'Sliding Rate')
    ]
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, ((f1, f2), (name1, name2)) in enumerate(zip(feature_pairs, pair_names)):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        
        # Grid oluştur
        x1_min, x1_max = X_train[:, f1].min(), X_train[:, f1].max()
        x2_min, x2_max = X_train[:, f2].min(), X_train[:, f2].max()
        
        # Genişlet
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min
        x1_min -= x1_range * 0.1
        x1_max += x1_range * 0.1
        x2_min -= x2_range * 0.1
        x2_max += x2_range * 0.1
        
        xx1, xx2 = numpy.meshgrid(
            numpy.linspace(x1_min, x1_max, 30),
            numpy.linspace(x2_min, x2_max, 30)
        )
        
        # Diğer feature'ları ortalama ile doldur
        grid_points = numpy.zeros((xx1.ravel().shape[0], 4))
        grid_points[:, f1] = xx1.ravel()
        grid_points[:, f2] = xx2.ravel()
        
        # Diğer feature'lar için ortalama kullan
        for i in range(4):
            if i != f1 and i != f2:
                grid_points[:, i] = numpy.mean(X_train[:, i])
        
        # Tahmin yap
        Z = svm_model.predict(grid_points)
        Z = Z.reshape(xx1.shape)
        
        # Surface plot
        surf = ax.plot_surface(xx1, xx2, Z, alpha=0.7, cmap='coolwarm')
        
        # Gerçek veri noktaları
        ax.scatter(X_train[:, f1], X_train[:, f2], Y_train, 
                  c=Y_train, cmap='plasma', s=30, alpha=0.8)
        
        ax.set_xlabel(feature_names[f1], fontsize=10)
        ax.set_ylabel(feature_names[f2], fontsize=10)
        ax.set_zlabel('Wear Rate', fontsize=10)
        ax.set_title(f'{name1} vs {name2}', fontsize=12)
    
    plt.suptitle('SVM Feature Interaction Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Kaydet
    filename = f"{graph_dir}/svm_feature_interactions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Feature Interactions grafiği kaydedildi: {filename}")
    
    return fig

# Veri okuma
data = pd.read_csv("../../synthethic-data.csv", header=None, 
                   names=['sic', 'graphite', 'weight', 'sliding_rate', 'wear_rate'])

# İlk 576 örnek eğitim verisi, son 48 örnek test verisi
train_size = len(data) - 48
train_data = data[:train_size]
test_data = data[train_size:]

print(f"Toplam veri: {len(data)}")
print(f"Eğitim verisi: {len(train_data)}")
print(f"Test verisi: {len(test_data)}")

# Eğitim ve test verisi
X_train = train_data[['sic', 'graphite', 'weight', 'sliding_rate']].values
Y_train = train_data['wear_rate'].values
X_test = test_data[['sic', 'graphite', 'weight', 'sliding_rate']].values
Y_test = test_data['wear_rate'].values

print("\n=== SVM EĞİTİMİ BAŞLIYOR ===")

# SVM modelini ayarla
target_r2_threshold = 0.9180
print(f"Hedef R² threshold: {target_r2_threshold}")

# Grid Search ile en iyi parametreleri bul
print("En iyi hyperparametreler aranıyor...")
param_grid = {
    'svr__C': [0.1, 1, 10, 100, 1000],
    'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'svr__epsilon': [0.01, 0.1, 0.2]
}

# Pipeline oluştur (Scaling + SVM)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

# Grid Search
grid_search = GridSearchCV(
    pipe, param_grid, 
    cv=5, 
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Eğit
grid_search.fit(X_train, Y_train)

# En iyi modeli al
best_svm = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\n=== EN İYİ SVM PARAMETRELERİ ===")
print(f"En iyi CV R² skoru: {best_score:.6f}")
print("En iyi parametreler:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Test verisiyle tahmin yap
test_predictions = best_svm.predict(X_test)
train_predictions = best_svm.predict(X_train)

# Test R²'yi hesapla
test_r2 = r2_score(Y_test, test_predictions)
train_r2 = r2_score(Y_train, train_predictions)

print(f"\n=== SVM EĞİTİM SONUÇLARI ===")
print(f"Eğitim R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
print(f"Hedef threshold: {target_r2_threshold:.6f}")
print(f"Hedef ulaşıldı: {'✅ EVET' if test_r2 >= target_r2_threshold else '❌ HAYIR'}")

if test_r2 < target_r2_threshold:
    print(f"Hedefe kalan: {target_r2_threshold - test_r2:.6f}")

# Overfitting kontrolü
overfitting = train_r2 - test_r2
print(f"Overfitting kontrolü: {overfitting:.6f}")
if overfitting > 0.1:
    print("⚠️  OVERFITTING RİSKİ YÜKSEK!")
elif overfitting > 0.05:
    print("⚠️  Hafif overfitting")
else:
    print("✅ Overfitting riski düşük")

print("="*60)

# Tahmin değerlerini al
predictions = test_predictions.flatten()
actual_values = Y_test.flatten()

print(f"\n=== EĞİTİM VE TEST VERİSİ BİLGİLERİ ===")
print(f"Eğitim için kullanılan örnek sayısı: {len(X_train)}")
print(f"Test için kullanılan örnek sayısı: {len(X_test)}")
print(f"Model eğitim verisiyle eğitildi, test verisiyle değerlendirildi.")

# Metrikleri hesapla
r2 = test_r2
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

print(f"\n=== METRİK YORUMLARI ===")
if r2 > 0.9:
    print("R² > 0.9: Mükemmel")
elif r2 > 0.8:
    print("R² > 0.8: Çok İyi")
elif r2 > 0.7:
    print("R² > 0.7: İyi")
else:
    print("R² < 0.7: Zayıf")

if mape < 10:
    print("MAPE < 10%: Mükemmel")
elif mape < 20:
    print("MAPE < 20%: İyi")
elif mape < 50:
    print("MAPE < 50%: Orta")
else:
    print("MAPE > 50%: Zayıf")

if kge > 0.75:
    print("KGE > 0.75: Çok İyi")
elif kge > 0.5:
    print("KGE > 0.5: İyi")
else:
    print("KGE < 0.5: Zayıf")

if nse > 0.75:
    print("NSE > 0.75: Çok İyi")
elif nse > 0.5:
    print("NSE > 0.5: İyi")
else:
    print("NSE < 0.5: Zayıf")

# Gerçek ve tahmin değerleri tablosu
print(f"\n=== GERÇEK VE TAHMİN DEĞERLERİ TABLOSU ===")
print(f"{'Sıra':>4s}  {'Gerçek Değer':>12s}  {'Tahmin Değer':>12s}  {'Mutlak Hata':>11s}  {'Kare Hata':>9s}")

for i in range(len(actual_values)):
    actual = actual_values[i]
    predicted = predictions[i]
    abs_error = abs(actual - predicted)
    squared_error = (actual - predicted)**2
    print(f"{i+1:4d}      {actual:8.6f}      {predicted:8.6f}     {abs_error:8.6f}   {squared_error:8.6f}")

# Özet istatistikler
print(f"\n=== ÖZET İSTATİSTİKLER ===")
print(f"Test Örnek Sayısı: {len(actual_values)}")
print(f"Ortalama Gerçek Değer: {numpy.mean(actual_values):.6f}")
print(f"Ortalama Tahmin Değer: {numpy.mean(predictions):.6f}")
print(f"Maksimum Mutlak Hata: {numpy.max(numpy.abs(actual_values - predictions)):.6f}")
print(f"Minimum Mutlak Hata: {numpy.min(numpy.abs(actual_values - predictions)):.6f}")

print(f"\n=== EĞİTİM VERİSİ ÜZERİNDE KONTROL ===")
print(f"Eğitim R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
if train_r2 - test_r2 > 0.1:
    print("Overfitting kontrolü: VAR")
else:
    print("Overfitting kontrolü: YOK")

# Regresyon grafiği
print(f"\n=== TEST VERİSİ REGRESYON GRAFİĞİ ===")
plt.figure(figsize=(10, 8))

# Scatter plot
plt.scatter(actual_values, predictions, alpha=0.7, color='blue', s=50, edgecolors='black', linewidth=0.5)

# Perfect prediction line (45-degree line)
min_val = min(numpy.min(actual_values), numpy.min(predictions))
max_val = max(numpy.max(actual_values), numpy.max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

# Regression line
regressor = LinearRegression()
regressor.fit(actual_values.reshape(-1, 1), predictions)
line_predictions = regressor.predict(actual_values.reshape(-1, 1))
plt.plot(actual_values, line_predictions, 'g-', linewidth=2, label='Regression Line')

plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title(f'SVM Test Data Regression Plot\nR² = {r2:.4f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Kaydet
regression_filename = f"{graph_dir}/svm_test_regression_plot.png"
plt.savefig(regression_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Regresyon grafiği kaydedildi: {regression_filename}")

# Regresyon doğrusu parametreleri
slope = regressor.coef_[0]
intercept = regressor.intercept_
print(f"\n=== REGRESYON DOĞRUSU PARAMETRELERİ ===")
print(f"Denklem: y = {slope:.6f}x + {intercept:.6f}")
print(f"Eğim (a): {slope:.6f}")
print(f"Y-kesimi (b): {intercept:.6f}")
print(f"R² değeri: {r2:.6f}")

# 3D Surface Modeling grafiği çiz - Farklı tipler
print(f"\n=== 3D SURFACE MODELING ===")

# Farklı surface model tiplerini çiz
surface_types = [
    ("surface", "Smooth Surface Model"),
    ("wireframe", "Wireframe Model"), 
    ("combo", "Hybrid Surface Model"),
    ("contour3d", "3D Contour Model"),
    ("trisurf", "Triangulated Surface")
]

print("Farklı surface model tipleri oluşturuluyor...")
for i, (stype, description) in enumerate(surface_types):
    print(f"{i+1}. {description} oluşturuluyor...")
    plot_3d_surface_modeling(X_train, Y_train, best_svm, best_svm.named_steps['scaler'], 
                            title=f"SVM {description}", surface_type=stype)
    
    # Farklı isimlerle kaydet
    filename = f"{graph_dir}/svm_3d_surface_{stype}_modeling.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   -> {filename} kaydedildi!")
    plt.close()

print("Tüm surface model tipleri oluşturuldu!")

# Feature Interactions grafiği
print(f"\n=== FEATURE INTERACTIONS ANALYSIS ===")
plot_feature_importance_surface(X_train, Y_train, best_svm, best_svm.named_steps['scaler'])

# SVM Training Progress (Cross-validation scores)
print(f"\n=== SVM EĞİTİM PERFORMANSI GRAFİĞİ ===")
cv_results = grid_search.cv_results_
mean_scores = cv_results['mean_test_score']
std_scores = cv_results['std_test_score']

# En iyi skorları al
top_10_indices = numpy.argsort(mean_scores)[-10:]
top_10_scores = mean_scores[top_10_indices]
top_10_stds = std_scores[top_10_indices]

plt.figure(figsize=(12, 6))
x_pos = numpy.arange(len(top_10_scores))

plt.bar(x_pos, top_10_scores, yerr=top_10_stds, alpha=0.7, 
        color='skyblue', edgecolor='navy', linewidth=1)
plt.axhline(y=target_r2_threshold, color='red', linestyle='--', 
           linewidth=2, label=f'Target R² = {target_r2_threshold}')

plt.xlabel('Model Configuration', fontsize=12)
plt.ylabel('Cross-Validation R² Score', fontsize=12)
plt.title('SVM Top 10 Model Configurations\n(Cross-Validation Performance)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(x_pos, [f'Config {i+1}' for i in range(len(top_10_scores))])
plt.tight_layout()

# Kaydet
cv_filename = f"{graph_dir}/svm_cv_performance.png"
plt.savefig(cv_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Cross-validation performance grafiği kaydedildi: {cv_filename}")

# Uncertainty Analysis ile Quantile Regression
print(f"\n=== KUANTİL REGRESYON İLE BELİRSİZLİK (QR) ===")

# Quantile Regression modelleri
quantiles = [0.05, 0.5, 0.95]
quantile_models = {}

for q in quantiles:
    print(f"Training Quantile {q} model...")
    qr = Pipeline([
        ('scaler', StandardScaler()),
        ('quantile', QuantileRegressor(quantile=q, alpha=0.01, solver='highs'))
    ])
    qr.fit(X_train, Y_train)
    quantile_models[q] = qr

# Test verisiyle tahmin yap
quantile_predictions = {}
for q in quantiles:
    quantile_predictions[q] = quantile_models[q].predict(X_test)

# Uncertainty metrics hesapla
def calculate_picp_ace_pinaw(y_true, y_lower, y_upper, confidence_level):
    """
    PICP: Prediction Interval Coverage Probability
    ACE: Average Coverage Error 
    PINAW: Prediction Interval Normalized Average Width
    """
    n = len(y_true)
    
    # PICP: Kaç tane nokta aralık içinde
    coverage = numpy.sum((y_true >= y_lower) & (y_true <= y_upper))
    picp = coverage / n
    
    # ACE: Gerçek kapsama - beklenen kapsama
    ace = picp - confidence_level
    
    # PINAW: Normalize edilmiş ortalama genişlik
    interval_width = y_upper - y_lower
    pinaw = numpy.mean(interval_width) / (numpy.max(y_true) - numpy.min(y_true))
    
    return picp, ace, pinaw

# 90% PI metrikleri
picp_90, ace_90, pinaw_90 = calculate_picp_ace_pinaw(
    Y_test, quantile_predictions[0.05], quantile_predictions[0.95], 0.9
)

print(f"\n--- 90% PI Metrikleri (QR) ---")
print(f"PINC: {0.9000:.4f}")
print(f"PICP: {picp_90:.4f}")
print(f"ACE: {ace_90:.4f}")
print(f"PINAW: {pinaw_90:.4f}")

# Prediction Interval grafiği
plt.figure(figsize=(14, 8))

# Test noktalarını sırala
sort_indices = numpy.argsort(Y_test)
sorted_actual = Y_test[sort_indices]
sorted_lower = quantile_predictions[0.05][sort_indices]
sorted_median = quantile_predictions[0.5][sort_indices]
sorted_upper = quantile_predictions[0.95][sort_indices]

x_axis = numpy.arange(len(sorted_actual))

# Plot
plt.fill_between(x_axis, sorted_lower, sorted_upper, alpha=0.3, color='lightblue', 
                label='90% Prediction Interval')
plt.plot(x_axis, sorted_actual, 'ro-', linewidth=2, markersize=6, 
         label='Actual Values', alpha=0.8)
plt.plot(x_axis, sorted_median, 'b-', linewidth=2, 
         label='Median Prediction (Q50)', alpha=0.8)
plt.plot(x_axis, sorted_lower, 'g--', linewidth=1, alpha=0.7, label='Q05')
plt.plot(x_axis, sorted_upper, 'g--', linewidth=1, alpha=0.7, label='Q95')

plt.xlabel('Test Sample Index (sorted)', fontsize=12)
plt.ylabel('Wear Rate', fontsize=12)
plt.title(f'SVM Quantile Regression - 90% Prediction Interval\nPICP: {picp_90:.3f}, ACE: {ace_90:.3f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Kaydet
pi_filename = f"{graph_dir}/svm_prediction_interval_90%_pi.png"
plt.savefig(pi_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Tahmin aralığı grafiği kaydedildi: {pi_filename}")

# Quartile Forecast grafiği
print(f"\n=== QUARTİLE FORECAST GRAFİĞİ ===")
plt.figure(figsize=(14, 10))

# Test verisi indeksleri
test_indices = numpy.arange(len(Y_test))

# Ana grafik
plt.subplot(2, 1, 1)
plt.fill_between(test_indices, quantile_predictions[0.05], quantile_predictions[0.95], 
                alpha=0.3, color='lightcoral', label='90% Confidence Band (Q05-Q95)')
plt.plot(test_indices, Y_test, 'ko-', linewidth=2, markersize=6, 
         label='Actual Values', alpha=0.8)
plt.plot(test_indices, test_predictions, 'b-', linewidth=2, 
         label='SVM Predictions', alpha=0.8)
plt.plot(test_indices, quantile_predictions[0.05], 'r--', linewidth=1, 
         alpha=0.7, label='5% Quantile (Q05)')
plt.plot(test_indices, quantile_predictions[0.95], 'r--', linewidth=1, 
         alpha=0.7, label='95% Quantile (Q95)')

plt.xlabel('Test Sample Index', fontsize=12)
plt.ylabel('Wear Rate', fontsize=12)
plt.title('SVM Quartile Forecast Analysis', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Alt grafik - Error analysis
plt.subplot(2, 1, 2)
errors = Y_test - test_predictions
plt.plot(test_indices, errors, 'g-', linewidth=2, alpha=0.8, label='Prediction Errors')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.fill_between(test_indices, quantile_predictions[0.05] - test_predictions, 
                quantile_predictions[0.95] - test_predictions, 
                alpha=0.2, color='gray', label='Uncertainty Range')

plt.xlabel('Test Sample Index', fontsize=12)
plt.ylabel('Prediction Error', fontsize=12)
plt.title('Prediction Errors with Uncertainty', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Kaydet
quartile_filename = f"{graph_dir}/svm_quartile_forecast.png"
plt.savefig(quartile_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Quartile Forecast grafiği kaydedildi: {quartile_filename}")

# Quartile verilerinin özeti
print("Quartile verileri:")
print(f"  5% Quantile: min={numpy.min(quantile_predictions[0.05]):.3f}, max={numpy.max(quantile_predictions[0.05]):.3f}")
print(f"  95% Quantile: min={numpy.min(quantile_predictions[0.95]):.3f}, max={numpy.max(quantile_predictions[0.95]):.3f}")
print(f"  SVM Predictions: min={numpy.min(test_predictions):.3f}, max={numpy.max(test_predictions):.3f}")
print(f"  Actual Values: min={numpy.min(Y_test):.3f}, max={numpy.max(Y_test):.3f}")
print("Quartile Forecast analizi tamamlandı!")

# Farklı kapsama seviyeleri
coverage_levels = [(0.1, 0.9), (0.025, 0.975)]
print(f"\n--- Farklı Kapsama Seviyeleri (QR) ---")
print(f"{'q_low':>6s}  {'q_high':>6s}   {'PINC':>4s}   {'PICP':>4s}     {'ACE':>4s}  {'PINAW':>5s}")

for q_low, q_high in coverage_levels:
    pinc = q_high - q_low
    
    # Bu quantile'lar için model eğit (eğer yoksa)
    if q_low not in quantile_models:
        qr_low = Pipeline([
            ('scaler', StandardScaler()),
            ('quantile', QuantileRegressor(quantile=q_low, alpha=0.01, solver='highs'))
        ])
        qr_low.fit(X_train, Y_train)
        quantile_models[q_low] = qr_low
        
    if q_high not in quantile_models:
        qr_high = Pipeline([
            ('scaler', StandardScaler()),
            ('quantile', QuantileRegressor(quantile=q_high, alpha=0.01, solver='highs'))
        ])
        qr_high.fit(X_train, Y_train)
        quantile_models[q_high] = qr_high
    
    # Tahmin yap
    pred_low = quantile_models[q_low].predict(X_test)
    pred_high = quantile_models[q_high].predict(X_test)
    
    # Metrikleri hesapla
    picp, ace, pinaw = calculate_picp_ace_pinaw(Y_test, pred_low, pred_high, pinc)
    
    print(f"{q_low:6.4f}  {q_high:6.4f} {pinc:6.4f} {picp:6.4f} {ace:8.4f} {pinaw:7.4f}")

# Belirsizlik analizi sonuçları
print(f"\n=== BELİRSİZLİK ANALİZİ VE İYİLEŞTİRME ÖNERİLERİ ===")
print(f"\n📊 90% PI Durumu:")
print(f"   Hedef kapsama (PINC): {90.00:.2f}%")
print(f"   Gerçek kapsama (PICP): {picp_90*100:.1f}%")
print(f"   Fark (ACE): {ace_90:.3f}")

if ace_90 < -0.1:
    print(f"   ⚠️  CİDDİ UNDERCOVERAGE: Bantlar çok dar!")
    correction_factor = 0.9 / picp_90
    print(f"\n🔧 İyileştirme Önerisi:")
    print(f"   Bant genişliği çarpanı: {correction_factor:.2f}")
    print(f"   Daha güvenli kuantiller: q=0.01-0.99 (98% PI)")
elif ace_90 < -0.05:
    print(f"   ⚠️  Hafif undercoverage")
else:
    print(f"   ✅ Kabul edilebilir kapsama")

print(f"\n=== SVM vs ANFIS KARŞILAŞTIRMASI ===")
print("SVM avantajları:")
print("✅ Kernel trick ile non-linear ilişkileri modelleyebilir")
print("✅ Outlier'lara karşı robust")
print("✅ Yüksek boyutlu verilerde etkili")
print("✅ Global optimum bulma garantisi")
print("✅ Hyperparameter tuning ile esneklik")

print("\nSVM vs ANFIS karşılaştırma grafiği oluşturuluyor...")

# Basit karşılaştırma grafiği
plt.figure(figsize=(12, 8))

methods = ['SVM', 'Previous ANFIS']
r2_scores = [test_r2, 0.918594]  # ANFIS'ten aldığımız sonuç
mse_scores = [mse, 0.048335]
mae_scores = [mae, 0.164973]

x = numpy.arange(len(methods))
width = 0.25

plt.subplot(2, 2, 1)
plt.bar(x, r2_scores, width, label='R²', color=['skyblue', 'lightcoral'])
plt.axhline(y=target_r2_threshold, color='red', linestyle='--', alpha=0.7)
plt.ylabel('R² Score')
plt.title('R² Comparison')
plt.xticks(x, methods)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.bar(x, mse_scores, width, label='MSE', color=['lightgreen', 'orange'])
plt.ylabel('MSE')
plt.title('MSE Comparison') 
plt.xticks(x, methods)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.bar(x, mae_scores, width, label='MAE', color=['gold', 'purple'])
plt.ylabel('MAE')
plt.title('MAE Comparison')
plt.xticks(x, methods)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
training_time = [grid_search.refit_time_, 10]  # ANFIS'ten yaklaşık epoch sayısı
plt.bar(x, training_time, width, label='Training Time', color=['lightblue', 'pink'])
plt.ylabel('Training Time (relative)')
plt.title('Training Time Comparison')
plt.xticks(x, methods)
plt.grid(True, alpha=0.3)

plt.suptitle('SVM vs ANFIS Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()

# Kaydet
comparison_filename = f"{graph_dir}/svm_vs_anfis_comparison.png"
plt.savefig(comparison_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"SVM vs ANFIS karşılaştırma grafiği kaydedildi: {comparison_filename}")

print(f"\n=== TÜM GRAFİKLER TAMAMLANDI ===")
print("Oluşturulan grafikler:")
print(f"1. {regression_filename}")
print(f"2. {graph_dir}/svm_3d_surface_modeling.png")
print(f"3. {graph_dir}/svm_feature_interactions.png") 
print(f"4. {cv_filename}")
print(f"5. {pi_filename}")
print(f"6. {quartile_filename}")
print(f"7. {comparison_filename}")

print(f"\n🎯 SVM EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
print(f"Hedef R² = {target_r2_threshold:.4f}")
print(f"Ulaşılan R² = {test_r2:.6f}")
print(f"Sonuç: {'✅ BAŞARILI' if test_r2 >= target_r2_threshold else '❌ HEDEF ULAŞILAMADI'}")
