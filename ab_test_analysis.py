import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Для воспроизводимости результатов
np.random.seed(42)

def generate_ab_data(n_A=500, n_B=500):
    """
    Генерируем синтетические данные для A/B-теста.
    Группа A - оригинальный интерфейс, группа B - новый интерфейс.
    """
    # Предположим, что ключевая метрика – время, проведенное на сайте (в секундах)
    data_A = pd.DataFrame({
        'group': 'A',
        'time_on_site': np.random.normal(loc=300, scale=50, size=n_A)  # среднее время 300 сек
    })
    data_B = pd.DataFrame({
        'group': 'B',
        'time_on_site': np.random.normal(loc=320, scale=50, size=n_B)  # новое оформление дает среднее время 320 сек
    })
    data = pd.concat([data_A, data_B]).reset_index(drop=True)
    return data

def analyze_ab_test(data):
    """
    Выполняем анализ A/B-теста, включая t-тест и визуализацию.
    """
    # Выделяем данные по группам
    group_A = data[data['group'] == 'A']['time_on_site']
    group_B = data[data['group'] == 'B']['time_on_site']
    
    # Выполняем независимый t-тест
    t_stat, p_value = stats.ttest_ind(group_A, group_B)
    print(f"t-статистика: {t_stat:.2f}")
    print(f"p-значение: {p_value:.4f}")
    
    # Визуализация распределения времени на сайте для обеих групп
    plt.figure(figsize=(10, 6))
    sns.histplot(group_A, color='blue', label='Группа A', kde=True, stat="density", alpha=0.6)
    sns.histplot(group_B, color='green', label='Группа B', kde=True, stat="density", alpha=0.6)
    plt.title("Распределение времени на сайте по группам")
    plt.xlabel("Время на сайте (сек)")
    plt.legend()
    plt.show()
    
    return t_stat, p_value

if __name__ == '__main__':
    # Генерация данных A/B-теста
    df_ab = generate_ab_data(500, 500)
    
    # Анализ A/B-теста
    t_statistic, p_val = analyze_ab_test(df_ab)
