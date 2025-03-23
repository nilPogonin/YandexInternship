import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Для воспроизводимости результатов
np.random.seed(42)

def generate_data(n_samples=1000):
    """
    Генерируем синтетические данные для анализа оттока клиентов.
    """
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, size=n_samples),
        'monthly_charges': np.random.uniform(20, 120, size=n_samples),
        'tenure': np.random.randint(1, 72, size=n_samples),  # месяцев пользования услугами
        'num_services': np.random.randint(1, 5, size=n_samples)
    })
    # Генерация вероятности оттока с помощью логической комбинации признаков
    # Чем выше ежемесячные расходы и чем меньше длительность пользования, тем выше вероятность оттока
    prob_churn = (data['monthly_charges'] / 120) * (1 - data['tenure'] / 72) * (data['num_services'] / 5)
    data['churn'] = np.where(np.random.rand(n_samples) < prob_churn, 1, 0)
    return data

def exploratory_analysis(data):
    """
    Выполняем первичный анализ данных: распределения и корреляции.
    """
    print("Статистика по данным:")
    print(data.describe())
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='churn', data=data)
    plt.title("Распределение оттока клиентов")
    plt.xlabel("Отток (0 - остались, 1 - ушли)")
    plt.ylabel("Количество клиентов")
    plt.show()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Корреляционная матрица")
    plt.show()

def model_training(data):
    """
    Обучаем модель логистической регрессии для прогнозирования оттока.
    """
    X = data.drop('churn', axis=1)
    y = data['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Отчёт по классификации:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Матрица ошибок")
    plt.xlabel("Предсказанное значение")
    plt.ylabel("Истинное значение")
    plt.show()
    
    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend()
    plt.show()
    
    return model

if __name__ == '__main__':
    # Генерация данных
    df = generate_data(1000)
    
    # Предварительный анализ
    exploratory_analysis(df)
    
    # Обучение модели
    trained_model = model_training(df)
