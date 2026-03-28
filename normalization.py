"""
Модуль нормализации данных для анализа микробиома
Поддерживает различные методы нормализации и трансформации данных
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class NormalizationProcessor:
    """Класс для нормализации микробиомных данных"""
    
    def __init__(self):
        self.normalization_methods = {
            'none': 'No normalization',
            'relative': 'Relative abundance (%)',
            'clr': 'CLR (Centered Log-Ratio)',
            'log': 'Log10 transformation',
            'sqrt': 'Square root transformation',
            'asinh': 'Inverse hyperbolic sine',
            'zscore': 'Z-score normalization'
        }
    
    def normalize(self, data, method='none'):
        """
        Применение выбранного метода нормализации
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для нормализации (образцы x OTU)
        method : str
            Метод нормализации
            
        Returns:
        --------
        pd.DataFrame
            Нормализованные данные
        """
        if data.empty:
            return data
            
        if method == 'none':
            return data
        elif method == 'relative':
            return self.get_relative_abundance(data)
        elif method == 'clr':
            return self.clr_transform(data)
        elif method == 'log':
            return self.log_transform(data)
        elif method == 'sqrt':
            return self._sqrt_transform(data)
        elif method == 'asinh':
            return self._asinh_transform(data)
        elif method == 'zscore':
            return self._zscore_transform(data)
        else:
            return data
    
    def get_relative_abundance(self, data):
        """
        Преобразование в относительную численность (%)
        
        Parameters:
        -----------
        data : pd.DataFrame
            Абсолютные значения численности
            
        Returns:
        --------
        pd.DataFrame
            Относительная численность в процентах
        """
        # Сумма по строкам (для каждого образца)
        row_sums = data.sum(axis=1)
        
        # Избегаем деления на ноль
        row_sums = row_sums.replace(0, 1)
        
        # Преобразование в проценты
        relative_data = data.div(row_sums, axis=0) * 100
        
        return relative_data
    
    def clr_transform(self, data):
        """
        Публичный метод для CLR (Centered Log-Ratio) трансформации
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для CLR трансформации
            
        Returns:
        --------
        pd.DataFrame
            CLR трансформированные данные
        """
        return self._clr_transform(data)
    
    def log_transform(self, data):
        """
        Публичный метод для логарифмической трансформации
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для логарифмической трансформации
            
        Returns:
        --------
        pd.DataFrame
            Логарифмически трансформированные данные
        """
        return self._log_transform(data)
    
    def _clr_transform(self, data):
        """
        CLR (Centered Log-Ratio) трансформация
        
        Parameters:
        -----------
        data : pd.DataFrame
            Компоненты данных (образцы x компоненты)
            
        Returns:
        --------
        pd.DataFrame
            CLR трансформированные данные
        """
        # Добавляем псевдосчет для избежания логарифма нуля
        data_pseudo = data + 1
        
        # Вычисляем геометрическое среднее для каждого образца
        geometric_mean = np.exp(np.log(data_pseudo).mean(axis=1))
        
        # CLR трансформация
        clr_data = np.log(data_pseudo.div(geometric_mean, axis=0))
        
        return clr_data
    
    def _log_transform(self, data):
        """
        Логарифмическая трансформация
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для трансформации
            
        Returns:
        --------
        pd.DataFrame
            Логарифмически трансформированные данные
        """
        # log10(x + 1) для избежания логарифма нуля
        return np.log10(data + 1)
    
    def _sqrt_transform(self, data):
        """
        Квадратный корень трансформации
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для трансформации
            
        Returns:
        --------
        pd.DataFrame
            Трансформированные данные
        """
        return np.sqrt(data)
    
    def _asinh_transform(self, data):
        """
        Обратная гиперболическая синус трансформация
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для трансформации
            
        Returns:
        --------
        pd.DataFrame
            Трансформированные данные
        """
        return np.arcsinh(data)
    
    def _zscore_transform(self, data):
        """
        Z-score нормализация
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для нормализации
            
        Returns:
        --------
        pd.DataFrame
            Z-score нормализованные данные
        """
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(data)
        
        return pd.DataFrame(scaled_values, 
                          index=data.index, 
                          columns=data.columns)
    
    def create_normalized_summary(self, original_data, normalized_data, method):
        """
        Создание сводки по нормализации данных
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Исходные данные
        normalized_data : pd.DataFrame
            Нормализованные данные
        method : str
            Метод нормализации
            
        Returns:
        --------
        dict
            Сводка по нормализации
        """
        summary = {
            'method': method,
            'method_description': self.normalization_methods.get(method, 'Unknown'),
            'original_shape': original_data.shape,
            'normalized_shape': normalized_data.shape,
            'original_stats': {
                'min': original_data.min().min(),
                'max': original_data.max().max(),
                'mean': original_data.mean().mean(),
                'std': original_data.std().mean()
            },
            'normalized_stats': {
                'min': normalized_data.min().min(),
                'max': normalized_data.max().max(),
                'mean': normalized_data.mean().mean(),
                'std': normalized_data.std().mean()
            }
        }
        
        return summary

    def get_available_methods(self):
        """
        Получение списка доступных методов нормализации
        
        Returns:
        --------
        dict
            Словарь методов нормализации
        """
        return self.normalization_methods.copy()

    def validate_data(self, data):
        """
        Валидация данных перед нормализацией
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для валидации
            
        Returns:
        --------
        bool
            True если данные корректны, False иначе
        """
        if not isinstance(data, pd.DataFrame):
            print("Ошибка: данные должны быть pandas DataFrame")
            return False
        
        if data.empty:
            print("Ошибка: данные пустые")
            return False
        
        if data.isnull().all().all():
            print("Ошибка: все данные равны NaN")
            return False
        
        if (data < 0).any().any():
            print("Предупреждение: обнаружены отрицательные значения")
        
        return True
