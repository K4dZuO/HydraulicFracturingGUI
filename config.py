"""
Константы приложения для GUI и расчетов.
"""

# Размеры окна
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# Геометрия виджетов
TAB_WIDGET_X = 350
TAB_WIDGET_Y = 20
TAB_WIDGET_WIDTH = 1250
TAB_WIDGET_HEIGHT = 950
LEFT_PANEL_MAX_WIDTH = 300

# Пороги качества интерполяции
RMSE_EXCELLENT_THRESHOLD = 0.01
RMSE_GOOD_THRESHOLD = 0.1

# Параметры по умолчанию для скважины
DEFAULT_K = 5.0  # проницаемость
DEFAULT_MU = 1.0  # вязкость
DEFAULT_B = 1.0  # объемный коэффициент
DEFAULT_PHI = 0.2  # пористость
DEFAULT_C_T = 1e-4  # общая сжимаемость

# Параметры фильтров
DEFAULT_OUTLIER_THRESHOLD = 1.5  # порог для обнаружения выбросов (IQR)
DEFAULT_SAVGOL_WINDOW_LENGTH = 5  # длина окна для фильтра Savitzky-Golay
DEFAULT_SAVGOL_POLYORDER = 2  # порядок полинома для фильтра Savitzky-Golay

# Параметры для расчета
N_PARAMETERS_PER_POINT = 2  # давление + дебит
REPORT_SEPARATOR_LENGTH = 60  # длина разделителя в отчетах

# Параметры типовых кривых
TYPE_CURVE_N_POINTS = 100  # количество точек для типовых кривых
TYPE_CURVE_TIME_MIN = -1  # минимум логарифма времени
TYPE_CURVE_TIME_MAX = 3  # максимум логарифма времени

# Методы интерполяции
INTERPOLATION_METHODS = ('rbf', 'gp')  # методы интерполяции по умолчанию

