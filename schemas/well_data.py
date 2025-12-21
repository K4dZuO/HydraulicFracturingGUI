from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import pandas as pd


class WellData(BaseModel):
    """
    Датакласс для данных разведки месторождений и ГРП.
    Содержит все необходимые параметры для анализа скважины.
    """
    # Параметры ГРП
    skin: float = Field(ge=-10, le=50, description="Фактор скин-эффекта")
    h: float = Field(gt=0, description="Толщина пласта, м")
    n: int = Field(ge=1, le=100, description="Количество трещин")
    w: float = Field(gt=0, description="Ширина трещины, м")
    l: float = Field(gt=0, description="Длина трещины, м")
    a_l: float = Field(gt=0, le=1, description="Отношение a/L")
    
    # Координаты элемента
    elem_idx: int = Field(ge=1, description="Индекс элемента")
    x: float = Field(description="Координата X, м")
    y: float = Field(description="Координата Y, м")
    
    # Временные данные
    t: float = Field(ge=0, description="Время, ч")
    p: float = Field(gt=0, description="Давление, атм")
    dp: float = Field(description="Изменение давления, атм")
    q: float = Field(description="Дебит, м³/сут")
    
    model_config = ConfigDict(extra='forbid')


class WellTimeSeries(BaseModel):
    """
    Датакласс для временных рядов данных скважины.
    Содержит массивы временных данных для анализа.
    """
    time: pd.Series = Field(description="Временной ряд")
    pressure: pd.Series = Field(description="Давление")
    pressure_derivative: Optional[pd.Series] = Field(default=None, description="Производная давления")
    depression: pd.Series = Field(default=None, description="Депрессия")
    flow_rate: pd.Series = Field(description="Дебит")
    flow_rate_derivative: Optional[pd.Series] = Field(default=None, description="Производная дебита")
    
    # Безразмерные параметры из данных (если есть)
    X: Optional[pd.Series] = Field(default=None, description="Безразмерный фильтрационный параметр X из данных")
    Y: Optional[pd.Series] = Field(default=None, description="Безразмерный ёмкостной параметр Y из данных")
    
    # Параметры скважины (константы для всего временного ряда)
    skin: float = Field(description="Фактор скин-эффекта")
    thickness: float = Field(description="Толщина пласта")
    fractures_count: int = Field(description="Количество трещин")
    fracture_width: float = Field(description="Ширина трещины")
    fracture_length: float = Field(description="Длина трещины")
    a_l_ratio: float = Field(description="Отношение a/L")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FlowRegimeAnalysis(BaseModel):
    """
    Результаты анализа режима течения.
    """
    regime_type: str = Field(description="Тип режима течения")
    confidence: float = Field(ge=0, le=1, description="Уверенность в классификации")
    characteristic_time: Optional[float] = Field(default=None, description="Характерное время перехода")
    parameters: dict = Field(description="Параметры режима")


class TypeCurveMatch(BaseModel):
    """
    Результаты сопоставления с эталонными кривыми.
    """
    curve_type: str = Field(description="Тип эталонной кривой")
    match_quality: float = Field(ge=0, le=1, description="Качество сопоставления")
    estimated_parameters: dict = Field(description="Оцененные параметры")
    residuals: pd.Series = Field(description="Остатки")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
