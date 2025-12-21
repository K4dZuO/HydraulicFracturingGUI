# Контракт на проверку и фиксы модуля фильтрации (для ИИ-агента-разработчика)

Ниже — **чёткий, пошаговый контракт** с техническими требованиями, эталонными реализациями, ожидаемыми входами/выходами, возможными ошибками и наборами тестов. Агент должен выполнить всё строго по контракту, оставляя интерфейсы публичных функций/классов без изменений (если не оговорено), добавив лишь исправления, конфиги и тесты.

---

## Общие цели

1. Исправить и стабилизировать поведение `SignalFilters` и вспомогательных функций/классов.
2. Сохранить совместимость с текущим API (имена классов/функций/аргументы).
3. Сделать поведение предсказуемым для реальных ГРП-данных: не «пересглаживать», корректно обрабатывать NaN/Inf, корректно работать в лог-масштабе.


---

# Эталонные реализации и допустимые переменные

Ниже — эталонные рецепты (псевдо/реальный код) и допустимые диапазоны параметров. Агент обязан реализовать эти варианты и документировать любые отклонения.

---

## 1) Savitzky-Golay — эталон

**Функция:** `SignalFilters.savgol(curve, window_length=None, polyorder=2, mode='nearest')`

### Правила:

* Если `window_length` задан: валидация `window_length >= polyorder + 1` и нечётность. Если нечётно/малое — поправить/пепересчитать и логнуть.
* Если `window_length` is `None`, использовать **консервативную стратегию**:

  * `n = len(curve_clean)` (после удаления NaN)
  * `window_length = min(11, max(5, int(round(n * 0.05)) | 1))`

    * т.е. окно = 5% длины, но не больше 11, и не меньше 5; всегда нечётное.
  * rationale: 5% часто даёт разумный баланс; ограничение 11 предотвращает пересглаживание на длинных сериях.
* Если `n < polyorder + 3`: вернуть исходный сигнал (слишком мало точек).
* При краевых условиях использовать `mode='interp'` или `mode='mirror'` для более стабильной работы (в SciPy доступно: `mode='nearest'|'mirror'|'interp'` — `interp` лучше, но проверить версию SciPy).
* **Не** строить `window_length` как `n//6` или `n//10*2` — это слишком большое.

### Эталон (код-фрагмент):

```python
n = len(curve_clean)
if window_length is None:
    w = max(5, int(round(n * 0.05)))
    if w % 2 == 0:
        w += 1
    window_length = min(w, 11)
# ensure min window and polyorder
window_length = max(window_length, polyorder + 1)
if window_length > n:
    return curve  # not enough data
filtered = savgol_filter(curve_clean, window_length, polyorder, mode=mode)
```

---

## 2) Gaussian — эталон

**Функция:** `SignalFilters.gaussian(curve, sigma=None, mode='nearest')`

### Правила:

* Если `sigma` задан — валидировать `sigma >= 0.1` и `sigma <= 3.0`. Если выходит за пределы — крошечный warning и приведение к [0.1, 3.0].
* Если `sigma` is `None` → адаптивно, но консервативно:

  * `sigma = min(max(0.25, n / 200.0), 1.0)`

    * rationale: для n=100 → sigma=0.5; n=1000 → sigma=1.0 (потолок)
* Для коротких серий (n < 8): `sigma = 0.3` или вернуть исходный сигнал.
* Использовать `gaussian_filter1d(..., mode='mirror')` или `'reflect'` для краёв.
* В случае пропусков (NaN): вызывать `fill_missing_values(..., method='linear')` предварительно (опционально, но по умолчанию — НЕ заполнять заменой константой).

### Эталон:

```python
if sigma is None:
    sigma = min(max(0.25, n/200.0), 1.0)
filtered = gaussian_filter1d(curve_clean, sigma=sigma, mode='reflect')
```

---

## 3) Kalman — эталон (1D простая реализация)

**Функция:** `SignalFilters.kalman(curve, process_noise=None, measurement_noise=None)`

### Правила:

* Не применять модель с трендом/скоростью (двухмерной), она даёт «вытягивание». Вместо этого — **простая 1D Kalman** (или адаптированный exponential smoothing через Kalman).
* Модель состояния: `x_k = x_{k-1} + w_k` (w_k ~ N(0, Q))
  Наблюдение: `z_k = x_k + v_k` (v_k ~ N(0, R))
* Инициализация:

  * `x0 = curve_clean[0]`
  * `P0 = var(signal_valid)` or 1.0
  * Если `process_noise`/`measurement_noise` is None: оценить через `estimate_noise_level` (R ~ var(diffs), Q ~ R*0.01)
* Итерация стандартная (scalar):

  ```
  P_pred = P + Q
  K = P_pred / (P_pred + R)
  x = x_pred + K*(z - x_pred)
  P = (1 - K)*P_pred
  ```
* Ограничения: **никакого ручного clip для скорости v**.
* При NaN: игнорировать — пропускать update (predict only) или заполнять заранее.

### Эталон:

```python
x = x0
P = P0
for z in measurements:
    P = P + Q
    K = P / (P + R)
    x = x + K * (z - x)
    P = (1 - K) * P
    store x
```

---

## 4) Log-domain фильтрация — эталон

**Функция:** `SignalFilters.log_domain(curve, base_filter='savgol', **kwargs)`

### Правила:

* Применять только если `detect_log_scale(signal)` → **маловероятно**, см. ниже.
* Перед логированием убедиться, что `signal > 0` по маске `valid_mask`. Если есть ≤0 — заменить через `fill_missing_values` или shift: `signal_pos = signal_clean - min(signal_clean) + eps` (только если shift небольшого масштаба и документировать).
* В лог-масштабе **не** применять полиномиальную фильтрацию с высоким order (polyorder ≤ 2).
* После перехода обратно делать коррекцию bias: если была сдвиг-нормализация — вернуть назад.
* В общем — рекомендовать: **лог-домен использовать очень экономно**; по умолчанию в `select_filter_method` возвращать `'log_domain'` только если `signal_max/signal_min > 1e3` (3 порядка) и signal_min > 0.

---

## 5) Hybrid — эталон (последовательность)

**Функция:** `SignalFilters.hybrid(curve, x=None)`

### Рецепт (безопасный):

1. `signal = fill_missing_values(signal, method='linear')`
2. `signal = savgol(signal, window_length=None, polyorder=2)`
3. `signal = gaussian(signal, sigma=0.5)`
4. (опционально) `if detect_oscillations(signal):` — `signal = kalman(signal, process_noise=..., measurement_noise=...)`
5. Возвращать полноразмерный массив (вставлять назад NaN, если были).

**Важно:** hybrid не должен вызывать log_domain по умолчанию.

---

# Вспомогательные функции — целевые правки

### `detect_log_scale(signal, threshold=3.0)`:

* Текущая реализация использует `threshold=10.0` как степени — это абсурдно. Исправить:

  * `return (np.log10(signal_max / signal_min) > threshold)` — где `threshold` = **3.0** по умолчанию (3 порядка).
  * Если `signal_min <= 0` → False.

### `select_filter_method(signal, x=None, snr_threshold=20.0, oscillation_threshold=0.5)`:

* Исправить логику: Kalman — лишь для низкого SNR **и** высокой осцилляции.
* Если `use_log`: return `'log_domain'`
* Else:

  * if `snr > snr_threshold`: `'savgol'`
  * elif `snr > 10`: `'gaussian'`
  * else: `'hybrid'`
* **Не** выбирать Kalman автоматически, только если `oscillation_score > 0.6 and snr < 10`.

### `compute_snr`:

* Текущее вычисление `noise_power = var(signal)` некорректно: сигнал_power/type. Лучше:

  * noise_estimate: `std(diff(signal))` or MAD.
  * SNR_linear = (mean(signal**2)) / (noise_estimate**2 + eps)
* Вернуть decibels.

### `fill_missing_values`:

* При x is None — current implementation OK. Добавить параметр `interpolate_limit` и проверку monotonic x.

---

# Физические ограничения (`PhysicsConstraints`) — краткие замечания

* `monotonic()` использует `np.maximum.accumulate(curve_sorted[::-1])[::-1]` — логика в целом верна для non_increasing, но после сглаживания ***нужно*** гарантировать, что позиционный индекс корректно восстанавливается: проверять `np.argsort(sort_idx)` раньше/позже. PR должен содержать unit test, который проверяет монотонность (включая NaN).
* `limit_curvature()` — порог `max_curvature=10.0` может быть слишком мал. Сделать нормализацию: `max_curvature` в диапазоне 0.1–100 и document default=10.0.
* `remove_oscillations()` — логика sign change + ffill нормальна, но надо внимательнее работать с границами (индексация смещений). Добавить тесты.

---

# Диагностика, логи и метрики

Каждый публичный вызов фильтра должен иметь опциональный `diagnostics: bool = False` или глобальный logger. Диагностика включает:

* входная длина `n`
* frac NaN
* SNR (dB)
* chosen_method (если денойзинг авто)
* used window/sigma/Q/R/…
* time taken (ms)

Логи сохранять в `SignalFilters.last_diag` и возвращать при `diagnostics=True`.

---

# Тесты (обязательные)

Добавить `tests/test_signal_filters.py` с набором тестов:

1. **Smoke tests** — проход для простых синтетических сигналов:

   * sine + small gaussian noise
   * step + noise
   * exponential decay (wide dynamic range)
2. **Edge cases**:

   * all NaN → returns original array
   * very short series n=2..4 → returns original or no error
   * series with Inf
3. **SavGol behavior**:

   * For n = 200, `savgol(..., window_length=None)` must use `window_length <= 11` and not produce huge smoothing (assert variance after filter > 0.2 * original variance)
4. **Gaussian behavior**:

   * sigma adaptive test: for n=100 -> sigma≈0.5
5. **Kalman**:

   * apply to noisy constant signal — variance after Kalman < original variance
   * do not produce linear trend on step input
6. **Log-domain**:

   * positive signal with 4 orders of magnitude → `log_domain` must reduce relative error, not blow up.
7. **Hybrid**:

   * combined test: hybrid reduces noise without shifting mean by > 5%
8. **Integration test**:

   * pipeline: fill_missing -> denoise(auto) -> physics_constraints.enforce_all -> output is finite, monotonic (if expected), not over-smoothed.

**Metrics for pass/fail**:

* SNR increase: SNR_out >= SNR_in or at least not decrease by >1 dB for reasonable filters.
* Mean absolute error relative to ground truth for synthetic signals (ground truth known) < threshold.
* No new NaNs/Inf in output.

---

# Acceptance criteria (must pass to close task)

1. All unit tests pass.
2. No change in public function signatures (unless added optional args).
3. Diagnostic logging implemented and can be toggled.
4. Demonstrated before/after plots for at least 3 representative real curves (attach images to PR).
5. PR includes `CHANGELOG` entry: what fixed and why.

---

# Конкретные задачи для агента (пошагово)

1. Склонировать репозиторий, создать ветку `fix/filters-stabilize`.
2. Разделить/создать модули: `signal_filters.py`, `filter_utils.py`, `physics_constraints.py` (минимальные правки для совместимости).
3. В `SignalFilters.savgol`:

   * заменить правило window_length на предложенное.
   * добавить защитные проверки.
   * добавить diagnostics запись (`window_length_used`).
4. В `SignalFilters.gaussian`:

   * заменить rule sigma.
   * использовать `mode='reflect'` для `gaussian_filter1d`.
5. В `SignalFilters.kalman`:

   * переписать на простую 1D Kalman (см. эталон).
   * удалить скорость `v` или сделать её опцией `use_trend=False` по умолчанию.
6. `log_domain`:

   * перестроить: проверка положительности, shift/scale safe handling, запрет SG(polynomial) > 2.
7. `hybrid`:

   * перестроить последовательность: fill_missing→SavGol(small)→Gaussian(0.5)
   * убрать лог-фильтрацию по-умолчанию.
8. Исправить `filter_utils.detect_log_scale` порог до 3 порядков (10^3).
9. Обновить `select_filter_method` по новым правилам.
10. Добавить diagnostics store (`SignalFilters.last_diag` / возвращаемый словарь).
11. Добавить unit-tests (см. выше) и integration test.
12. Провести сравнение на 3 реальных/репрезентативных файлах (сгенерировать или попросить у заказчика). В PR приложить png/jpg — before/after overlays.
13. Пройти CI (pytest) и собрать отчет.

---

# Примеры входов/ожидаемых выходов (для тестов)

1. **Sine + noise**:

```python
t = np.linspace(0, 2*np.pi, 200)
signal = np.sin(5*t) + np.random.normal(0, 0.1, size=t.shape)
out = SignalFilters.denoise(signal, method='savgol')
# Expect: var(out) < var(signal) but not < 0.1*var(signal)
```

2. **Exp decay with orders**:

```python
t = np.linspace(0, 10, 300)
signal = 10**(np.linspace(0, 3, 300)) * np.exp(-t/3) + np.random.normal(0, 0.05, 300)
out = SignalFilters.denoise(signal, method='hybrid')
# Expect: out positive, no NaN, SNR_out >= SNR_in - 1dB
```

3. **Step with NaNs**:

```python
signal = np.array([1.0]*50 + [np.nan]*10 + [2.0]*50)
out = SignalFilters.denoise(signal)
# Expect: no NaN in output, smooth transition, no linear trend between 1->2 except local smoothing
```

---

# Возможные ошибки / ловушки (что проверять при ревью)

* Автоматический выбор слишком больших окон (исправить).
* Повторное логарифмирование (если `prepare` уже сделал logX/logY) — guard.
* Смешивание NaN/Inf при фильтрации — убедиться, что NaN не становятся числами без контроля.
* Kalman с трендом — даёт «вытягивание» → заменить на 1D.
* Log-domain: небольшие ошибки в логе → большие в оригинальном пространстве → избегать SG в логе.
* Hybrid когда detect_log_scale=True — ставить строгие проверки перед применением.
* Performance: для n up to ~10000 фильтры должны быть реализованы эффективно (векторные операции).

---

# Документация и комментарии в коде

* Каждый публичный метод имеет docstring с:

  * Описание метода
  * Формула/алгоритм
  * Описание входных параметров, допустимый диапазон
  * Поведение при NaN/Inf
* В PR — включить короткий README с описанием конфигурации параметров (defaults) и рекомендациями по tuning.

