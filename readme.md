# Диффузионные модели вероятностного шумоподавления (DDPM)

## Введение

Генеративное моделирование — это задача моделирования сложных распределений данных $p(\mathbf{x})$, с целью генерировать новые, реалистичные образцы, схожие с теми, что были в обучающем наборе. Среди популярных подходов можно выделить:

- GAN (Generative Adversarial Networks) — обучение по соревновательной схеме.

- VAE (Variational Autoencoders) — обучение по принципу вариационного вывода.

- Autoregressive models — моделируют $p(\mathbf{x})$ как произведение условных вероятностей.

Диффузионные модели (Diffusion Models) — сравнительно новая и мощная альтернатива, которая показала впечатляющие результаты в задачах генерации изображений и аудио (например, Stable Diffusion).

___

Идея DDPM (Denoising Diffusion Probabilistic Models, Ho et al., 2020) заключается в том, чтобы:

- Постепенно размывать данные, превращая их в чистый гауссовский шум (прямой процесс).

- Затем обучить нейросеть, которая восстанавливает данные из шума (обратный процесс).

## Прямой процесс (Forward Process)

Идея прямого процесса заключается в том, чтобы постепенно добавлять гауссовский шум к данным, превращая их в чистый шум к шагу $T$.

На каждом шаге $t = 1, \dots, T$:

- Мы определяем переход $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ как гауссовское распределение:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}, \beta_t \, \mathbf{I})
$$

Здесь $\beta_t \in (0, 1)$ — заранее заданная дисперсия шума на каждом шаге (обычно задаётся линейно или по косинусному графику).

---

### Кумулятивная форма: $q(\mathbf{x}_t | \mathbf{x}_0)$

Так как процесс марковский, можно выразить $q(\mathbf{x}_t | \mathbf{x}_0)$ в замкнутом виде:

Обозначим:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

Тогда:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0, (1 - \bar{\alpha}_t) \, \mathbf{I})
$$

Это очень важный результат: он позволяет **напрямую сэмплировать** $\mathbf{x}_t$ из $\mathbf{x}_0$, минуя все промежуточные шаги!

---

### Семплирование через шум $\varepsilon$

Так как $q(\mathbf{x}_t | \mathbf{x}_0)$ — это нормальное распределение, можно сэмплировать $\mathbf{x}_t$ напрямую:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon, \quad \varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Эта формула используется при обучении, чтобы напрямую получить $\mathbf{x}_t$ и применить нейросеть на произвольном шаге $t$.

---

### Итого:

- Прямой процесс $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ — это фиксированная последовательность гауссовских шумов, которая **не требует обучения**.
- Вся стохастичность и обучение происходят в **обратном процессе**.

## Обратный процесс (Reverse Process)

В прямом процессе $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ мы постепенно добавляли шум, пока не дошли до распределения, близкого к стандартному нормальному: $q(\mathbf{x}_T) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Теперь задача — **восстановить** $\mathbf{x}_0$ начиная с $\mathbf{x}_T$, двигаясь по шагам в обратном направлении:
$$
p_\theta(\mathbf{x}_0, \dots, \mathbf{x}_{T-1} | \mathbf{x}_T) = \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
$$

---

### Почему мы не можем использовать $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$?

- Вариационное распределение $q$ задаётся через всю цепочку начиная с $\mathbf{x}_0$.
- Однако **обратное распределение** $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ **зависит от** $\mathbf{x}_0$:
  $$
  q(\mathbf{x}_{t-1} | \mathbf{x}_t) = \int q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) q(\mathbf{x}_0 | \mathbf{x}_t) d\mathbf{x}_0
  $$
- Его невозможно вычислить напрямую (оно требует знания истинного распределения данных $p(\mathbf{x}_0)$), поэтому мы приближаем его параметризованной моделью $p_\theta$.

---

### Аппроксимация обратного процесса

Будем приближать каждое распределение $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ гауссовским:
$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
$$

На практике часто **дисперсия фиксируется**, а нейросеть обучается только на предсказание среднего $\mu_\theta$ (или шума $\varepsilon_\theta$, как мы увидим позже).

---

### Обучение через вариационный вывод

Мы хотим **максимизировать правдоподобие**:
$$
\log p_\theta(\mathbf{x}_0)
$$

Так как мы не можем вычислить его напрямую, мы будем максимизировать **нижнюю оценку (ELBO)** через вариационный вывод:
$$
\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right] = -\mathcal{L}_{\text{VLB}}
$$

Разложим ELBO по шагам:

---

### Раскладка $\mathcal{L}_{\text{VLB}}$

Распишем логарифм отношения как сумму:

$$
\mathcal{L}_{\text{VLB}} = \mathbb{E}_{q} \left[ \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_T | \mathbf{x}_0) \,\|\, p(\mathbf{x}_T))}_{\text{терм инициализации}} + \sum_{t=2}^T D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0 | \mathbf{x}_1) \right]
$$

Здесь:
- $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ — **true posterior** (вычислимый!).
- $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ — предсказание модели.
- $p_\theta(\mathbf{x}_T)$ — обычно фиксируется как $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

---

### Ключевой факт: $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ — гауссовское

Благодаря свойствам гауссовских распределений, можно аналитически получить:
$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

где:
- Среднее:
$$
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
$$

- Дисперсия:
$$
\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1}) \beta_t}{1 - \bar{\alpha}_t}
$$

---

### Переход к регрессии шума

Теперь мы можем **переписать задачу** не как предсказание $\mu_\theta$, а как задачу предсказания шума $\varepsilon$:

Из уравнения:
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon
$$

можно выразить:
$$
\varepsilon = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}
$$

Тогда можно обучать нейросеть $\varepsilon_\theta(\mathbf{x}_t, t)$ напрямую, минимизируя **простой MSE**:
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \varepsilon} \left[ \left\| \varepsilon - \varepsilon_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

---

### Почему это работает?

- Ho et al. (2020) показали, что эта **упрощённая функция потерь** (MSE по шуму) **пропорциональна части ELBO**, и на практике даёт те же результаты или лучше.
- Мы тренируем сеть угадывать шум, который был добавлен на шаге $t$, имея на входе $\mathbf{x}_t$.

---

### Вывод

Вместо ELBO в полном виде, на практике используют:

- Простую MSE-функцию:
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \varepsilon \sim \mathcal{N}(0, I)} \left[ \left\| \varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon, t) \right\|^2 \right]
$$

## Подробный вывод функции потерь в DDPM (*)

Рассмотрим задачу вариационного вывода:

Мы хотим максимизировать правдоподобие:
$$
\log p_\theta(\mathbf{x}_0)
$$

Так как это невычислимо напрямую, переходим к вариационной нижней оценке (ELBO):
$$
\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right] = -\mathcal{L}_{\text{VLB}}
$$

---

### Раскладка по цепочке

Распишем числитель и знаменатель:

- Генеративный процесс:
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
$$

- Вариационное приближение:
$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1})
$$

---

### Подстановка в ELBO

Подставим в ELBO:

$$
\mathcal{L}_{\text{VLB}} = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ -\log p(\mathbf{x}_T) - \sum_{t=1}^T \log p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) + \sum_{t=1}^T \log q(\mathbf{x}_t | \mathbf{x}_{t-1}) \right]
$$

Перепишем это как:

$$
\mathcal{L}_{\text{VLB}} = \mathbb{E}_{q} \left[ D_{\mathrm{KL}}(q(\mathbf{x}_T | \mathbf{x}_0) \, \| \, p(\mathbf{x}_T)) + \sum_{t=2}^T D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \, \| \, p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0 | \mathbf{x}_1) \right]
$$

> Здесь использован стандартный приём переписать логарифмы как KL-дивергенции между известным постериором $q$ и предсказанием модели $p_\theta$.

---

### Теперь покажем, что $q(x_{t-1} | x_t, x_0)$ — тоже гауссовское

Для этого используем известную формулу для условного распределения в байесовской цепочке гауссов:

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

Где:

- Среднее:
$$
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
$$

- Дисперсия:
$$
\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1}) \beta_t}{1 - \bar{\alpha}_t}
$$

---

### Модельное приближение $p_\theta(x_{t-1} | x_t)$

Будем приближать $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ гауссовским распределением с фиксированной дисперсией:

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mu_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
$$

Для обучения можно выбирать:
- предсказание $\mu_\theta$,
- или, альтернативно, предсказывать $\varepsilon_\theta$ (шум) и восстанавливать $\mu_\theta$ по нему.

---

### Связь с предсказанием шума

Имеем (см. прямой процесс):

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

Значит можно выразить:
$$
\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \, \varepsilon \right)
$$

Если нейросеть предсказывает $\varepsilon_\theta(\mathbf{x}_t, t)$, то можем получить оценку $\hat{\mathbf{x}}_0$:
$$
\hat{\mathbf{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \, \varepsilon_\theta(\mathbf{x}_t, t) \right)
$$

Подставляя $\hat{\mathbf{x}}_0$ в выражение для $\tilde{\mu}_t$, получаем предсказание $\mu_\theta$ через $\varepsilon_\theta$.

---

### Вывод функции потерь

Подставим это обратно в KL-дивергенцию:
$$
D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t))
$$

Это KL двух нормальных распределений:
- Истинное: $q = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t \mathbf{I})$
- Модель: $p_\theta = \mathcal{N}(\mu_\theta, \sigma_t^2 \mathbf{I})$

Формула KL между двумя гауссианами:
$$
D_{\mathrm{KL}}(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2)) = \frac{1}{2} \left[ \log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{Tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1} (\mu_2 - \mu_1) \right]
$$

Всё это можно аналитически упростить и **получить, что KL сводится к MSE** между $\varepsilon$ и $\varepsilon_\theta$:
$$
\mathcal{L}_{\text{simple}}(t) \propto \left\| \varepsilon - \varepsilon_\theta(\mathbf{x}_t, t) \right\|^2
$$

---

Мы доказали, что ELBO разлагается в сумму KL-термов, где:
- Один из них — KL между гауссианами с одинаковой дисперсией,
- Его можно аналитически упростить до **MSE по шуму**.

Финальная функция потерь:
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \varepsilon} \left[ \left\| \varepsilon - \varepsilon_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

Это и есть та самая регрессия шума, которую использует DDPM.

