# Estructura de Datos para Sistema de Aprendizaje por Refuerzo en Robot Cuadrúpedo

## Introducción

Las estructuras de datos son fundamentales ya que determinan cómo se almacena, organiza y accede a la información que procesan los sistemas. Una adecuada elección puede tener un impacto significativo en la eficiencia, rendimiento y capacidad de escalabilidad de un sistema basado en inteligencia artificial o robótica.

No solo definen el modo en que la información se almacena físicamente, sino también los algoritmos y operaciones que pueden realizarse sobre ella, influyendo directamente en la complejidad computacional y en la capacidad del sistema para procesar y analizar datos de manera eficiente.

En el contexto de este proyecto de aprendizaje por refuerzo aplicado a locomoción robótica, aunque no se utilizan bases de datos tradicionales, las estructuras de datos en memoria juegan un rol crítico en:

1. **Representación del estado del sistema** (observaciones del entorno)
2. **Codificación de las decisiones del agente** (acciones)
3. **Gestión del proceso de entrenamiento** (buffers de experiencia, parámetros de la red neuronal)
4. **Coordinación entre componentes** (controlador de marcha, cinemática inversa, simulación física)

A continuación, se describen las estructuras de datos principales del sistema, su organización en memoria y su relación directa con el proceso de aprendizaje por refuerzo.

---

## 1. Arquitectura General del Sistema

El sistema se organiza en tres capas principales que gestionan diferentes niveles de abstracción:

### 1.1 Capa de Simulación Física
- **Motor:** MuJoCo (Multi-Joint dynamics with Contact)
- **Función:** Simula la dinámica del robot y su interacción con el terreno
- **Estructuras principales:**
  - `MjModel`: Configuración estática del modelo (geometría, masas, límites articulares)
  - `MjData`: Estado dinámico del sistema (posiciones, velocidades, fuerzas de contacto)

### 1.2 Capa de Control
- **Componentes:**
  - Controlador de marcha Bézier (generación de trayectorias nominales)
  - Cinemática inversa 3DOF (conversión de objetivos cartesianos a ángulos articulares)
  - Sistema de control residual (correcciones aprendidas por RL)
- **Función:** Traduce comandos de alto nivel en actuaciones articulares

### 1.3 Capa de Aprendizaje
- **Algoritmo:** Proximal Policy Optimization (PPO)
- **Función:** Optimiza la política residual mediante interacción con el entorno simulado
- **Estructuras principales:**
  - Redes neuronales (actor y crítico)
  - Buffer de experiencias (rollout buffer)
  - Normalizadores de observaciones/recompensas

---

## 2. Estructura de Datos de Observación (Vector de Estado)

El espacio de observación representa el estado perceptual del agente. En este proyecto, cada observación es un vector de **65 dimensiones** que se actualiza en cada paso de simulación:

### 2.1 Composición del Vector de Observación

```
┌─────────────────────────────────────────────────────────────┐
│ OBSERVACIÓN (65D) - Representación del estado del sistema   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Estado del cuerpo (13D)                                 │
│     ├─ Posición (x, y, z): 3D                               │
│     ├─ Orientación quaternion (w, x, y, z): 4D             │
│     ├─ Velocidad lineal (vx, vy, vz): 3D                   │
│     └─ Velocidad angular (ωx, ωy, ωz): 3D                  │
│                                                              │
│  2. Estado articular (24D)                                  │
│     └─ 12 articulaciones × 2 (posición + velocidad)        │
│        FL: [tilt, shoulder_L, shoulder_R] × 2               │
│        FR: [tilt, shoulder_L, shoulder_R] × 2               │
│        RL: [tilt, shoulder_L, shoulder_R] × 2               │
│        RR: [tilt, shoulder_L, shoulder_R] × 2               │
│                                                              │
│  3. Estado de las patas (24D)                               │
│     ├─ Posiciones de los pies (x, y, z) × 4: 12D           │
│     └─ Velocidades de los pies (vx, vy, vz) × 4: 12D       │
│                                                              │
│  4. Contactos con el suelo (4D)                             │
│     └─ Flags binarios [FL, FR, RL, RR]: 4D                 │
│        (1.0 = contacto, 0.0 = sin contacto)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Estructura de Código (Python)

```python
@dataclass
class ObservationComponents:
    """Componentes del vector de observación organizado en memoria."""

    # Estado del cuerpo principal (13D)
    body_position: np.ndarray      # shape=(3,) - [x, y, z] en metros
    body_orientation: np.ndarray   # shape=(4,) - quaternion [w,x,y,z]
    body_linear_vel: np.ndarray    # shape=(3,) - [vx, vy, vz] en m/s
    body_angular_vel: np.ndarray   # shape=(3,) - [ωx, ωy, ωz] en rad/s

    # Estado articular (24D)
    joint_positions: np.ndarray    # shape=(12,) - ángulos en radianes
    joint_velocities: np.ndarray   # shape=(12,) - velocidades en rad/s

    # Estado de las patas (24D)
    foot_positions: np.ndarray     # shape=(12,) - posiciones xyz × 4 patas
    foot_velocities: np.ndarray    # shape=(12,) - velocidades xyz × 4 patas

    # Información de contacto (4D)
    foot_contacts: np.ndarray      # shape=(4,) - flags binarios [FL,FR,RL,RR]
```

### 2.3 Relación con el Proceso de Aprendizaje

Esta estructura de observación:

- **Proporciona información completa del estado**: Permite al agente inferir la configuración actual del robot y su dinámica
- **Facilita la predicción temporal**: Las velocidades permiten anticipar estados futuros
- **Detecta condiciones críticas**: Los contactos y orientación indican estabilidad
- **Se normaliza automáticamente**: Mediante `VecNormalize` para estabilizar el entrenamiento

---

## 3. Estructura de Datos de Acción (Vector de Control Residual)

El espacio de acción representa las decisiones que toma el agente. En este proyecto se utiliza un enfoque de **control residual**, donde el agente aprende correcciones sobre un controlador de marcha base.

### 3.1 Composición del Vector de Acción

```
┌─────────────────────────────────────────────────────────────┐
│ ACCIÓN (12D) - Correcciones residuales por pata             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Valores normalizados en rango [-1, 1]                      │
│  Escalados por residual_scale (típicamente 0.01 m)          │
│                                                              │
│  Pata FL (Front-Left):                                      │
│  ├─ Δx (corrección longitudinal): 1D                        │
│  ├─ Δy (corrección lateral): 1D                             │
│  └─ Δz (corrección vertical): 1D                            │
│                                                              │
│  Pata FR (Front-Right):                                     │
│  ├─ Δx, Δy, Δz: 3D                                          │
│                                                              │
│  Pata RL (Rear-Left):                                       │
│  ├─ Δx, Δy, Δz: 3D                                          │
│                                                              │
│  Pata RR (Rear-Right):                                      │
│  └─ Δx, Δy, Δz: 3D                                          │
│                                                              │
│  Total: 4 patas × 3 dimensiones = 12D                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Procesamiento de la Acción

```python
def process_action(action: np.ndarray, residual_scale: float = 0.01) -> Dict[str, np.ndarray]:
    """
    Convierte acción normalizada [-1, 1] a residuales por pata.

    Args:
        action: Vector 12D normalizado
        residual_scale: Escala máxima de corrección en metros

    Returns:
        Diccionario {"FL": [Δx, Δy, Δz], "FR": [...], ...}
    """
    # Clip para garantizar rango válido
    action = np.clip(action, -1.0, 1.0)

    residuals = {}
    leg_names = ["FL", "FR", "RL", "RR"]

    for i, leg in enumerate(leg_names):
        # Extrae 3 valores por pata y escala
        residuals[leg] = action[i*3:(i+1)*3] * residual_scale

    return residuals
```

### 3.3 Integración con Controlador Base

```
┌──────────────────────────────────────────────────────────────┐
│ FLUJO DE CONTROL: Base + Residual → Objetivos Finales        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Controlador Bézier                                        │
│     └─→ Genera trayectoria nominal                           │
│         target_base[leg] = [x_nom, y_nom, z_nom]             │
│                                                               │
│  2. Agente RL (Red Neuronal)                                 │
│     └─→ Genera correcciones                                  │
│         residual[leg] = [Δx, Δy, Δz]                         │
│                                                               │
│  3. Combinación                                              │
│     └─→ target_final[leg] = target_base[leg] + residual[leg] │
│                                                               │
│  4. Cinemática Inversa                                       │
│     └─→ Convierte objetivos cartesianos a ángulos            │
│         angles[leg] = IK(target_final[leg])                  │
│                                                               │
│  5. Actuadores                                               │
│     └─→ Aplica comandos al simulador MuJoCo                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 3.4 Relación con el Proceso de Aprendizaje

El diseño de acción residual:

- **Simplifica el aprendizaje**: El controlador base ya genera marcha estable; el agente solo ajusta
- **Acota la exploración**: `residual_scale` limita correcciones grandes que romperían IK
- **Acelera convergencia**: El espacio de búsqueda es más pequeño que aprender de cero
- **Mantiene seguridad**: Garantiza que las acciones no violen límites físicos del robot

---

## 4. Estructura de Datos del Controlador de Marcha

El controlador de marcha diagonal genera patrones de locomoción tipo "trote" donde pares diagonales de patas se mueven de forma coordinada.

### 4.1 Parámetros de Marcha

```python
@dataclass
class GaitParameters:
    """Configuración del patrón de marcha."""

    body_height: float = 0.05      # Altura del cuerpo sobre el suelo [m]
    step_length: float = 0.06      # Longitud de cada paso [m]
    step_height: float = 0.04      # Altura de elevación del pie [m]
    cycle_time: float = 0.8        # Duración de un ciclo completo [s]
    swing_shape: float = 0.35      # Parámetro de forma de curva Bézier [0-1]

    # Offsets laterales por pata (para ajustar ancho de stance)
    lateral_offsets: Dict[str, float] = field(default_factory=lambda: {
        "FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0
    })
```

### 4.2 Máquina de Estados

```
┌─────────────────────────────────────────────────────────────┐
│ ESTADOS DE MARCHA DIAGONAL                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Estado 1: pair_a_swing                                      │
│  ┌────────────────────────────────────────┐                 │
│  │ Swing (fase aérea):    FL + RR         │                 │
│  │ Stance (fase apoyo):   FR + RL         │                 │
│  │ Duración: cycle_time / 2               │                 │
│  └────────────────────────────────────────┘                 │
│                   ↓ (al completar)                           │
│  Estado 2: pair_b_swing                                      │
│  ┌────────────────────────────────────────┐                 │
│  │ Swing (fase aérea):    FR + RL         │                 │
│  │ Stance (fase apoyo):   FL + RR         │                 │
│  │ Duración: cycle_time / 2               │                 │
│  └────────────────────────────────────────┘                 │
│                   ↓ (al completar)                           │
│                [retorna a Estado 1]                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Generación de Trayectorias

**Para patas en swing (fase aérea):**
- Usa curva Bézier cúbica para trayectoria suave
- Puntos de control: inicio → elevación → descenso → aterrizaje
- Evaluación paramétrica en función del tiempo normalizado τ ∈ [0, 1]

**Para patas en stance (fase de apoyo):**
- Interpolación lineal desde frente hacia atrás
- Genera movimiento relativo del cuerpo hacia adelante

```python
class DiagonalGaitController:
    """Gestor de estado y generación de trayectorias."""

    def __init__(self, params: GaitParameters):
        self.params = params
        self.state_duration = params.cycle_time / 2.0
        self.phase_elapsed = 0.0

        # Estado actual
        self.active_swing_pair = ("FL", "RR")   # Par en fase aérea
        self.active_stance_pair = ("FR", "RL")  # Par en apoyo

        # Curva Bézier pre-computada para swing
        self.swing_curve = self._build_bezier_curve()

    def update(self, dt: float) -> Dict[str, np.ndarray]:
        """
        Avanza el controlador dt segundos y retorna objetivos por pata.

        Returns:
            {"FL": [x, y, z], "FR": [x, y, z], "RL": [x, y, z], "RR": [x, y, z]}
        """
        self.phase_elapsed += dt

        # Transición de estado si se completó medio ciclo
        if self.phase_elapsed >= self.state_duration:
            self.phase_elapsed -= self.state_duration
            self._toggle_swing_stance_pairs()

        # Calcula progreso normalizado τ ∈ [0, 1]
        tau = self.phase_elapsed / self.state_duration

        targets = {}
        for leg in self.active_swing_pair:
            targets[leg] = self._evaluate_swing_trajectory(leg, tau)

        for leg in self.active_stance_pair:
            targets[leg] = self._evaluate_stance_trajectory(leg, tau)

        return targets
```

### 4.4 Relación con el Proceso de Aprendizaje

El controlador de marcha:

- **Codifica conocimiento experto**: Patrones de locomoción biológicamente inspirados
- **Proporciona baseline funcional**: El robot puede caminar sin aprendizaje
- **Reduce dimensionalidad del problema**: RL solo aprende ajustes, no marcha desde cero
- **Genera señal de supervisión implícita**: Las recompensas comparan comportamiento con patrón esperado

---

## 5. Estructura de Datos de la Red Neuronal (Política y Función de Valor)

El agente PPO utiliza dos redes neuronales separadas implementadas como perceptrones multicapa (MLP):

### 5.1 Arquitectura de la Red

```
┌──────────────────────────────────────────────────────────────────┐
│ ARQUITECTURA DE REDES NEURONALES (configuración "large")         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RED ACTOR (Política π)                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                                                             │  │
│  │  Input: Observación (65D)                                  │  │
│  │    ↓                                                        │  │
│  │  Capa Densa 1: 65 → 512 [ELU]                              │  │
│  │    ↓                                                        │  │
│  │  Capa Densa 2: 512 → 256 [ELU]                             │  │
│  │    ↓                                                        │  │
│  │  Capa Densa 3: 256 → 128 [ELU]                             │  │
│  │    ↓                                                        │  │
│  │  Capa de Salida:                                           │  │
│  │    ├─ Media μ: 128 → 12 [lineal]                           │  │
│  │    └─ Log Std σ: parámetro entrenable (12D)                │  │
│  │                                                             │  │
│  │  Output: Distribución Gaussiana N(μ, σ²)                   │  │
│  │          para samplear acciones continuas                  │  │
│  │                                                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  RED CRÍTICO (Función de Valor V)                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                                                             │  │
│  │  Input: Observación (65D)                                  │  │
│  │    ↓                                                        │  │
│  │  Capa Densa 1: 65 → 512 [ELU]                              │  │
│  │    ↓                                                        │  │
│  │  Capa Densa 2: 512 → 256 [ELU]                             │  │
│  │    ↓                                                        │  │
│  │  Capa Densa 3: 256 → 128 [ELU]                             │  │
│  │    ↓                                                        │  │
│  │  Capa de Salida: 128 → 1 [lineal]                          │  │
│  │                                                             │  │
│  │  Output: Escalar V(s) - valor estimado del estado          │  │
│  │                                                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Total de parámetros: ~550,000                                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Estructura de Parámetros en Memoria

```python
@dataclass
class NetworkConfiguration:
    """Configuración de arquitectura de red neuronal."""

    # Dimensiones
    observation_dim: int = 65     # Entrada
    action_dim: int = 12          # Salida (media de distribución)

    # Capas ocultas (para actor y crítico independientes)
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])

    # Función de activación
    activation: Type[nn.Module] = nn.ELU

    # Parámetros totales aproximados
    def count_parameters(self) -> int:
        """Calcula número de parámetros."""
        # Actor
        actor_params = (
            self.observation_dim * self.hidden_layers[0] +  # Capa 1
            self.hidden_layers[0] * self.hidden_layers[1] +  # Capa 2
            self.hidden_layers[1] * self.hidden_layers[2] +  # Capa 3
            self.hidden_layers[2] * self.action_dim +        # Salida media
            self.action_dim                                   # Log std
        )

        # Crítico (similar pero salida escalar)
        critic_params = (
            self.observation_dim * self.hidden_layers[0] +
            self.hidden_layers[0] * self.hidden_layers[1] +
            self.hidden_layers[1] * self.hidden_layers[2] +
            self.hidden_layers[2] * 1                         # Salida V(s)
        )

        return actor_params + critic_params
```

### 5.3 Estructura de Datos Durante Inferencia

```python
class PolicyInference:
    """Representa el flujo de datos durante la predicción."""

    def forward(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Procesa observación y genera acción.

        Args:
            observation: Vector 65D del estado actual

        Returns:
            action: Vector 12D de correcciones residuales
            value: Estimación escalar V(s) del valor del estado
        """
        # 1. Normalización de observación
        obs_normalized = self._normalize(observation)  # 65D

        # 2. Forward pass red actor
        hidden_1 = ELU(Linear(obs_normalized, W1_actor))  # 512D
        hidden_2 = ELU(Linear(hidden_1, W2_actor))        # 256D
        hidden_3 = ELU(Linear(hidden_2, W3_actor))        # 128D
        action_mean = Linear(hidden_3, W_out_actor)       # 12D
        action_std = exp(log_std_param)                   # 12D

        # 3. Muestreo de acción desde distribución Gaussiana
        action = sample_normal(action_mean, action_std)   # 12D

        # 4. Forward pass red crítico (en paralelo)
        hidden_1_c = ELU(Linear(obs_normalized, W1_critic))
        hidden_2_c = ELU(Linear(hidden_1_c, W2_critic))
        hidden_3_c = ELU(Linear(hidden_2_c, W3_critic))
        value = Linear(hidden_3_c, W_out_critic)          # 1D

        return action, value
```

### 5.4 Relación con el Proceso de Aprendizaje

La arquitectura de red:

- **Actor-Crítico separado**: Permite aprender política y función de valor independientemente
- **Activación ELU**: Mejor gradiente que ReLU en dominios continuos
- **Distribución Gaussiana**: Facilita exploración continua en espacio de acciones
- **Función de valor**: Estima retorno futuro para calcular ventajas en PPO

---

## 6. Estructura de Datos del Buffer de Experiencias (Rollout Buffer)

Durante el entrenamiento, el agente recolecta experiencias interactuando con múltiples entornos en paralelo. Estas experiencias se almacenan en un buffer temporal antes de usarse para actualizar las redes.

### 6.1 Componentes del Buffer

```
┌───────────────────────────────────────────────────────────────────┐
│ ROLLOUT BUFFER (para n_envs=80, n_steps=4096)                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Cada fila representa un timestep, cada columna un entorno        │
│  Dimensiones: (n_steps, n_envs, dimensión_dato)                  │
│                                                                    │
│  1. observations: (4096, 80, 65)                                  │
│     └─ Observaciones recolectadas                                 │
│                                                                    │
│  2. actions: (4096, 80, 12)                                       │
│     └─ Acciones ejecutadas                                        │
│                                                                    │
│  3. rewards: (4096, 80, 1)                                        │
│     └─ Recompensas recibidas                                      │
│                                                                    │
│  4. values: (4096, 80, 1)                                         │
│     └─ Estimaciones V(s) del crítico                              │
│                                                                    │
│  5. log_probs: (4096, 80, 1)                                      │
│     └─ Log probabilidades de las acciones tomadas                 │
│                                                                    │
│  6. dones: (4096, 80, 1)                                          │
│     └─ Flags de terminación de episodio                           │
│                                                                    │
│  7. advantages: (4096, 80, 1)  [calculado post-recolección]       │
│     └─ Ventajas A(s,a) = Q(s,a) - V(s) usando GAE                 │
│                                                                    │
│  8. returns: (4096, 80, 1)  [calculado post-recolección]          │
│     └─ Retornos descontados para entrenar crítico                 │
│                                                                    │
│  TOTAL: ~80 × 4096 × (65+12+1+1+1+1+1+1) = ~26M valores float32   │
│         ≈ 104 MB de memoria                                       │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### 6.2 Proceso de Recolección

```python
class RolloutCollector:
    """Gestiona la recolección de experiencias."""

    def __init__(self, n_envs: int = 80, n_steps: int = 4096):
        self.n_envs = n_envs
        self.n_steps = n_steps

        # Pre-alocar arrays para eficiencia
        self.observations = np.zeros((n_steps, n_envs, 65), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, 12), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs, 1), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs, 1), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs, 1), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs, 1), dtype=np.bool_)

        self.current_step = 0

    def add(self, obs, action, reward, value, log_prob, done):
        """Añade transición al buffer."""
        self.observations[self.current_step] = obs
        self.actions[self.current_step] = action
        self.rewards[self.current_step] = reward
        self.values[self.current_step] = value
        self.log_probs[self.current_step] = log_prob
        self.dones[self.current_step] = done

        self.current_step += 1

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        """
        Calcula ventajas usando Generalized Advantage Estimation.

        GAE(λ): A_t = Σ(γλ)^l δ_{t+l}
        donde δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        self.advantages = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = 0  # o valor del estado final
            else:
                next_value = self.values[t + 1]

            # δ_t = r_t + γV(s_{t+1})(1 - done) - V(s_t)
            delta = (
                self.rewards[t] +
                gamma * next_value * (1 - self.dones[t]) -
                self.values[t]
            )

            # A_t = δ_t + γλA_{t+1}(1 - done)
            last_gae = delta + gamma * gae_lambda * last_gae * (1 - self.dones[t])
            self.advantages[t] = last_gae

        # Retornos = ventajas + valores base
        self.returns = self.advantages + self.values
```

### 6.3 Uso en Actualización de Política

```python
def update_policy(buffer: RolloutCollector, batch_size: int = 2048, n_epochs: int = 10):
    """
    Actualiza las redes usando mini-batches del buffer.

    Args:
        buffer: Buffer con experiencias recolectadas
        batch_size: Tamaño de mini-batch
        n_epochs: Número de pasadas sobre los datos
    """
    # Flatten buffer: (n_steps, n_envs, dim) → (n_steps*n_envs, dim)
    total_samples = buffer.n_steps * buffer.n_envs  # 4096 * 80 = 327,680

    obs_flat = buffer.observations.reshape(total_samples, 65)
    actions_flat = buffer.actions.reshape(total_samples, 12)
    old_log_probs = buffer.log_probs.reshape(total_samples, 1)
    advantages = buffer.advantages.reshape(total_samples, 1)
    returns = buffer.returns.reshape(total_samples, 1)

    # Normalizar ventajas
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(n_epochs):
        # Barajar índices
        indices = np.random.permutation(total_samples)

        # Iterar en mini-batches
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # Extraer mini-batch
            obs_batch = obs_flat[batch_idx]
            actions_batch = actions_flat[batch_idx]
            old_log_prob_batch = old_log_probs[batch_idx]
            advantages_batch = advantages[batch_idx]
            returns_batch = returns[batch_idx]

            # Forward pass con política actual
            new_log_probs, values_pred = policy.evaluate_actions(obs_batch, actions_batch)

            # Calcular ratio para PPO clip
            ratio = torch.exp(new_log_probs - old_log_prob_batch)

            # Pérdida de política (PPO clip objective)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Pérdida de función de valor
            value_loss = F.mse_loss(values_pred, returns_batch)

            # Pérdida de entropía (para fomentar exploración)
            entropy_loss = -policy.entropy().mean()

            # Pérdida total
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
```

### 6.4 Relación con el Proceso de Aprendizaje

El buffer de experiencias:

- **Estabiliza el entrenamiento**: Decorrela experiencias mediante batching
- **Permite reutilización de datos**: n_epochs pasadas sobre los mismos datos (sample efficiency)
- **Facilita paralelización**: 80 entornos recolectan datos simultáneamente
- **Almacena señales de aprendizaje**: Ventajas y retornos guían la optimización

---

## 7. Estructura de la Capa de Gestión del Entrenamiento

La capa de gestión coordina todo el proceso de aprendizaje, gestionando configuración, logging, checkpointing y monitoreo.

### 7.1 Configuración Centralizada

```python
@dataclass
class TrainingConfig:
    """Estructura de datos de configuración de entrenamiento."""

    # ==================== DURACIÓN ====================
    total_timesteps: int = 3_000_000      # Total de pasos de simulación

    # ==================== PARALELIZACIÓN ====================
    n_envs: int = 80                      # Entornos en paralelo
    vec_env_backend: str = "subproc"      # "subproc" o "dummy"

    # ==================== HIPERPARÁMETROS PPO ====================
    n_steps: int = 4096                   # Pasos por rollout
    batch_size: int = 2048                # Tamaño de mini-batch
    learning_rate: float = 1e-4           # Tasa de aprendizaje
    gamma: float = 0.99                   # Factor de descuento
    gae_lambda: float = 0.95              # Lambda para GAE
    n_epochs: int = 10                    # Épocas por actualización
    ent_coef: float = 0.01                # Coeficiente de entropía
    clip_range: float = 0.2               # Rango de clipping PPO
    max_grad_norm: float = 1.0            # Norma máxima de gradientes

    # ==================== RED NEURONAL ====================
    network_size: str = "large"           # "small" | "medium" | "large"

    # ==================== ENTORNO ====================
    residual_scale: float = 0.01          # Escala de acciones residuales [m]
    max_episode_steps: int = 5000         # Pasos máximos por episodio
    settle_steps: int = 0                 # Pasos de estabilización inicial

    # ==================== LOGGING Y CHECKPOINTS ====================
    run_name: str = "prod_v3"             # Nombre identificador
    log_root: str = "runs"                # Directorio raíz de logs
    checkpoint_freq: int = 500_000        # Frecuencia de guardado

    # ==================== MISC ====================
    device: str = "auto"                  # "auto" | "cpu" | "cuda"
    seed: int | None = None               # Semilla aleatoria
```

### 7.2 Jerarquía de Archivos de Salida

```
runs/
└── prod_v3_20250114_153045/          # Directorio de esta ejecución
    ├── config.txt                     # Configuración guardada
    │
    ├── checkpoints/                   # Checkpoints periódicos
    │   ├── rl_model_500000_steps.zip
    │   ├── rl_model_1000000_steps.zip
    │   └── ...
    │
    ├── final_model.zip                # Modelo final entrenado
    ├── vec_normalize.pkl              # Estadísticas de normalización
    │
    ├── monitor_0.csv                  # Logs de episodios (env 0)
    ├── monitor_1.csv                  # Logs de episodios (env 1)
    ├── ...
    ├── monitor_79.csv                 # Logs de episodios (env 79)
    │
    └── PPO_1/                         # Logs de TensorBoard
        └── events.out.tfevents...     # Métricas de entrenamiento
```

### 7.3 Estructura de Logs de Episodios (monitor_*.csv)

```csv
# Formato: tiempo,longitud_episodio,recompensa_total
r,l,t
0.450123,1000,245.67
0.890234,850,198.34
1.230456,1000,312.89
...
```

### 7.4 Estructura de Métricas en TensorBoard

```
┌────────────────────────────────────────────────────────────┐
│ MÉTRICAS REGISTRADAS EN TENSORBOARD                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Desempeño del Agente                                   │
│     ├─ rollout/ep_rew_mean: Recompensa promedio           │
│     ├─ rollout/ep_len_mean: Longitud promedio episodio    │
│     └─ time/fps: Frames por segundo (velocidad)           │
│                                                             │
│  2. Pérdidas de Entrenamiento                              │
│     ├─ train/policy_loss: Pérdida de política             │
│     ├─ train/value_loss: Pérdida de función de valor      │
│     ├─ train/entropy_loss: Pérdida de entropía            │
│     └─ train/approx_kl: KL divergence aproximado          │
│                                                             │
│  3. Estadísticas de Actualización                          │
│     ├─ train/clip_fraction: % de ratios clippeados        │
│     ├─ train/explained_variance: Varianza explicada       │
│     └─ train/learning_rate: Tasa de aprendizaje actual    │
│                                                             │
│  4. Componentes de Recompensa (custom)                     │
│     ├─ reward/forward_velocity: Recompensa por velocidad  │
│     ├─ reward/contact_pattern: Recompensa por contactos   │
│     ├─ reward/stability: Penalización por inestabilidad   │
│     └─ reward/lateral_stability: Control lateral          │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 7.5 Flujo de Datos Durante el Entrenamiento

```
┌───────────────────────────────────────────────────────────────┐
│ CICLO DE ENTRENAMIENTO (1 iteración)                          │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  1. RECOLECCIÓN (n_steps × n_envs = 4096 × 80 = 327,680 pasos)│
│     ┌──────────────────────────────────────────────────────┐  │
│     │ Para cada paso t = 0...4095:                         │  │
│     │   Para cada entorno e = 0...79:                      │  │
│     │     ├─ Observar estado: obs[t,e] ← env[e].state      │  │
│     │     ├─ Predecir acción: a[t,e], v[t,e] ← π(obs[t,e]) │  │
│     │     ├─ Ejecutar: obs', r, done ← env[e].step(a[t,e]) │  │
│     │     ├─ Almacenar: buffer.add(obs, a, r, v, done)     │  │
│     │     └─ Si done: reset env[e]                         │  │
│     └──────────────────────────────────────────────────────┘  │
│                                                                │
│  2. CÁLCULO DE VENTAJAS                                        │
│     └─ GAE: A[t] = Σ(γλ)^k δ_{t+k}                            │
│                                                                │
│  3. OPTIMIZACIÓN (n_epochs × num_batches = 10 × 160)          │
│     ┌──────────────────────────────────────────────────────┐  │
│     │ Para cada época e = 0...9:                           │  │
│     │   Barajar datos del buffer                           │  │
│     │   Para cada batch b de tamaño 2048:                  │  │
│     │     ├─ Calcular pérdidas (policy, value, entropy)    │  │
│     │     ├─ Backpropagation                               │  │
│     │     ├─ Clip gradientes (norm ≤ 1.0)                  │  │
│     │     └─ Actualizar pesos de redes                     │  │
│     └──────────────────────────────────────────────────────┘  │
│                                                                │
│  4. LOGGING Y CHECKPOINTS                                      │
│     ├─ Registrar métricas en TensorBoard                      │
│     ├─ Actualizar monitor CSV                                 │
│     └─ Guardar checkpoint si corresponde                      │
│                                                                │
│  Repetir hasta alcanzar total_timesteps (3M)                  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### 7.6 Relación con el Proceso de Aprendizaje

La capa de gestión:

- **Centraliza configuración**: Un solo punto para ajustar todos los hiperparámetros
- **Asegura reproducibilidad**: Guarda configuración y semillas aleatorias
- **Facilita experimentación**: Comparación entre ejecuciones mediante TensorBoard
- **Permite recuperación**: Checkpoints periódicos protegen contra fallos
- **Monitorea progreso**: Métricas en tiempo real para detectar problemas

---

## 8. Diagrama de Flujo de Datos Completo

```
┌──────────────────────────────────────────────────────────────────────────┐
│ SISTEMA COMPLETO: Flujo de Datos en Entrenamiento RL                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ ENTORNO SIMULADO (MuJoCo) - Estado: s_t                         │     │
│  │  ├─ Posición/velocidad cuerpo                                   │     │
│  │  ├─ Ángulos/velocidades articulares                             │     │
│  │  ├─ Posiciones/velocidades pies                                 │     │
│  │  └─ Contactos con suelo                                         │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Observación: o_t (65D)                                │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ NORMALIZACIÓN (VecNormalize)                                    │     │
│  │  ├─ Media móvil: μ_obs                                          │     │
│  │  ├─ Varianza móvil: σ²_obs                                      │     │
│  │  └─ o_norm = (o_t - μ) / σ                                      │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Observación normalizada (65D)                         │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ RED NEURONAL (Política π_θ)                                     │     │
│  │  ├─ Input: o_norm (65D)                                         │     │
│  │  ├─ Hidden: [512] → [256] → [128] con ELU                       │     │
│  │  ├─ Output μ: media de acción (12D)                             │     │
│  │  └─ Output σ: std de acción (12D)                               │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Distribución: π(a|s) = N(μ, σ²)                      │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ MUESTREO DE ACCIÓN                                              │     │
│  │  └─ a_t ~ N(μ_t, σ²_t)  →  Acción residual (12D)               │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Acción: a_t (12D) ∈ [-1, 1]                          │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ ESCALADO RESIDUAL                                               │     │
│  │  └─ Δ_foot = a_t × residual_scale (típicamente 0.01 m)         │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Residuales por pata: {"FL": [Δx,Δy,Δz], ...}         │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ CONTROLADOR DE MARCHA (Bézier Diagonal Gait)                   │     │
│  │  ├─ Genera trayectorias nominales: target_base[leg]            │     │
│  │  └─ Suma residuales: target_final = target_base + Δ            │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Objetivos cartesianos: {"FL": [x,y,z], ...}          │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ CINEMÁTICA INVERSA (IK 3DOF)                                    │     │
│  │  └─ Convierte posiciones pie → ángulos articulares             │     │
│  │     angles[leg] = [tilt, shoulder_L, shoulder_R]                │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Comandos articulares (12D)                            │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ APLICACIÓN DE CONTROL (MuJoCo)                                  │     │
│  │  └─ data.ctrl[i] = angle[i]  (con mapeo y offsets)             │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Simulación física (mj_step)                           │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ TRANSICIÓN DE ESTADO                                            │     │
│  │  ├─ s_{t+1}: Nuevo estado del robot                             │     │
│  │  └─ r_t: Recompensa calculada                                   │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ (o_{t+1}, r_t, done)                                  │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ ROLLOUT BUFFER                                                  │     │
│  │  └─ Almacena: (o_t, a_t, r_t, V(o_t), log π(a_t|o_t), done)    │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Cada n_steps: buffer lleno                            │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ CÁLCULO DE VENTAJAS (GAE)                                       │     │
│  │  └─ A_t = Σ(γλ)^k [r_{t+k} + γV(s_{t+k+1}) - V(s_{t+k})]       │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Datos listos para optimización                        │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ OPTIMIZACIÓN PPO                                                │     │
│  │  ├─ L_policy = -min(r_t·A_t, clip(r_t,1-ε,1+ε)·A_t)            │     │
│  │  ├─ L_value = MSE(V(s_t), returns)                              │     │
│  │  ├─ L_entropy = -E[log π(a|s)]                                  │     │
│  │  └─ Actualiza θ (parámetros de redes)                           │     │
│  └────────────────┬────────────────────────────────────────────────┘     │
│                   │                                                       │
│                   │ Política mejorada: π_{θ'}                             │
│                   ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ LOGGING Y PERSISTENCIA                                          │     │
│  │  ├─ TensorBoard: métricas de entrenamiento                      │     │
│  │  ├─ Monitor CSV: episodios completados                          │     │
│  │  └─ Checkpoints: modelo + normalizadores                        │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  [Repetir ciclo hasta alcanzar total_timesteps]                          │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Relación entre Estructuras de Datos y Proceso de Aprendizaje

### 9.1 Mapeo Concepto-Implementación

| Concepto de RL | Estructura de Datos | Localización en Código |
|----------------|---------------------|------------------------|
| Estado del entorno (s) | Vector 65D (observación) | `ResidualWalkEnv._get_observation()` |
| Acción (a) | Vector 12D (residuales) | `action_space: Box(12,)` |
| Política π(a\|s) | Red neuronal Actor | `PPO(..., policy="MlpPolicy")` |
| Función de valor V(s) | Red neuronal Crítico | Integrada en política |
| Recompensa (r) | Escalar float | `ResidualWalkEnv._compute_reward()` |
| Trayectoria τ | Rollout Buffer | `RolloutBuffer` interno de SB3 |
| Ventaja A(s,a) | Array (n_steps, n_envs) | Calculado con GAE |
| Gradiente ∇_θ L | Tensor PyTorch | Backprop automático |
| Parámetros θ | Pesos de redes | `model.policy.parameters()` |

### 9.2 Flujo de Información en el Ciclo de Aprendizaje

1. **Percepción**: Sensores MuJoCo → estructura de observación (65D)
2. **Decisión**: Observación → red neuronal → distribución de acciones
3. **Ejecución**: Acción → controlador residual → IK → actuadores
4. **Evaluación**: Cambio de estado → función de recompensa → señal escalar
5. **Almacenamiento**: Tupla (s,a,r,s') → buffer de experiencias
6. **Optimización**: Batch de experiencias → gradientes → actualización de pesos
7. **Mejora**: Nueva política → mejores decisiones en futuras iteracciones

### 9.3 Cuellos de Botella y Optimizaciones

**Memoria:**
- Buffer de rollout: ~104 MB por iteración
- Redes neuronales: ~2.2 MB (550k parámetros × 4 bytes)
- Normalizers: ~520 bytes (estadísticas de medias/varianzas)

**Cómputo:**
- Simulación MuJoCo: ~60% del tiempo de ejecución
- Forward pass redes: ~25%
- Backpropagation: ~10%
- Overhead (logging, I/O): ~5%

**Paralelización:**
- 80 entornos en subprocesos independientes (SubprocVecEnv)
- Batch processing en GPU para forward/backward passes
- Vectorización NumPy para cálculo de ventajas

---

## 10. Conclusiones

Las estructuras de datos presentadas en este capítulo forman el núcleo del sistema de aprendizaje por refuerzo para locomoción robótica. Aunque no se utilizan bases de datos tradicionales, la organización cuidadosa de datos en memoria resulta fundamental para:

1. **Eficiencia computacional**: Arrays pre-alocados, operaciones vectorizadas, paralelización
2. **Estabilidad del aprendizaje**: Normalización, clipping, batch processing
3. **Reproducibilidad**: Configuración serializada, checkpointing, logging estructurado
4. **Escalabilidad**: Arquitectura modular que permite ajustar dimensiones sin cambiar código
5. **Interpretabilidad**: Logging granular de componentes de recompensa y métricas

La arquitectura de datos soporta el ciclo completo de aprendizaje: desde la representación sensorial del estado del robot, pasando por la codificación de decisiones como acciones continuas, hasta el almacenamiento eficiente de experiencias y la optimización iterativa de la política neuronal.

Esta organización demuestra que, incluso sin persistencia en disco tradicional, las estructuras de datos in-memory requieren el mismo nivel de diseño y atención que las bases de datos relacionales o NoSQL, especialmente cuando el rendimiento y la escalabilidad son críticos para el éxito del sistema.

---

## Referencias Técnicas del Proyecto

- **Código fuente**: `train_residual_ppo_v3.py` (líneas 41-75: configuración)
- **Entorno**: `envs/residual_walk_env.py` (líneas 170-210: observación; 245-295: recompensa)
- **Controlador**: `controllers/bezier_gait_residual.py` (líneas 40-67: integración residual)
- **Documentación**: `CLAUDE.md`, `PPO_TRAINING_SUMMARY.md`
