# Robot Cuadrúpedo con Control Adaptativo

## TFG Ricardo Casimiro

Simulación de robot cuadrúpedo 12DOF con MuJoCo, Gymnasium, ROS2 y aprendizaje por refuerzo profundo PPO(StableBaseline3)

El desarrollo CAD del robot es propio(OpenScad) y puede ser usado en la vida real(no testeado)

El robot implementa un diseño de patas SCARA paralelo de 3DOF(5BarLinkage), cinematica inversa 3DOF, curvas de Bezier para locomocion (marchas), y control adaptativo mediante residuales aprendidos durante el entrenamiento

El test principal compara un controlador de marcha puramente cinemático (baseline) con un controlador adaptativo entrenado con aprendizaje por refuerzo (RL) para navegación en terreno irregular.

## [Video Demo](https://drive.google.com/file/d/1ZFX_Mz6WEISDz5IWsRfotk7-QPtXtzc1/view)



## Características Principales

- **Control Cinemático de Marcha Diagonal**: Generador de marcha tipo trote con pares de patas diagonales
- **Control Adaptativo con RL**: Política entrenada con Stable Baselines 3 (PPO) que añade correcciones residuales a la marcha baseline
- **Simulación Física**: Motor MuJoCo para física realista y renderizado
- **Integración ROS2**: Comunicación entre simulación y GUI mediante topics y servicios
- **GUI con Joystick**: Interfaz PyQt5 con soporte para control mediante joystick
- **Terrenos Múltiples**: Entorno plano y terreno irregular para entrenamiento/evaluación

https://github.com/user-attachments/assets/0e2e2d83-e022-4127-b9a3-516ac36cb123

## Instalación

### 1. Dependencias del Sistema

Primero, instala ROS2 Jazzy (en Ubuntu 24.04):

```bash
# Configurar repositorios de ROS2
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Instalar ROS2 Jazzy
sudo apt update
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions

# Dependencias de ROS2 para Python
sudo apt install ros-jazzy-rclpy ros-jazzy-std-msgs ros-jazzy-sensor-msgs ros-jazzy-std-srvs ros-jazzy-cv-bridge
```

### 2. Dependencias de Python

```bash
# Clonar repositorio
git clone <url-del-repo>
cd walk2

# Instalar dependencias de Python
pip install -r requirements.txt
```

Las dependencias incluyen:
- **Simulación**: mujoco, numpy, scipy
- **Control de marcha**: transitions, bezier
- **Aprendizaje por refuerzo**: stable-baselines3, torch
- **Visualización**: matplotlib, pandas
- **GUI**: PyQt5, pyqtgraph, pygame, pillow

### 3. Configurar Entorno ROS2

Para usar la simulación con ROS2 (GUI y comunicación), siempre ejecuta primero:

```bash
source /opt/ros/jazzy/setup.bash
```

## Uso

### Test Principal: Comparación Baseline vs Adaptativo

Este test ejecuta **tres simulaciones** consecutivas para demostrar la efectividad del control adaptativo:

```bash
python3 tests/compare_baseline_adaptive.py \
    --model runs/adaptive_gait_20251115_180640/final_model.zip \
    --normalize runs/adaptive_gait_20251115_180640/vec_normalize.pkl \
    --seconds 17
```

#### ¿Qué hace este test?

El script ejecuta automáticamente tres casos y genera gráficas comparativas:

**Paso 1: Baseline en Terreno Plano** (Izquierda - Azul)
- Usa el controlador cinemático puro sin RL
- Terreno completamente plano (`world.xml`)
- **Resultado esperado**: Funciona bien, progresión lineal estable
- Demuestra que el modelo cinemático baseline funciona correctamente en condiciones ideales

**Paso 2: Baseline en Terreno Irregular** (Centro - Rojo)
- Mismo controlador cinemático puro
- Terreno con elevaciones y depresiones aleatorias (`world_train.xml`)
- **Resultado esperado**: Rendimiento degradado significativamente
- El robot avanza mucho menos debido a que el controlador no se adapta a las irregularidades
- Muestra pérdida de eficiencia de ~40-60% comparado con terreno plano

**Paso 3: Adaptativo RL en Terreno Irregular** (Derecha - Verde)
- Controlador baseline + correcciones de política entrenada con RL
- Mismo terreno irregular que Paso 2
- **Resultado esperado**: Mejora sustancial, alcanza distancia mucho mayor
- El modelo adaptativo añade ajustes dinámicos que compensan las irregularidades
- Recupera gran parte del rendimiento perdido, mejorando ~80-150% vs Paso 2

#### Salida del Test

El script genera:

1. **Terminal**: Tabla comparativa con métricas
   ```
    ===============================================================================================
    COMPARISON SUMMARY - THREE SIMULATIONS
    ===============================================================================================

    Metric                         Step 1: Baseline     Step 2: Baseline     Step 3: Adaptive    
                                   (Flat)               (Rough)              (Rough)             
    -----------------------------------------------------------------------------------------------
    Duration (s)                   17.00                17.00                17.00               
    Data points                    9069                 8966                 7128                
    Start X (m)                    0.000                0.000                0.000               
    End X (m)                      0.506                0.299                3.191               
    Distance traveled (m)          0.506                0.299                3.191               
    Average velocity (m/s)         0.030                0.018                0.188               

    Performance Comparison:
      Step 2 vs Step 1 (Rough vs Flat):      -40.9%
      Step 3 vs Step 2 (Adaptive vs Rough):  +967.2%
    ===============================================================================================
   ```
   


2. **Gráfica de 3 paneles**: `tests/baseline_vs_adaptive_comparison.png`
   - Cada panel muestra tiempo vs posición X
   - Marcadores de inicio (verde) y fin (rojo)
   - Cuadro de estadísticas con distancia, velocidad y % de cambio


<img width="2679" height="728" alt="Image" src="https://github.com/user-attachments/assets/2b51f4d2-c070-4e63-8125-2ee96cce5359" />

3. **Archivos de trayectoria JSON**:
   - `tests/trajectory_step1_baseline_flat.json`
   - `tests/trajectory_step2_baseline_rough.json`
   - `tests/trajectory_step3_adaptive_rough.json`

### Simulación Simple Standalone

Para ejecutar una simulación básica sin ROS2 ni GUI:

```bash
# Simulación con terreno irregular (default)
python3 height_control.py

# Modo headless (sin ventana, útil para CI/testing)
MUJOCO_GL=egl timeout 60 python3 height_control.py
```

Este script:
- Carga el modelo del robot y el terreno rugoso
- Ejecuta el controlador de marcha diagonal
- Muestra la simulación en tiempo real
- Presiona `Espacio` para pausar/reanudar

### Simulación con GUI y Joystick

Para usar la interfaz gráfica con control por joystick, necesitas **dos terminales**:

#### Terminal 1: Iniciar GUI

```bash
source /opt/ros/jazzy/setup.bash
cd gui
python3 gui.py
```

La GUI mostrará:
- Feed de cámara en primera persona desde el robot
- Controles de joystick (si hay gamepad conectado)
- Botón para reiniciar simulación

#### Terminal 2: Iniciar Simulación

```bash
source /opt/ros/jazzy/setup.bash
python3 sim.py                    # Terreno plano (default)
# O bien:
python3 sim.py --terrain rough    # Terreno irregular
```

#### Control con Joystick

Una vez ambos procesos estén corriendo:
- **Stick analógico**: Adelante/Atrás para movimiento
- **Botón X en GUI**: Reinicia la simulación
- Los comandos se envían vía topic ROS2 `/movement_command`:
  - `0` = detener (congela la marcha)
  - `1` = adelante (marcha normal)
  - `2` = atrás (invierte dirección)

#### Topics ROS2 Disponibles

```bash
# Monitorear cámara (10 Hz, 640x480 RGB)
ros2 topic hz /robot_camera

# Ver comandos de movimiento
ros2 topic echo /movement_command

# Ver posición y orientación del robot [x, y, z, roll, pitch, yaw]
ros2 topic echo /body_state

# Reiniciar simulación manualmente
ros2 service call /restart_simulation std_srvs/Trigger
```

## Estructura del Proyecto

```
walk2/
├── model/                        # Modelos MuJoCo
│   ├── world.xml                # Terreno plano
│   ├── world_train.xml          # Terreno irregular (heightfield)
│   ├── robot.xml                # Definición del robot cuadrúpedo
│   └── assets/                  # Mallas STL de las piezas
├── controllers/
│   ├── gait_controller.py       # Controlador de marcha diagonal baseline
│   └── adaptive_gait_controller.py  # Wrapper para RL + baseline
├── envs/
│   └── adaptive_gait_env.py     # Entorno Gym para entrenamiento RL
├── utils/
│   ├── ik.py                    # Cinemática inversa 3DOF SCARA paralelo
│   ├── control_utils.py         # Utilidades de control
│   └── sensor_utils.py          # Procesamiento de sensores
├── gui/
│   └── gui.py                   # Interfaz PyQt5 con joystick
├── tests/
│   ├── compare_baseline_adaptive.py  # Test comparativo principal
│   └── baseline_vs_adaptive_comparison.png  # Gráfica de resultados
├── runs/                        # Modelos entrenados
│   └── adaptive_gait_20251115_180640/
│       ├── final_model.zip      # Política PPO entrenada
│       └── vec_normalize.pkl    # Estadísticas de normalización
├── height_control.py            # Simulación standalone
├── sim.py                       # Simulación con ROS2
├── play_adaptive_policy.py      # Ejecutar política entrenada
```

## Desarrollo

### Verificación de IK y Control

```bash
# Test de cinemática inversa
python3 utils/ik.py

# Test de controlador de marcha
python3 controllers/gait_controller.py

# Calcular espacio de trabajo alcanzable
python3 foot_range_calculator.py
```

### Depuración ROS2

```bash
# Listar nodos activos (debería mostrar robot_control_node y gui_ros_node)
ros2 node list

# Listar topics activos
ros2 topic list

# Publicar comando manualmente
ros2 topic pub /movement_command std_msgs/Int32 "data: 1"
```

### Entrenamiento de Nuevos Modelos

Para entrenar un nuevo modelo adaptativo, revisa los scripts en `envs/adaptive_gait_env.py` y configura el entrenamiento con Stable Baselines 3 (PPO).

## Notas Técnicas

### Marcos de Coordenadas

- **IK de pata**: Frame local donde Z apunta hacia abajo (dirección de gravedad)
- **Controlador de marcha**: +X = adelante, pero se invierte mediante `FORWARD_SIGN = -1.0` para coincidir con el frame de IK de la pata
- **Mundo MuJoCo**: Sistema de coordenadas global estándar

### Parámetros de Marcha

Configurados en `gait_controller.py`:
- `body_height = 0.08` m
- `step_length = 0.04` m
- `step_height = 0.02` m
- `cycle_time = 1.2` s
- Trayectoria de swing: Curva Bézier cúbica
- Trayectoria de stance: Barrido lineal

### Parámetros de IK

- `L1 = 0.045` m (eslabón superior)
- `L2 = 0.06` m (eslabón inferior)
- `base_dist = 0.021` m (separación entre brazos paralelos)
- Modo predeterminado: 2 (brazo A abajo, brazo B arriba)

## Referencias

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [ROS2 Jazzy Documentation](https://docs.ros.org/en/jazzy/)
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)


