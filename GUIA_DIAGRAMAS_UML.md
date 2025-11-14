# Guía de Diagramas UML para Tesis

## 📐 ¿Qué son estos diagramas?

Los **diagramas UML (Unified Modeling Language)** son el estándar internacional para documentar la arquitectura de software. Son formalmente aceptados en tesis académicas y publicaciones científicas.

A diferencia de los diagramas anteriores que generé (que eran visualizaciones genéricas), estos son **diagramas UML formales** que representan las estructuras de datos de tu proyecto de forma profesional.

---

## 🎯 Tipos de Diagramas Generados

### 1. **Diagrama de Clases: Estructuras de Datos Principales**
**Archivo:** `uml_diagrams/01_estructuras_datos.png` (167.5 KB)

**Tipo:** Diagrama de Clases UML

**Qué muestra:**
- `ObservationVector` (65D) - estructura del vector de observación
- `ActionVector` (12D) - estructura del vector de acción residual
- `GaitParameters` - parámetros de configuración de marcha
- `RolloutBuffer` - estructura del buffer de experiencias (4096×80)
- `TrainingConfig` - configuración de hiperparámetros PPO
- `TrainingMetrics` - métricas registradas durante entrenamiento

**Relaciones entre estructuras:**
- Agregación (◇): Una estructura contiene a otra
- Dependencia (⇢): Una estructura usa a otra
- Composición (◆): Una estructura es parte esencial de otra

**Uso en la tesis:**
> "La Figura X muestra el diagrama de clases UML de las estructuras de datos principales almacenadas en memoria durante el proceso de aprendizaje por refuerzo. Cada clase representa una estructura de datos con sus atributos (dimensiones, tipos) y métodos (operaciones disponibles)."

---

### 2. **Diagrama de Clases: Sistema Completo**
**Archivo:** `uml_diagrams/02_sistema_completo.png` (173.4 KB)

**Tipo:** Diagrama de Clases UML con Paquetes

**Qué muestra:**
- **Paquete "Capa de Simulación"**: `MuJoCoSimulator`
- **Paquete "Capa de Control"**:
  - `BezierGaitResidualController`
  - `DiagonalGaitController`
  - `IKSolver` (cinemática inversa)
  - `ControlUtils`
- **Paquete "Capa de Aprendizaje"**:
  - `ResidualWalkEnv` (entorno Gymnasium)
  - `SensorReader`
  - `PPOAgent`
  - `ActorCriticPolicy`
  - `MLP` (red neuronal)
- **Paquete "Capa de Gestión"**:
  - `VecEnv` (entornos vectorizados)
  - `VecNormalize` (normalización)
  - `CheckpointCallback`
  - `TensorBoardLogger`

**Relaciones clave:**
- Composición: `ResidualWalkEnv` contiene `BezierGaitResidualController`
- Agregación: `PPOAgent` usa `ResidualWalkEnv`
- Dependencia: `ControlUtils` modifica `MuJoCoSimulator`

**Uso en la tesis:**
> "La Figura X ilustra la arquitectura completa del sistema mediante un diagrama de clases UML organizado en cuatro paquetes que representan las capas de abstracción: simulación física, control, aprendizaje y gestión. Las flechas indican las relaciones de dependencia y composición entre componentes."

---

### 3. **Diagrama de Secuencia: Flujo de Entrenamiento**
**Archivo:** `uml_diagrams/03_flujo_entrenamiento.png` (212.5 KB)

**Tipo:** Diagrama de Secuencia UML

**Qué muestra:**
- Secuencia temporal completa de **una iteración de entrenamiento PPO**
- Interacciones entre todos los componentes del sistema
- Flujo de datos desde observación hasta actualización de pesos

**Fases mostradas:**
1. **Inicialización**: Creación de entornos, redes neuronales, buffer
2. **Recolección** (4096 pasos):
   - Observación del entorno
   - Predicción de acción por red neuronal
   - Ejecución de acción (controlador + IK + simulación)
   - Cálculo de recompensa
   - Almacenamiento en buffer
3. **Cálculo de ventajas** (GAE):
   - Procesamiento del buffer completo
4. **Optimización** (10 épocas × 160 mini-batches):
   - Forward pass
   - Cálculo de pérdidas (policy, value, entropy)
   - Backpropagation
   - Actualización de pesos
5. **Logging**: Registro en TensorBoard y checkpointing

**Uso en la tesis:**
> "La Figura X presenta un diagrama de secuencia UML que detalla el flujo temporal de datos y control durante una iteración completa del algoritmo PPO. Las flechas verticales representan el tiempo progresando hacia abajo, mientras que las flechas horizontales muestran mensajes (llamadas a métodos) entre componentes."

---

### 4. **Diagrama de Clases: Relación Datos-Aprendizaje**
**Archivo:** `uml_diagrams/04_relacion_datos_aprendizaje.png` (189.6 KB)

**Tipo:** Diagrama de Clases UML con Mapeo Conceptual

**Qué muestra:**
- **Paquete "Conceptos Teóricos de RL"** (abstractos):
  - Estado (s)
  - Acción (a)
  - Recompensa (r)
  - Política π(a|s)
  - Función de Valor V(s)
  - Ventaja A(s,a)
  - Trayectoria τ

- **Paquete "Implementación en Memoria"** (concretos):
  - `ObservationVector` [ndarray 65D]
  - `ActionVector` [ndarray 12D]
  - `RewardScalar` [float32]
  - `ActorCriticPolicy` [PyTorch Module]
  - `CriticNetwork` [PyTorch Module]
  - `AdvantageArray` [ndarray (4096,80,1)]
  - `RolloutBuffer` [múltiples ndarrays]
  - `NormalizationStats` [RunningMeanStd]

- **Relaciones de realización** (líneas punteadas):
  - Muestran cómo cada concepto teórico se implementa en memoria

- **Relaciones de uso** (flechas sólidas):
  - Muestran el flujo de datos durante el aprendizaje

**Uso en la tesis:**
> "La Figura X establece el mapeo directo entre los conceptos teóricos de aprendizaje por refuerzo (paquete superior) y sus implementaciones concretas en estructuras de datos en memoria (paquete inferior). Las relaciones <<implementa>> muestran cómo cada concepto abstracto se materializa en código, mientras que las flechas sólidas ilustran el flujo de información durante el ciclo de aprendizaje."

---

## 📊 Comparación: DER vs NoSQL vs UML

| Característica | DER | NoSQL | UML (Clases) |
|----------------|-----|-------|--------------|
| **Propósito** | Modelar bases de datos relacionales | Modelar BD no relacionales | Modelar estructuras en memoria/código |
| **Elementos** | Tablas, llaves, relaciones | Documentos, colecciones | Clases, atributos, métodos |
| **Persistencia** | Disco (SQL) | Disco (MongoDB, etc.) | Memoria (RAM) |
| **Apropiado para RL** | ❌ No | ❌ No | ✅ **Sí** |
| **Aceptación académica** | ✅ Alta (para BBDD) | ✅ Media (para NoSQL) | ✅ **Muy Alta** (para software) |
| **Tu caso** | No hay BD relacional | No hay BD NoSQL | **Ideal para tu proyecto** |

---

## ✅ Ventajas de Usar UML en Tu Tesis

### 1. **Formalidad Académica**
- UML es un estándar ISO/IEC 19505
- Usado en ingeniería de software a nivel mundial
- Ampliamente aceptado en publicaciones científicas

### 2. **Precisión Técnica**
- Notación exacta para tipos de datos (`ndarray[65]`, `Dict[str, float]`)
- Relaciones claras (agregación, composición, dependencia)
- Métodos y atributos explícitos

### 3. **Claridad Visual**
- Organización en paquetes (capas de arquitectura)
- Colores para diferenciar componentes
- Notas explicativas integradas

### 4. **Cumple Requisitos del Profesor**

| Requisito | Cómo lo cumple el UML |
|-----------|------------------------|
| "Estructura de datos en memoria" | ✅ Diagrama 1: muestra cada estructura (ObservationVector, ActionVector, Buffer) con dimensiones exactas |
| "Relación con aprendizaje" | ✅ Diagrama 4: mapea conceptos teóricos RL ↔ estructuras concretas |
| "Capa de gestión" | ✅ Diagrama 2: paquete "Capa de Gestión" con TrainingConfig, VecEnv, Callbacks |

---

## 📝 Cómo Incluir en Tu Documento de Tesis

### Opción 1: Una figura por subsección

```latex
\subsection{Estructuras de Datos Principales}

Las estructuras de datos fundamentales del sistema se organizan según
su función en el proceso de aprendizaje. La Figura \ref{fig:uml_estructuras}
muestra el diagrama de clases UML de estas estructuras.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\textwidth]{uml_diagrams/01_estructuras_datos.png}
    \caption{Diagrama UML de clases: estructuras de datos principales almacenadas en memoria}
    \label{fig:uml_estructuras}
\end{figure}

Como se observa en la figura, el vector de observación (ObservationVector)
tiene una dimensión total de 65, compuesta por...
```

### Opción 2: Integrar múltiples figuras

```latex
\section{Estructura de Datos del Sistema}

\subsection{Arquitectura General}
La arquitectura del sistema se organiza en cuatro capas (Figura \ref{fig:uml_sistema})...

\subsection{Flujo de Datos Durante Entrenamiento}
El proceso de entrenamiento sigue una secuencia bien definida (Figura \ref{fig:uml_flujo})...

\subsection{Mapeo entre Teoría e Implementación}
Cada concepto teórico de RL se implementa mediante estructuras concretas
(Figura \ref{fig:uml_mapeo})...
```

### Opción 3: Para Word

1. **Insertar imagen:**
   - Insertar → Imagen → seleccionar archivo PNG
   - Ajustar tamaño (recomendado: ancho = ancho de página)

2. **Agregar epígrafe:**
   - Click derecho → Insertar título
   - "Figura X: Diagrama UML de clases..."

3. **Referencia cruzada:**
   - "Como se muestra en la Figura X..."
   - Insertar → Referencia cruzada → seleccionar figura

---

## 🎓 Ejemplo de Texto Académico

### Fragmento para tu tesis:

> **3.2 Arquitectura de Datos del Sistema**
>
> El sistema de aprendizaje por refuerzo para locomoción robótica se fundamenta en un conjunto de estructuras de datos organizadas jerárquicamente. Si bien el sistema no emplea bases de datos tradicionales para persistencia, las estructuras en memoria requieren un diseño cuidadoso para garantizar eficiencia computacional y correcta representación del estado del agente.
>
> La Figura 3.1 presenta el diagrama de clases UML de las estructuras de datos principales. El `ObservationVector` encapsula el estado perceptual del agente en un arreglo de 65 dimensiones, incluyendo la pose del cuerpo (posición y orientación), velocidades lineales y angulares, configuración articular y estado de contacto con el suelo. Esta representación proporciona información suficiente para que la política neuronal pueda inferir la dinámica del sistema.
>
> El `ActionVector` (12 dimensiones) codifica las decisiones del agente como correcciones residuales sobre un controlador de marcha base. Esta arquitectura de control residual reduce significativamente el espacio de búsqueda comparado con aprender la marcha desde cero.
>
> El `RolloutBuffer` almacena temporalmente 327,680 transiciones (4096 pasos × 80 entornos paralelos) antes de cada actualización de política. Esta estructura multidimensional facilita el cálculo eficiente de ventajas mediante Generalized Advantage Estimation (GAE) y permite mini-batch stochastic gradient descent.
>
> La relación entre estas estructuras y el proceso de aprendizaje se detalla en la Figura 3.4, que establece el mapeo directo entre conceptos teóricos de aprendizaje por refuerzo (estados, acciones, políticas) y sus implementaciones concretas en memoria.

---

## 🔧 Editabilidad

Si necesitas modificar los diagramas:

1. **Editar el archivo `.puml`** (son archivos de texto)
   ```bash
   nano uml_diagrams/01_estructuras_datos.puml
   ```

2. **Re-renderizar**
   ```bash
   python3 renderizar_uml.py
   ```

3. **Sintaxis básica PlantUML:**
   ```plantuml
   class NombreClase {
       + atributo_publico: tipo
       - atributo_privado: tipo
       --
       + metodo_publico(): tipo_retorno
   }

   ClaseA --> ClaseB : relación
   ClaseA *-- ClaseB : composición
   ClaseA o-- ClaseB : agregación
   ```

---

## 📚 Referencias para la Tesis

Puedes citar:

> **Lenguaje de Modelado Unificado (UML)**
>
> - Object Management Group (OMG). (2017). *OMG Unified Modeling Language (OMG UML), Version 2.5.1*. Retrieved from https://www.omg.org/spec/UML/2.5.1/
>
> - Fowler, M. (2003). *UML Distilled: A Brief Guide to the Standard Object Modeling Language* (3rd ed.). Addison-Wesley Professional.

Para diagramas de sistemas de RL:

> - Dulac-Arnold, G., et al. (2019). "Challenges of Real-World Reinforcement Learning". *ICML Workshop on RL4RealLife*.

---

## ✨ Resumen

### Lo que tienes ahora:

✅ **4 diagramas UML profesionales** (formato estándar ISO)
✅ **Archivos fuente editables** (.puml)
✅ **Script de renderizado automático** (renderizar_uml.py)
✅ **Documentación completa** (este archivo)

### Tipos de diagramas:

1. **Diagrama de Clases** - Estructuras de datos
2. **Diagrama de Clases con Paquetes** - Sistema completo
3. **Diagrama de Secuencia** - Flujo temporal de entrenamiento
4. **Diagrama de Clases** - Mapeo teoría-implementación

### Por qué son mejores que los anteriores:

| Característica | Diagramas Anteriores | Diagramas UML |
|----------------|----------------------|---------------|
| Estándar | Visualizaciones custom | ISO/IEC 19505 (UML) |
| Tipo | Genéricos | Formales (Clases, Secuencia) |
| Apropiado para | Presentaciones | Tesis académica |
| Editabilidad | Código Python | Texto PlantUML |
| Tamaño archivo | 183-322 KB | 167-212 KB |
| Uso académico | ⚠️ Informal | ✅ **Formal** |

---

## 🚀 Próximos Pasos

1. **Revisa los diagramas generados** en `uml_diagrams/*.png`
2. **Lee el documento de tesis** (`ESTRUCTURA_DATOS_TESIS.md`)
3. **Integra los diagramas UML** en tu documento en lugar de los genéricos
4. **Adapta el texto** según el estilo de tu institución
5. **Valida con tu profesor** que estos diagramas UML cumplen los requisitos

---

¿Necesitas que genere más diagramas UML o modifique los existentes?
