# Resumen de diagramas UML simplificados

1. **01_estructuras_datos.puml**  
   Resume las estructuras que manejan observaciones, acciones, parámetros de marcha y almacenamiento de experiencias. Muestra cómo `RolloutBuffer` agrupa vectores de observación y acción, y cómo `TrainingConfig` define dimensiones y límites para el entrenamiento residual PPO.

2. **02_sistema_completo.puml**  
   Ofrece una vista por capas (simulación, control, aprendizaje y gestión) del sistema completo. Solo conserva las clases principales y sus asociaciones para entender el flujo desde MuJoCo y controladores Bézier hasta el agente PPO y los servicios de logging/checkpoints.

3. **03_flujo_entrenamiento.puml**  
   Diagrama de secuencia reducido que sigue una iteración típica de entrenamiento: inicialización del script, recolección de pasos en paralelo, cálculo de ventajas GAE, optimización por épocas y registro de métricas/checkpoints.

4. **04_relacion_datos_aprendizaje.puml**  
   Mapea conceptos teóricos de RL (estado, acción, política, etc.) con sus estructuras concretas (`ObservationVector`, `ActionVector`, `RolloutBuffer`, etc.) y señala el flujo básico de datos: normalización, inferencia, evaluación de recompensas y actualización de la política.
