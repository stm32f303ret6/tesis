#!/usr/bin/env python3
"""
Genera diagramas visuales para la sección de estructura de datos de la tesis.

Este script crea visualizaciones de:
1. Flujo de datos del sistema completo
2. Arquitectura de la red neuronal
3. Estructura del vector de observación
4. Estructura del buffer de experiencias
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Configuración de estilo
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 150

def create_observation_structure_diagram():
    """Diagrama de la estructura del vector de observación (65D)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Título
    ax.text(5, 9.5, 'Estructura del Vector de Observación (65D)',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Componentes
    components = [
        {"name": "Estado del Cuerpo", "dims": 13, "color": "#FF6B6B",
         "items": ["Posición (x,y,z): 3D", "Orientación (w,x,y,z): 4D",
                   "Vel. Lineal (vx,vy,vz): 3D", "Vel. Angular (ωx,ωy,ωz): 3D"]},
        {"name": "Estado Articular", "dims": 24, "color": "#4ECDC4",
         "items": ["12 articulaciones:", "  - Posiciones: 12D", "  - Velocidades: 12D"]},
        {"name": "Estado de Patas", "dims": 24, "color": "#45B7D1",
         "items": ["4 patas × 3 coord:", "  - Posiciones pies: 12D", "  - Velocidades pies: 12D"]},
        {"name": "Contactos", "dims": 4, "color": "#96CEB4",
         "items": ["Flags binarios:", "  FL, FR, RL, RR: 4D"]},
    ]

    y_start = 8.0
    for comp in components:
        # Rectángulo principal
        rect = FancyBboxPatch((0.5, y_start-1.5), 4, 1.3,
                               boxstyle="round,pad=0.05",
                               edgecolor=comp["color"],
                               facecolor=comp["color"],
                               alpha=0.3, linewidth=2)
        ax.add_patch(rect)

        # Título del componente
        ax.text(2.5, y_start-0.85, f"{comp['name']} ({comp['dims']}D)",
                ha='center', va='center', fontsize=10, fontweight='bold')

        # Detalles
        detail_text = "\n".join(comp["items"])
        ax.text(5.5, y_start-0.85, detail_text,
                ha='left', va='center', fontsize=8, family='monospace')

        y_start -= 1.8

    # Indicador de total
    total_box = FancyBboxPatch((0.5, 0.3), 9, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black',
                                facecolor='#FFE66D',
                                alpha=0.5, linewidth=2)
    ax.add_patch(total_box)
    ax.text(5, 0.6, 'TOTAL: 13 + 24 + 24 + 4 = 65 dimensiones',
            ha='center', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig

def create_network_architecture_diagram():
    """Diagrama de la arquitectura de redes neuronales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    def draw_network(ax, title, layers, output_labels):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.text(5, 11.5, title, ha='center', va='top',
                fontsize=12, fontweight='bold')

        y_positions = [10, 8, 6, 4, 2]
        colors = ['#E8F4F8', '#B8E6F0', '#88D8E8', '#58CAE0', '#FFE66D']

        for i, (layer_name, neurons, activation) in enumerate(layers):
            # Caja de capa
            width = 6
            height = 0.8
            x_center = 5

            rect = FancyBboxPatch((x_center - width/2, y_positions[i] - height/2),
                                   width, height,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='black',
                                   facecolor=colors[i],
                                   linewidth=1.5)
            ax.add_patch(rect)

            # Texto
            if activation:
                label = f"{layer_name}\n{neurons} neuronas [{activation}]"
            else:
                label = f"{layer_name}\n{neurons}"

            ax.text(x_center, y_positions[i], label,
                    ha='center', va='center', fontsize=9, fontweight='bold')

            # Flechas entre capas
            if i < len(layers) - 1:
                arrow = FancyArrowPatch((x_center, y_positions[i] - height/2 - 0.1),
                                        (x_center, y_positions[i+1] + height/2 + 0.1),
                                        arrowstyle='->', lw=2, color='gray',
                                        mutation_scale=20)
                ax.add_patch(arrow)

        # Output
        if output_labels:
            y_out = 0.8
            for i, label in enumerate(output_labels):
                x_pos = 2 + i * 3
                ax.text(x_pos, y_out, label, ha='center', va='center',
                        fontsize=8, bbox=dict(boxstyle='round',
                        facecolor='lightgreen', alpha=0.5))

    # Red Actor
    actor_layers = [
        ("INPUT", "Observación (65D)", None),
        ("Capa Densa 1", 512, "ELU"),
        ("Capa Densa 2", 256, "ELU"),
        ("Capa Densa 3", 128, "ELU"),
        ("OUTPUT", "μ (12D) + log σ (12D)", "Lineal"),
    ]
    draw_network(ax1, "RED ACTOR (Política π)", actor_layers,
                 ["Media μ", "Std σ"])

    # Red Crítico
    critic_layers = [
        ("INPUT", "Observación (65D)", None),
        ("Capa Densa 1", 512, "ELU"),
        ("Capa Densa 2", 256, "ELU"),
        ("Capa Densa 3", 128, "ELU"),
        ("OUTPUT", "V(s) - Valor (1D)", "Lineal"),
    ]
    draw_network(ax2, "RED CRÍTICO (Función de Valor V)", critic_layers,
                 ["Valor V(s)"])

    plt.tight_layout()
    return fig

def create_action_structure_diagram():
    """Diagrama de la estructura de acción residual."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Título
    ax.text(5, 9.5, 'Estructura de Acción Residual (12D)',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Acción normalizada
    rect1 = FancyBboxPatch((0.5, 7.5), 9, 1.2,
                            boxstyle="round,pad=0.05",
                            edgecolor='#FF6B6B',
                            facecolor='#FF6B6B',
                            alpha=0.2, linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 8.3, 'Acción de Red Neuronal: a ∈ [-1, 1]¹²',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 7.8, 'Vector normalizado de 12 dimensiones',
            ha='center', va='center', fontsize=9, style='italic')

    # Flecha
    arrow1 = FancyArrowPatch((5, 7.4), (5, 6.8),
                             arrowstyle='->', lw=2.5, color='black',
                             mutation_scale=25)
    ax.add_patch(arrow1)
    ax.text(5.5, 7.1, 'Escalado\n× 0.01 m', ha='left', va='center', fontsize=9)

    # Residuales por pata
    legs = ["FL (Front-Left)", "FR (Front-Right)", "RL (Rear-Left)", "RR (Rear-Right)"]
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    y_start = 6.2
    for i, (leg, color) in enumerate(zip(legs, colors)):
        rect = FancyBboxPatch((0.5, y_start - i*1.2), 4.5, 0.9,
                               boxstyle="round,pad=0.05",
                               edgecolor=color,
                               facecolor=color,
                               alpha=0.3, linewidth=1.5)
        ax.add_patch(rect)

        ax.text(2.75, y_start - i*1.2 + 0.45, leg,
                ha='center', va='center', fontsize=10, fontweight='bold')

        # Componentes xyz
        components = ['Δx (longitudinal)', 'Δy (lateral)', 'Δz (vertical)']
        x_positions = [5.5, 6.8, 8.1]
        for comp, x_pos in zip(components, x_positions):
            small_box = FancyBboxPatch((x_pos-0.5, y_start - i*1.2 + 0.1), 0.9, 0.6,
                                        boxstyle="round,pad=0.02",
                                        edgecolor='gray',
                                        facecolor='white',
                                        linewidth=1)
            ax.add_patch(small_box)
            ax.text(x_pos, y_start - i*1.2 + 0.4, comp,
                    ha='center', va='center', fontsize=7)

    # Integración con controlador base
    y_integration = 1.3
    rect_integration = FancyBboxPatch((0.5, y_integration - 0.5), 9, 1,
                                       boxstyle="round,pad=0.05",
                                       edgecolor='green',
                                       facecolor='lightgreen',
                                       alpha=0.3, linewidth=2)
    ax.add_patch(rect_integration)
    ax.text(5, y_integration + 0.2, 'Integración con Controlador Base',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_integration - 0.2,
            'target_final[pata] = target_base[pata] + residual[pata]',
            ha='center', va='center', fontsize=9, family='monospace')

    plt.tight_layout()
    return fig

def create_training_flow_diagram():
    """Diagrama de flujo de entrenamiento."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Título
    ax.text(5, 13.5, 'Flujo de Datos en Ciclo de Entrenamiento RL',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Componentes del flujo
    stages = [
        {"name": "1. ENTORNO\nMuJoCo", "y": 12, "color": "#FF6B6B",
         "desc": "Estado: s_t (posición, velocidad, contactos)"},
        {"name": "2. OBSERVACIÓN\n65D", "y": 10.5, "color": "#4ECDC4",
         "desc": "Vector normalizado del estado"},
        {"name": "3. RED NEURONAL\nActor", "y": 9, "color": "#45B7D1",
         "desc": "π(a|s) → N(μ, σ²)"},
        {"name": "4. ACCIÓN\n12D Residual", "y": 7.5, "color": "#96CEB4",
         "desc": "Correcciones: a_t ~ N(μ, σ²)"},
        {"name": "5. CONTROLADOR\nBézier + Residual", "y": 6, "color": "#FFEAA7",
         "desc": "target_final = base + residual"},
        {"name": "6. CINEMÁTICA\nInversa IK", "y": 4.5, "color": "#DFE6E9",
         "desc": "Objetivos → ángulos articulares"},
        {"name": "7. SIMULACIÓN\nmj_step", "y": 3, "color": "#FFD93D",
         "desc": "Física: s_t → s_{t+1}, recompensa r_t"},
    ]

    # Dibujar etapas
    for i, stage in enumerate(stages):
        # Caja
        rect = FancyBboxPatch((0.5, stage["y"] - 0.5), 3.5, 0.9,
                               boxstyle="round,pad=0.05",
                               edgecolor=stage["color"],
                               facecolor=stage["color"],
                               alpha=0.4, linewidth=2)
        ax.add_patch(rect)

        # Nombre
        ax.text(2.25, stage["y"], stage["name"],
                ha='center', va='center', fontsize=9, fontweight='bold')

        # Descripción
        ax.text(5.0, stage["y"], stage["desc"],
                ha='left', va='center', fontsize=8)

        # Flecha a siguiente etapa
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((2.25, stage["y"] - 0.6),
                                    (2.25, stages[i+1]["y"] + 0.55),
                                    arrowstyle='->', lw=2, color='black',
                                    mutation_scale=20)
            ax.add_patch(arrow)

    # Ciclo de vuelta (buffer y optimización)
    y_buffer = 1.5

    # Buffer
    rect_buffer = FancyBboxPatch((0.5, y_buffer - 0.4), 4.2, 0.7,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='purple',
                                  facecolor='#DDA0DD',
                                  alpha=0.4, linewidth=2)
    ax.add_patch(rect_buffer)
    ax.text(2.6, y_buffer, '8. ROLLOUT BUFFER\n(s_t, a_t, r_t, V(s_t), log π)',
            ha='center', va='center', fontsize=8, fontweight='bold')

    # Optimización
    rect_optim = FancyBboxPatch((5.3, y_buffer - 0.4), 4.2, 0.7,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='darkgreen',
                                 facecolor='#90EE90',
                                 alpha=0.4, linewidth=2)
    ax.add_patch(rect_optim)
    ax.text(7.4, y_buffer, '9. OPTIMIZACIÓN PPO\nActualizar θ (pesos de redes)',
            ha='center', va='center', fontsize=8, fontweight='bold')

    # Flechas del ciclo
    # Simulación → Buffer
    arrow_to_buffer = FancyArrowPatch((2.25, 2.4), (2.6, y_buffer + 0.35),
                                       arrowstyle='->', lw=2, color='purple',
                                       mutation_scale=20)
    ax.add_patch(arrow_to_buffer)

    # Buffer → Optimización
    arrow_buffer_optim = FancyArrowPatch((4.7, y_buffer), (5.3, y_buffer),
                                          arrowstyle='->', lw=2, color='darkgreen',
                                          mutation_scale=20)
    ax.add_patch(arrow_buffer_optim)

    # Optimización → Red Neuronal (ciclo)
    # Línea hacia arriba
    ax.plot([8.5, 8.5], [y_buffer + 0.4, 9], 'g--', lw=2)
    # Línea horizontal
    ax.plot([8.5, 4.2], [9, 9], 'g--', lw=2)
    # Flecha final
    arrow_update = FancyArrowPatch((4.2, 9), (4.0, 9),
                                    arrowstyle='->', lw=2, color='darkgreen',
                                    mutation_scale=20)
    ax.add_patch(arrow_update)
    ax.text(8.8, 6, 'Política\nMejorada', ha='left', va='center',
            fontsize=8, color='darkgreen', fontweight='bold')

    # Info de paralelización
    info_box = FancyBboxPatch((0.5, 0.1), 9, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor='orange',
                               facecolor='#FFE5B4',
                               alpha=0.5, linewidth=2)
    ax.add_patch(info_box)
    ax.text(5, 0.4,
            'Paralelización: 80 entornos × 4096 pasos = 327,680 transiciones por iteración',
            ha='center', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig

def create_buffer_structure_diagram():
    """Diagrama de la estructura del buffer de experiencias."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Título
    ax.text(5, 9.5, 'Estructura del Rollout Buffer',
            ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(5, 9.0, 'Dimensiones: (n_steps=4096, n_envs=80, dim_dato)',
            ha='center', va='top', fontsize=10, style='italic')

    # Componentes del buffer
    buffer_components = [
        ("observations", "(4096, 80, 65)", "#FF6B6B", "Observaciones recolectadas"),
        ("actions", "(4096, 80, 12)", "#4ECDC4", "Acciones ejecutadas"),
        ("rewards", "(4096, 80, 1)", "#45B7D1", "Recompensas recibidas"),
        ("values", "(4096, 80, 1)", "#96CEB4", "Estimaciones V(s) del crítico"),
        ("log_probs", "(4096, 80, 1)", "#FFEAA7", "Log probabilidades π(a|s)"),
        ("dones", "(4096, 80, 1)", "#DFE6E9", "Flags de terminación"),
        ("advantages", "(4096, 80, 1)", "#DDA0DD", "Ventajas A(s,a) [GAE]"),
        ("returns", "(4096, 80, 1)", "#90EE90", "Retornos descontados"),
    ]

    y_start = 8.2
    for name, shape, color, desc in buffer_components:
        # Rectángulo
        rect = FancyBboxPatch((0.5, y_start - 0.35), 2.5, 0.6,
                               boxstyle="round,pad=0.03",
                               edgecolor=color,
                               facecolor=color,
                               alpha=0.4, linewidth=1.5)
        ax.add_patch(rect)

        # Nombre y forma
        ax.text(1.75, y_start, f"{name}\n{shape}",
                ha='center', va='center', fontsize=8, fontweight='bold')

        # Descripción
        ax.text(3.3, y_start, desc,
                ha='left', va='center', fontsize=8)

        y_start -= 0.85

    # Visualización de matriz 3D
    ax.text(5, 1.8, 'Visualización Conceptual:',
            ha='center', va='top', fontsize=10, fontweight='bold')

    # Dibujar cubo representando (steps, envs, dim)
    # Cara frontal
    front_face = mpatches.Rectangle((3.5, 0.3), 3, 1.2,
                                     linewidth=2, edgecolor='black',
                                     facecolor='lightblue', alpha=0.5)
    ax.add_patch(front_face)

    # Cara lateral (perspectiva)
    side_points = np.array([[6.5, 0.3], [7.2, 0.7], [7.2, 1.9], [6.5, 1.5], [6.5, 0.3]])
    side_face = mpatches.Polygon(side_points, linewidth=2,
                                  edgecolor='black', facecolor='lightcoral', alpha=0.5)
    ax.add_patch(side_face)

    # Cara superior
    top_points = np.array([[3.5, 1.5], [6.5, 1.5], [7.2, 1.9], [4.2, 1.9], [3.5, 1.5]])
    top_face = mpatches.Polygon(top_points, linewidth=2,
                                 edgecolor='black', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(top_face)

    # Etiquetas de dimensiones
    ax.text(5, 0.1, 'n_envs (80)', ha='center', va='center', fontsize=8)
    ax.text(7.5, 1.0, 'dim', ha='left', va='center', fontsize=8)
    ax.text(3.2, 0.9, 'n_steps\n(4096)', ha='right', va='center', fontsize=8)

    # Info de memoria
    memory_box = FancyBboxPatch((0.5, 0.05), 9, 0.5,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='red',
                                 facecolor='#FFE5E5',
                                 alpha=0.5, linewidth=2)
    ax.add_patch(memory_box)
    ax.text(5, 0.3,
            'Uso de memoria: ~26M valores float32 ≈ 104 MB por buffer',
            ha='center', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig

# Generar todos los diagramas
if __name__ == "__main__":
    print("Generando diagramas para tesis...")

    print("1. Estructura de observación...")
    fig1 = create_observation_structure_diagram()
    fig1.savefig('diagrama_observacion.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    print("2. Arquitectura de redes neuronales...")
    fig2 = create_network_architecture_diagram()
    fig2.savefig('diagrama_redes_neuronales.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print("3. Estructura de acción...")
    fig3 = create_action_structure_diagram()
    fig3.savefig('diagrama_accion_residual.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

    print("4. Flujo de entrenamiento...")
    fig4 = create_training_flow_diagram()
    fig4.savefig('diagrama_flujo_entrenamiento.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)

    print("5. Estructura de buffer...")
    fig5 = create_buffer_structure_diagram()
    fig5.savefig('diagrama_buffer_experiencias.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)

    print("\n✓ Todos los diagramas generados exitosamente!")
    print("\nArchivos creados:")
    print("  - diagrama_observacion.png")
    print("  - diagrama_redes_neuronales.png")
    print("  - diagrama_accion_residual.png")
    print("  - diagrama_flujo_entrenamiento.png")
    print("  - diagrama_buffer_experiencias.png")
    print("\nPuedes incluir estos diagramas en tu documento de tesis.")
