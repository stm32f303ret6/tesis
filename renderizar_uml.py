#!/usr/bin/env python3
"""
Script para renderizar diagramas PlantUML a imágenes PNG de alta calidad.

Opciones de renderizado:
1. Usar servidor web de PlantUML (requiere internet)
2. Usar instalación local de PlantUML (requiere Java + plantuml.jar)
3. Usar Docker con plantuml/plantuml

Este script intenta todos los métodos en orden hasta que uno funcione.
"""

import os
import subprocess
import urllib.request
import urllib.parse
import zlib
import base64
from pathlib import Path

def encode_plantuml_url(plantuml_text):
    """
    Codifica texto PlantUML para la API web.

    Basado en: https://plantuml.com/text-encoding
    """
    # Comprimir con zlib
    compressed = zlib.compress(plantuml_text.encode('utf-8'))

    # Codificar en base64
    b64 = base64.b64encode(compressed).decode('ascii')

    # Reemplazar caracteres para URL
    encoded = b64.replace('+', '-').replace('/', '_')

    return encoded

def render_with_web_api(puml_file, output_file):
    """
    Renderiza usando el servidor web público de PlantUML.

    Args:
        puml_file: Ruta al archivo .puml
        output_file: Ruta de salida .png

    Returns:
        True si tuvo éxito, False en caso contrario
    """
    try:
        print(f"  → Intentando con API web de PlantUML...")

        # Leer archivo
        with open(puml_file, 'r', encoding='utf-8') as f:
            plantuml_text = f.read()

        # Codificar
        encoded = encode_plantuml_url(plantuml_text)

        # Construir URL (servidor público)
        url = f"http://www.plantuml.com/plantuml/png/{encoded}"

        print(f"    URL: {url[:80]}...")

        # Descargar imagen
        urllib.request.urlretrieve(url, output_file)

        print(f"    ✓ Éxito con API web")
        return True

    except Exception as e:
        print(f"    ✗ Falló API web: {e}")
        return False

def render_with_local_jar(puml_file, output_file):
    """
    Renderiza usando instalación local de PlantUML (Java).

    Requiere:
    - Java instalado
    - plantuml.jar en el directorio actual o en PATH

    Args:
        puml_file: Ruta al archivo .puml
        output_file: Ruta de salida .png

    Returns:
        True si tuvo éxito, False en caso contrario
    """
    try:
        print(f"  → Intentando con PlantUML local (Java)...")

        # Buscar plantuml.jar
        jar_locations = [
            "plantuml.jar",
            "/usr/share/plantuml/plantuml.jar",
            "/usr/local/bin/plantuml.jar",
            os.path.expanduser("~/plantuml.jar"),
        ]

        jar_path = None
        for location in jar_locations:
            if os.path.exists(location):
                jar_path = location
                break

        if not jar_path:
            print(f"    ✗ No se encontró plantuml.jar")
            return False

        # Ejecutar PlantUML
        cmd = ["java", "-jar", jar_path, "-tpng", str(puml_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # PlantUML genera PNG con mismo nombre que .puml
            generated_png = str(puml_file).replace('.puml', '.png')
            if os.path.exists(generated_png):
                # Mover a la ubicación deseada si es diferente
                if generated_png != str(output_file):
                    os.rename(generated_png, output_file)
                print(f"    ✓ Éxito con PlantUML local")
                return True

        print(f"    ✗ PlantUML local falló: {result.stderr}")
        return False

    except Exception as e:
        print(f"    ✗ Error con PlantUML local: {e}")
        return False

def render_with_docker(puml_file, output_file):
    """
    Renderiza usando contenedor Docker de PlantUML.

    Requiere:
    - Docker instalado y ejecutándose

    Args:
        puml_file: Ruta al archivo .puml
        output_file: Ruta de salida .png

    Returns:
        True si tuvo éxito, False en caso contrario
    """
    try:
        print(f"  → Intentando con Docker (plantuml/plantuml)...")

        # Verificar que Docker esté disponible
        result = subprocess.run(["docker", "--version"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ✗ Docker no está disponible")
            return False

        # Obtener rutas absolutas
        abs_puml = os.path.abspath(puml_file)
        abs_dir = os.path.dirname(abs_puml)
        filename = os.path.basename(puml_file)

        # Ejecutar contenedor
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_dir}:/data",
            "plantuml/plantuml",
            "-tpng", f"/data/{filename}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            generated_png = str(puml_file).replace('.puml', '.png')
            if os.path.exists(generated_png):
                if generated_png != str(output_file):
                    os.rename(generated_png, output_file)
                print(f"    ✓ Éxito con Docker")
                return True

        print(f"    ✗ Docker falló: {result.stderr}")
        return False

    except Exception as e:
        print(f"    ✗ Error con Docker: {e}")
        return False

def render_diagram(puml_file, output_file=None):
    """
    Renderiza un diagrama PlantUML intentando múltiples métodos.

    Args:
        puml_file: Ruta al archivo .puml
        output_file: Ruta de salida .png (opcional, por defecto mismo nombre)

    Returns:
        True si tuvo éxito, False en caso contrario
    """
    puml_path = Path(puml_file)

    if not puml_path.exists():
        print(f"✗ Archivo no encontrado: {puml_file}")
        return False

    if output_file is None:
        output_file = puml_path.with_suffix('.png')

    output_path = Path(output_file)

    print(f"\nRenderizando: {puml_path.name}")
    print(f"  Salida: {output_path}")

    # Intentar métodos en orden de preferencia
    methods = [
        render_with_web_api,
        render_with_local_jar,
        render_with_docker,
    ]

    for method in methods:
        if method(puml_path, output_path):
            return True

    print(f"✗ Todos los métodos fallaron para {puml_path.name}")
    return False

def main():
    """Renderiza todos los diagramas UML del proyecto."""

    print("=" * 80)
    print("Renderizador de Diagramas UML para Tesis")
    print("=" * 80)

    # Directorio de diagramas
    uml_dir = Path("uml_diagrams")

    if not uml_dir.exists():
        print(f"\n✗ Directorio no encontrado: {uml_dir}")
        print("  Asegúrate de estar en el directorio correcto del proyecto.")
        return 1

    # Buscar todos los archivos .puml
    puml_files = sorted(uml_dir.glob("*.puml"))

    if not puml_files:
        print(f"\n✗ No se encontraron archivos .puml en {uml_dir}")
        return 1

    print(f"\nEncontrados {len(puml_files)} diagramas:")
    for pf in puml_files:
        print(f"  - {pf.name}")

    print("\n" + "=" * 80)
    print("Renderizando diagramas...")
    print("=" * 80)

    # Renderizar cada diagrama
    success_count = 0
    for puml_file in puml_files:
        if render_diagram(puml_file):
            success_count += 1

    # Resumen
    print("\n" + "=" * 80)
    print("Resumen")
    print("=" * 80)
    print(f"Éxitos: {success_count}/{len(puml_files)}")

    if success_count == len(puml_files):
        print("\n✓ Todos los diagramas se renderizaron correctamente!")
        print("\nArchivos generados:")
        for puml_file in puml_files:
            png_file = puml_file.with_suffix('.png')
            if png_file.exists():
                size_kb = png_file.stat().st_size / 1024
                print(f"  - {png_file.name} ({size_kb:.1f} KB)")

        print("\nPuedes incluir estos diagramas en tu documento de tesis.")
        return 0
    else:
        print(f"\n⚠ Solo {success_count}/{len(puml_files)} diagramas se renderizaron.")
        print("\nPara instalar PlantUML localmente:")
        print("  1. Instalar Java: sudo apt install default-jre")
        print("  2. Descargar PlantUML:")
        print("     wget https://github.com/plantuml/plantuml/releases/download/v1.2023.13/plantuml-1.2023.13.jar")
        print("     mv plantuml-*.jar plantuml.jar")
        print("  3. Ejecutar nuevamente este script")

        print("\nAlternativamente, puedes usar Docker:")
        print("  docker pull plantuml/plantuml")

        return 1

if __name__ == "__main__":
    exit(main())
