import json
import os
import sys
from pathlib import Path
import subprocess

def install_kernel():
    """Instala el kernel personalizado para power_graph"""
    
    try:
        # Obtener la ruta del proyecto (donde está pyproject.toml)
        current_file = Path(__file__)
        project_path = current_file.parent.parent.parent.parent
        src_path = project_path / "src"
        
        print(f"📦 Instalando kernel para power_graph...")
        print(f"📁 Ruta del proyecto: {project_path}")
        print(f"📁 Ruta src: {src_path}")
        
        # Configuración del kernel
        kernel_config = {
            "argv": [
                sys.executable,  # Usar el mismo Python
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}"
            ],
            "display_name": "Power Graph Kernel",
            "language": "python",
            "env": {
                "PYTHONPATH": str(src_path)
            },
            "metadata": {
                "debugger": True
            }
        }
        
        # Directorio del kernel (depende del SO)
        if os.name == 'nt':  # Windows
            kernel_dir = Path.home() / "AppData" / "Roaming" / "jupyter" / "kernels" / "power_graph_kernel"
        else:  # Linux/Mac
            kernel_dir = Path.home() / ".local" / "share" / "jupyter" / "kernels" / "power_graph_kernel"
        
        # Crear directorio
        kernel_dir.mkdir(parents=True, exist_ok=True)
        
        # Escribir configuración
        kernel_json_path = kernel_dir / "kernel.json"
        with open(kernel_json_path, "w", encoding="utf-8") as f:
            json.dump(kernel_config, f, indent=2)
        
        print(f"✅ Kernel instalado en: {kernel_dir}")
        print(f"📄 Configuración: {kernel_json_path}")
        print("\n🎯 Para usar el kernel:")
        print("   1. Abre Jupyter: jupyter notebook")
        print("   2. Ve a Kernel → Change kernel")
        print("   3. Selecciona 'Power Graph Kernel'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error instalando kernel: {e}")
        return False

if __name__ == "__main__":
    install_kernel()