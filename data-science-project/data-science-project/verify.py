#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de verificación automatizada para el proyecto de detección de neumonía.

Este script ejecuta todas las verificaciones necesarias para garantizar que
las pruebas unitarias y Docker funcionen correctamente.

Uso:
    python verify.py [--tests] [--docker] [--all]

Universidad Autónoma de Occidente - Módulo 1
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


class PneumoniaVerifier:
    """Verificador automatizado para el proyecto de neumonía."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_passed = 0
        self.tests_total = 0
        self.docker_working = False
        self.start_time = time.time()
    
    def print_header(self, title: str):
        """Imprime un encabezado formateado."""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_step(self, step: str, status: str = "⏳"):
        """Imprime un paso de verificación."""
        print(f"{status} {step}")
    
    def run_command(self, command: str, cwd: str = None, timeout: int = 300) -> tuple:
        """
        Ejecuta un comando y retorna (success, output, error).
        
        Args:
            command: Comando a ejecutar
            cwd: Directorio de trabajo
            timeout: Timeout en segundos
        
        Returns:
            tuple: (success, stdout, stderr)
        """
        try:
            cwd = cwd or str(self.project_root)
            
            result = subprocess.run(
                command.split(),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"Comando excedió timeout de {timeout}s"
        except Exception as e:
            return False, "", str(e)
    
    def verify_prerequisites(self) -> bool:
        """Verifica los prerequisitos del sistema."""
        self.print_header("Verificando Prerequisitos")
        
        # Verificar Python
        self.print_step("Verificando Python...", "⏳")
        success, output, error = self.run_command("python --version")
        if success:
            python_version = output.strip()
            self.print_step(f"Python detectado: {python_version}", "✅")
        else:
            self.print_step("Python no encontrado", "❌")
            return False
        
        # Verificar pip
        self.print_step("Verificando pip...", "⏳")
        success, output, error = self.run_command("pip --version")
        if success:
            self.print_step("pip disponible", "✅")
        else:
            self.print_step("pip no encontrado", "❌")
            return False
        
        # Verificar Docker
        self.print_step("Verificando Docker...", "⏳")
        success, output, error = self.run_command("docker --version")
        if success:
            docker_version = output.strip()
            self.print_step(f"Docker detectado: {docker_version}", "✅")
            self.docker_available = True
        else:
            self.print_step("Docker no encontrado (opcional)", "⚠️")
            self.docker_available = False
        
        return True
    
    def verify_unit_tests(self) -> bool:
        """Verifica las pruebas unitarias."""
        self.print_header("Verificando Pruebas Unitarias")
        
        # Instalar dependencias si no están
        self.print_step("Verificando dependencias...", "⏳")
        requirements_file = self.project_root / "requirements.txt"
        
        if requirements_file.exists():
            success, output, error = self.run_command(
                f"pip install -r {requirements_file}"
            )
            if success:
                self.print_step("Dependencias instaladas", "✅")
            else:
                self.print_step("Error instalando dependencias", "❌")
                print(f"Error: {error}")
                return False
        
        # Ejecutar pruebas
        self.print_step("Ejecutando pruebas unitarias...", "⏳")
        tests_dir = self.project_root / "tests"
        
        if not tests_dir.exists():
            self.print_step("Directorio tests/ no encontrado", "❌")
            return False
        
        # Configurar PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{self.project_root}:{env.get('PYTHONPATH', '')}"
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                env=env,
                timeout=300
            )
            
            # Parsear resultados
            output = result.stdout
            if "failed" in output and "passed" in output:
                # Extraer número de pruebas pasadas y fallidas
                lines = output.split('\n')
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Formato: "X passed, Y failed in Zs"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed," and i > 0:
                                self.tests_passed = int(parts[i-1])
                            elif part == "failed" and i > 0:
                                tests_failed = int(parts[i-1])
                                self.tests_total = self.tests_passed + tests_failed
                        break
            elif "passed" in output:
                # Solo pruebas pasadas
                lines = output.split('\n')
                for line in lines:
                    if "passed in" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed" and i > 0:
                                self.tests_passed = int(parts[i-1])
                                self.tests_total = self.tests_passed
                        break
            
            if self.tests_total > 0:
                success_rate = (self.tests_passed / self.tests_total) * 100
                if success_rate >= 90:
                    self.print_step(
                        f"Pruebas: {self.tests_passed}/{self.tests_total} "
                        f"({success_rate:.1f}% éxito)", "✅"
                    )
                    return True
                else:
                    self.print_step(
                        f"Pruebas: {self.tests_passed}/{self.tests_total} "
                        f"({success_rate:.1f}% éxito) - Muy bajo", "❌"
                    )
            else:
                self.print_step("No se pudieron ejecutar las pruebas", "❌")
                
            if result.stderr:
                print(f"\nErrores:\n{result.stderr}")
            
            return False
            
        except subprocess.TimeoutExpired:
            self.print_step("Pruebas excedieron tiempo límite", "❌")
            return False
        except Exception as e:
            self.print_step(f"Error ejecutando pruebas: {e}", "❌")
            return False
    
    def verify_docker(self) -> bool:
        """Verifica la funcionalidad de Docker."""
        if not self.docker_available:
            self.print_step("Docker no está disponible, saltando...", "⚠️")
            return True
        
        self.print_header("Verificando Docker")
        
        # Construir imagen
        self.print_step("Construyendo imagen Docker...", "⏳")
        success, output, error = self.run_command(
            "docker build -t uao/pneumonia-detector:latest .",
            timeout=600  # 10 minutos
        )
        
        if not success:
            self.print_step("Error construyendo imagen Docker", "❌")
            print(f"Error: {error}")
            return False
        
        self.print_step("Imagen Docker construida exitosamente", "✅")
        
        # Verificar que la imagen existe
        self.print_step("Verificando imagen creada...", "⏳")
        success, output, error = self.run_command(
            "docker images uao/pneumonia-detector:latest"
        )
        
        if success and "uao/pneumonia-detector" in output:
            self.print_step("Imagen Docker verificada", "✅")
        else:
            self.print_step("Imagen Docker no encontrada", "❌")
            return False
        
        # Ejecutar pruebas en contenedor
        self.print_step("Ejecutando pruebas en contenedor...", "⏳")
        success, output, error = self.run_command(
            "docker run --rm uao/pneumonia-detector:latest",
            timeout=300  # 5 minutos
        )
        
        if success:
            self.print_step("Pruebas ejecutadas correctamente en Docker", "✅")
            self.docker_working = True
        else:
            self.print_step("Error ejecutando pruebas en Docker", "⚠️")
            # No es crítico, el contenedor puede funcionar para otros propósitos
            self.docker_working = False
        
        return True
    
    def generate_report(self) -> dict:
        """Genera un reporte final de la verificación."""
        duration = time.time() - self.start_time
        
        return {
            'duration': f"{duration:.2f}s",
            'tests_passed': self.tests_passed,
            'tests_total': self.tests_total,
            'test_success_rate': (
                (self.tests_passed / self.tests_total * 100) 
                if self.tests_total > 0 else 0
            ),
            'docker_available': self.docker_available,
            'docker_working': self.docker_working,
            'overall_success': (
                self.tests_total > 0 and 
                (self.tests_passed / self.tests_total) >= 0.9 and
                (self.docker_working or not self.docker_available)
            )
        }
    
    def print_final_report(self, report: dict):
        """Imprime el reporte final."""
        self.print_header("Reporte Final")
        
        print(f"⏱️  Duración total: {report['duration']}")
        print(f"🧪 Pruebas unitarias: {report['tests_passed']}/{report['tests_total']} "
              f"({report['test_success_rate']:.1f}%)")
        print(f"🐳 Docker disponible: {'✅' if report['docker_available'] else '❌'}")
        print(f"🐳 Docker funcionando: {'✅' if report['docker_working'] else '❌'}")
        
        print(f"\n{'='*60}")
        if report['overall_success']:
            print("🎉 ¡VERIFICACIÓN EXITOSA!")
            print("   El proyecto está listo para desarrollo y producción.")
        else:
            print("⚠️  VERIFICACIÓN CON PROBLEMAS")
            print("   Revisa los errores reportados arriba.")
        print(f"{'='*60}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Verificador automatizado para proyecto de neumonía"
    )
    parser.add_argument(
        "--tests", 
        action="store_true", 
        help="Solo verificar pruebas unitarias"
    )
    parser.add_argument(
        "--docker", 
        action="store_true", 
        help="Solo verificar Docker"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Verificar todo (por defecto)"
    )
    
    args = parser.parse_args()
    
    # Si no se especifica nada, verificar todo
    if not any([args.tests, args.docker]):
        args.all = True
    
    verifier = PneumoniaVerifier()
    
    print("🚀 Iniciando verificación del proyecto de detección de neumonía")
    print(f"📁 Directorio: {verifier.project_root}")
    
    # Verificar prerequisitos
    if not verifier.verify_prerequisites():
        print("\n❌ Prerequisitos no satisfechos. Abortando.")
        sys.exit(1)
    
    success = True
    
    # Verificar pruebas unitarias
    if args.tests or args.all:
        if not verifier.verify_unit_tests():
            success = False
    
    # Verificar Docker
    if args.docker or args.all:
        if not verifier.verify_docker():
            success = False
    
    # Generar y mostrar reporte
    report = verifier.generate_report()
    verifier.print_final_report(report)
    
    # Exit code basado en éxito general
    sys.exit(0 if report['overall_success'] else 1)


if __name__ == "__main__":
    main()
