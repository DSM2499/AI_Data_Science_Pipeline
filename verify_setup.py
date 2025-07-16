#!/usr/bin/env python3
"""Verification script to check if the Data Science Agent system is properly set up."""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False

def check_file_structure():
    """Check if all required files and directories exist."""
    required_paths = [
        "agents/",
        "memory/",
        "ui/",
        "utils/",
        "tests/",
        "project_output/",
        "config.py",
        "main.py",
        "requirements.txt",
        "pyproject.toml",
        "Makefile",
        ".env.example",
    ]
    
    print("\n📁 Checking file structure:")
    all_exist = True
    
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"  ✅ {path_str}")
        else:
            print(f"  ❌ {path_str}")
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if required dependencies can be imported."""
    dependencies = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly.express"),
        ("pydantic", "pydantic"),
    ]
    
    print("\n📦 Checking core dependencies:")
    available = []
    missing = []
    
    for name, import_path in dependencies:
        try:
            __import__(import_path)
            print(f"  ✅ {name}")
            available.append(name)
        except ImportError:
            print(f"  ❌ {name} (run 'pip install {name}')")
            missing.append(name)
    
    return len(missing) == 0, missing

def check_optional_dependencies():
    """Check optional dependencies."""
    optional_deps = [
        ("loguru", "loguru"),
        ("chromadb", "chromadb"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
    ]
    
    print("\n🔧 Checking optional dependencies:")
    for name, import_path in optional_deps:
        try:
            __import__(import_path)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} (install with 'pip install {name}')")

def check_directories():
    """Check if output directories exist."""
    output_dirs = [
        "project_output/data",
        "project_output/models", 
        "project_output/reports",
        "project_output/scripts",
        "project_output/logs",
        "memory/vector",
        "memory/symbolic",
    ]
    
    print("\n📂 Checking output directories:")
    for dir_path in output_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ⚠️  {dir_path} (will be created automatically)")

def main():
    """Run all verification checks."""
    print("🔍 Data Science Agent Setup Verification")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check file structure
    structure_ok = check_file_structure()
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Check directories
    check_directories()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if python_ok and structure_ok and deps_ok:
        print("✅ System ready! Core components are functional.")
        print("\n🚀 Next steps:")
        print("   1. Install missing optional dependencies: pip install -r requirements.txt")
        print("   2. Copy .env.example to .env and configure API keys")
        print("   3. Run the application: streamlit run main.py")
    else:
        print("❌ Setup incomplete. Please address the issues above.")
        if not python_ok:
            print("   - Upgrade Python to 3.9 or higher")
        if not structure_ok:
            print("   - Verify all project files are present")
        if not deps_ok:
            print(f"   - Install missing dependencies: pip install {' '.join(missing_deps)}")
    
    print("\n📚 Documentation:")
    print("   - README.md: Complete setup and usage guide")
    print("   - CLAUDE.md: Development guide")
    print("   - Agent_spec.md & Project_spec.md: Technical specifications")

if __name__ == "__main__":
    main()