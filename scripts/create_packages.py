"""Package initialization files creator."""

import os
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Directories that need __init__.py
PACKAGES = [
    'models',
    'models/vision',
    'models/audio',
    'models/fusion',
    'llm',
    'tts',
    'memory',
    'privacy',
    'tests'
]

def create_init_files():
    """Create __init__.py files for all packages."""
    for package in PACKAGES:
        init_path = PROJECT_ROOT / package / '__init__.py'
        init_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not init_path.exists():
            with open(init_path, 'w') as f:
                package_name = package.replace('/', '.').replace('\\', '.')
                f.write(f'"""Package: {package_name}"""\n')
            print(f"Created: {init_path}")
        else:
            print(f"Exists: {init_path}")

if __name__ == '__main__':
    create_init_files()
    print("\n✅ All package __init__.py files created!")
