"""
This file helps to update the __init__.py file in the distributions package. It
generates all import statements. However, aliases are not generated and have to
be copied from the old file.
"""
import os
import pkgutil
import subprocess
import importlib
import inspect
import shlex

# The name of the package
package_name = 'pyrecest.distributions'
output_file = 'new_init.py'  # The file to write the new import statements to

# Get a list of all tracked Python files
tracked_files = subprocess.run(shlex.split('git ls-files'), capture_output=True, text=True).stdout.splitlines()
tracked_files = [file[:-3].replace('/', '.') for file in tracked_files if file.endswith('.py')]

# Function to recursively walk through packages and generate import statements
def walk_packages(path, prefix):
    for _, module_name, is_pkg in pkgutil.iter_modules(path, prefix):
        if is_pkg:
            walk_packages([os.path.join(path[0], module_name.split('.')[-1])], module_name + ".")
        elif module_name in tracked_files:
            # Import the module
            module = importlib.import_module(module_name)
            # Get all class names in the module
            class_names = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module_name]
            # Generate import statements for each class
            for curr_class_name in class_names:
                import_statements.append(f"from {module_name} import {curr_class_name}")
                all_class_names.append(curr_class_name)

# Generate a list of all import statements and class names
import_statements: list[str] = []
all_class_names: list[str] = []
walk_packages([package_name.replace('.', '/')], package_name + '.')

# Write the import statements and the __all__ variable to the output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(import_statements))
    f.write('\n\n__all__ = [\n')
    for class_name in all_class_names:
        f.write(f'    "{class_name}",\n')
    f.write(']\n')
