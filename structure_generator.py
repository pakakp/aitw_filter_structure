#!/usr/bin/python3
"""
Wrapper script for wood microstructure generation.
Handles both Birch and Spruce microstructure.

Usage: generate_structure.py params.json birch|spruce
Output:
  - SaveBirch_0/ or SaveSpruce_0/  (structure directory)
  - output_dir.txt                  (name of created directory)
"""
import sys
import shutil
import tarfile
import os
import warnings
warnings.filterwarnings('ignore')

from wood_microstructure import (
    BirchMicrostructure, BirchParams,
    SpruceMicrostructure, SpruceParams,
)

if len(sys.argv) < 3:
    print("Usage: generate_structure.py params.json birch|spruce")
    sys.exit(1)

params_file = sys.argv[1]
wood_type   = sys.argv[2].lower()

print(f"Loading parameters from: {params_file}")
print(f"Wood type: {wood_type}")

# Record existing directories before generation
before = set(os.listdir('.'))

# Generate structure based on wood type
# Uses from_json() matching the original scripts exactly
print(f"\nGenerating {wood_type} structure...")

if wood_type == 'birch':
    params = BirchParams.from_json(params_file)
    for param in params:
        structure = BirchMicrostructure(param)
        structure.generate()
elif wood_type == 'spruce':
    params = SpruceParams.from_json(params_file)
    for param in params:
        structure = SpruceMicrostructure(param)
        structure.generate()
else:
    print(f"ERROR: Unknown wood type '{wood_type}'. Use 'birch' or 'spruce'.")
    sys.exit(1)

# Find the newly created directory by comparing before/after
after    = set(os.listdir('.'))
new_dirs = [d for d in (after - before) if d.startswith('Save')]

if not new_dirs:
    print("ERROR: No Save* directory was created!")
    sys.exit(1)

output_dir = new_dirs[0]
print(f"\nGenerated structure in: {output_dir}")
print(f"Contents:")
for f in sorted(os.listdir(output_dir)):
    print(f"  - {f}")

# ── CLEAN UP: Keep only FinalVolumeSlice subdirectory ──────────────
print(f"\nCleaning up {output_dir}...")
print("Keeping only: FinalVolumeSlice/")

# Check what subdirectories exist
subdirs = [d for d in os.listdir(output_dir) 
           if os.path.isdir(os.path.join(output_dir, d))]
print(f"Found subdirectories: {', '.join(subdirs)}")

# Remove everything except FinalVolumeSlice
for subdir in subdirs:
    if subdir != 'FinalVolumeSlice':
        path = os.path.join(output_dir, subdir)
        print(f"  Removing: {subdir}/")
        shutil.rmtree(path)

# Also remove any files in the root (keep only the directory)
for item in os.listdir(output_dir):
    item_path = os.path.join(output_dir, item)
    if os.path.isfile(item_path):
        print(f"  Removing file: {item}")
        os.remove(item_path)

# Tar  the output:
tar_name = 'SaveWood.tar.gz'
with tarfile.open(tar_name, 'w:gz') as tar:
    tar.add(output_dir)

# Only retrieve the tar file and output_dir.txt

# Write the directory name to a file so the workflow can read it
with open('output_dir.txt', 'w') as f:
    f.write(output_dir)

print(f"\nOutput directory name written to: output_dir.txt")
print(f"Done!")
