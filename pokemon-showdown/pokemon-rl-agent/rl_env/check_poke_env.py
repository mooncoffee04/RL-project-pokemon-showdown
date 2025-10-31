"""
check_poke_env.py: Diagnostic script to verify poke-env installation

This script checks what's available in your poke-env installation
to help debug import issues.
"""

print("="*70)
print(" POKE-ENV INSTALLATION CHECK")
print("="*70)

# Check poke-env version
try:
    import poke_env
    print(f"\n✅ poke-env is installed")
    print(f"   Version: {poke_env.__version__ if hasattr(poke_env, '__version__') else 'Unknown'}")
    print(f"   Location: {poke_env.__file__}")
except ImportError as e:
    print(f"\n❌ poke-env is NOT installed: {e}")
    print("   Please install it with: pip install poke-env")
    exit(1)

# Check poke_env.player module
print("\n" + "-"*70)
print(" Checking poke_env.player module...")
print("-"*70)

try:
    from poke_env import player
    print("✅ poke_env.player module found")
    
    # List all available classes in player module
    player_classes = [name for name in dir(player) if not name.startswith('_')]
    print(f"\n   Available classes/functions ({len(player_classes)}):")
    for cls in sorted(player_classes):
        print(f"   - {cls}")
        
except ImportError as e:
    print(f"❌ Could not import poke_env.player: {e}")

# Check for Gen9EnvSinglePlayer specifically
print("\n" + "-"*70)
print(" Checking for Gen9EnvSinglePlayer...")
print("-"*70)

try:
    from poke_env.player import Gen9EnvPlayer
    print("✅ Gen9EnvPlayer is available!")
    print(f"   Class: {Gen9EnvPlayer}")
    print(f"   Module: {Gen9EnvPlayer.__module__}")
except ImportError as e:
    print(f"❌ Gen9EnvPlayer NOT found: {e}")
    print("\n   Trying alternative imports...")
    
    # Try other possible locations
    alternatives = [
        "poke_env.player.env_player.Gen9EnvPlayer",
        "poke_env.player.EnvPlayer.Gen9EnvPlayer",
        "poke_env.player.gen_env_player.Gen9EnvPlayer",
    ]
    
    for alt in alternatives:
        try:
            parts = alt.rsplit('.', 1)
            module_path = parts[0]
            class_name = parts[1]
            exec(f"from {module_path} import {class_name}")
            print(f"   ✅ Found at: {alt}")
            break
        except ImportError:
            print(f"   ❌ Not at: {alt}")

# Check other important imports
print("\n" + "-"*70)
print(" Checking other required imports...")
print("-"*70)

imports_to_check = [
    ("poke_env.player", "RandomPlayer"),
    ("poke_env.player", "Player"),
    ("poke_env.data", "GenData"),
    ("gymnasium.spaces", "Box"),
    ("stable_baselines3", "PPO"),
]

for module_name, class_name in imports_to_check:
    try:
        exec(f"from {module_name} import {class_name}")
        print(f"✅ {module_name}.{class_name}")
    except ImportError as e:
        print(f"❌ {module_name}.{class_name}: {e}")

# Final summary
print("\n" + "="*70)
print(" SUMMARY")
print("="*70)

try:
    from poke_env.player import Gen9EnvSinglePlayer
    print("✅ Your poke-env installation looks good!")
    print("   You should be able to use Gen9EnvSinglePlayer")
except ImportError:
    print("❌ There's an issue with your poke-env installation")
    print("\n   RECOMMENDED FIXES:")
    print("   1. Reinstall poke-env:")
    print("      pip uninstall poke-env -y")
    print("      pip install poke-env==0.10.0")
    print("\n   2. If that doesn't work, try installing from source:")
    print("      pip install git+https://github.com/hsahovic/poke-env.git")

print("="*70 + "\n")