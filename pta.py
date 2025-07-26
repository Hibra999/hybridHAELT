import inspect

# --- FINTA ---
try:
    from finta import TA as finta_TA
    # Gather all callables that look like indicators
    finta_members = inspect.getmembers(finta_TA, predicate=inspect.isfunction)
    # Filter out “private” functions
    finta_indicators = [name for name, fn in finta_members if not name.startswith("_")]
    print(f"FINTA exposes {len(finta_indicators)} indicators:")
    for name in sorted(finta_indicators):
        print(" ", name)
except ImportError:
    print("FINTA not installed.")

# --- STOCK_INDICATORS ---
try:
    from stock_indicators import indicators as stock_ind
    # stock_ind is a package module; list its submodules/classes
    stock_members = inspect.getmembers(stock_ind, predicate=lambda x: inspect.isclass(x) or inspect.isfunction(x))
    # Again filter out internals
    stock_indicators = [name for name, obj in stock_members if not name.startswith("_")]
    print(f"\nstock_indicators exposes {len(stock_indicators)} indicators/classes:")
    for name in sorted(stock_indicators):
        print(" ", name)
except ImportError:
    print("stock_indicators not installed.")
