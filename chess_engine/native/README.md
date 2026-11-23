# Native search core

`native_eval.cpp` contains the C++ search and evaluation core. The Python layer
loads it through `ctypes` (`native_eval.py`). It is required: all searching is
done natively.

## Build (Windows, MSVC)

```bat
build_native.bat
```

or manually from a Developer Command Prompt at the repository root:

```bat
cl /nologo /O2 /EHsc /LD /Fe:chess_engine\native\deeptactics_native_runtime_v2.dll chess_engine\native\native_eval.cpp
```

Check the result with `python -m chess_engine native-status`.
