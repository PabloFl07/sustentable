# Stepper playground

From the repository root, with `uv` installed:

```powershell
uv run python -m examples.stepper_prototype.app
```

Choose the layout, steps, navigation, validation, and grid placement. Click **Use sample dataset** to try the flow. Changing an option or clicking **Reset example** starts fresh.

This is UI only: no file loading or data processing. Step content lives in `*_step.py`; `workflow.py` arranges it, and `preview.py` supplies the sample grid.
