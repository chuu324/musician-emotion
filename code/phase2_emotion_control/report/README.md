# Report Assets

This directory contains charts and files for the project report.

## Charts

| File | Description |
|------|-------------|
| `figures/pearson_comparison.png` | Emotion Fidelity (Pearson r) across all 8 versions |
| `figures/fad_comparison.png` | Audio quality (FAD) comparison |
| `figures/clap_comparison.png` | Text-audio alignment (CLAP Score) |
| `figures/va_error.png` | VAE/AAE (absolute emotion error) comparison |
| `figures/va_target_pred_v3.png` | v3 target vs CLAP-predicted emotion scatter plot |
| `figures/dsp_overview.png` | Two-stage DSP pipeline diagram |

## Regenerating Charts

```bash
cd /home/eir/ddd/code/phase2_emotion_control
source ../.venv/bin/activate
python report/generate_charts.py
```

## Report Draft

See `6002_AI_Music_Generation_Report.md` in the project root.
