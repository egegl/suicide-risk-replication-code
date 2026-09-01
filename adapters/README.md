# R9 adapter placement

Place the two final full-data LoRA adapters here before inference:

```text
adapters/
  seed1-step160/
    adapter_config.json
    adapter_model.safetensors
  seed2-step120/
    adapter_config.json
    adapter_model.safetensors
```

These weights are not stored in Git. `predict.py` validates both directories before loading the base model.
