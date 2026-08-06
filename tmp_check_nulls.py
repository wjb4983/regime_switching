import json
import pathlib

paths = [
    'artifacts/experiments/model_comparison/pipeline/2449f69f70934f0e90c6c0f57f7ffdf5/comparison.json',
    'artifacts/experiments/model_comparison/pipeline/b4d55f29bc86491fadd97a7e604d2965/evaluation.json',
    'artifacts/experiments/model_comparison/pipeline/3af5eb693c394b98961c9b5a95393087/model/training_diagnostics.json',
    'artifacts/experiments/model_comparison/pipeline/3af5eb693c394b98961c9b5a95393087/model/metadata.json',
]

def contains_none(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return any(contains_none(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_none(item) for item in value)
    return False

for p in paths:
    path = pathlib.Path(p)
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    print(p, 'exists=', path.exists(), 'contains_none=', contains_none(data))
