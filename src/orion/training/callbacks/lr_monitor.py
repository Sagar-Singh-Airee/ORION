from __future__ import annotations

def learning_rates(optimizer) -> dict[str, float]:
    return {f"lr/group_{index}": group["lr"] for index, group in enumerate(optimizer.param_groups)}
