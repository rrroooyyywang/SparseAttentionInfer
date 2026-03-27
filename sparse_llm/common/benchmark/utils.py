import gc

import torch


def sample_next_token(
    next_token_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    next_token_logits = next_token_logits.float()
    if temperature <= 0:
        return next_token_logits.argmax(dim=-1, keepdim=True)

    scaled_logits = next_token_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
