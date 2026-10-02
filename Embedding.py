# Source code used from: https://huggingface.co/spaces/hesamation/primer-llm-embedding

import torch
import torch.nn as nn
from collections.abc import Sequence
from typing import Any, cast
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.embedding(input_ids))

def find_similar_embeddings(target_embedding: torch.Tensor | Sequence[float], n: int = 10,) -> list[tuple[str, float]]:
    """
    Find the n most similar embeddings to the target embedding using cosine similarity
    Args:
        target_embedding: The embedding vector to compare against
        n: Number of similar embeddings to return (default 3)
    Returns:
        List of tuples containing (word, similarity_score) sorted by similarity
    """
    if isinstance(target_embedding, torch.Tensor):
        target_tensor: torch.Tensor = target_embedding
    else:
        target_tensor = torch.tensor(target_embedding, dtype=torch.float32)

    all_embeddings: torch.Tensor = model.embedding.weight
    similarities: torch.Tensor = torch.nn.functional.cosine_similarity(target_tensor.unsqueeze(0), all_embeddings)
    top_n_similarities: torch.Tensor
    top_n_indices: torch.Tensor
    top_n_similarities, top_n_indices = torch.topk(similarities, n)
    results: list[tuple[str, float]] = []
    for idx, score in zip(top_n_indices, top_n_similarities):
        decoded: Any = tokenizer.decode(idx)  # pyright: ignore[reportUnknownMemberType]
        word: str = decoded if isinstance(decoded, str) else decoded[0]
        results.append((word, float(score.item())))
    return results

def prompt_to_embeddings(prompt: str) -> tuple[list[int], torch.Tensor, list[str]]:
    tokens: Any = tokenizer(prompt, return_tensors="pt")  # pyright: ignore[reportUnknownMemberType]
    input_ids: torch.Tensor = cast(torch.Tensor, tokens["input_ids"])
    outputs: torch.Tensor = model(input_ids)
    embeddings: torch.Tensor = outputs
    token_id_list: list[int] = tokenizer.encode(prompt, add_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]
    token_str: list[str] = []
    for t_id in token_id_list:
        decoded: Any = tokenizer.decode(t_id, skip_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]
        token_str.append(decoded if isinstance(decoded, str) else decoded[0])
    return token_id_list, embeddings, token_str

def print_tokens_and_embeddings(prompt: str,) -> tuple[list[int], torch.Tensor, list[str]]:
    token_id_list, embeddings, token_str = prompt_to_embeddings(prompt)
    tokens_and_ids: list[tuple[str, int]] = list(zip(token_str, token_id_list))
    print(f"Tokens & IDs: {tokens_and_ids}")
    print(f"Embeddings shape: {embeddings.shape}")
    return token_id_list, embeddings, token_str

vocab_size: int = 151936
dimensions: int = 1536
tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
embeddings_filename: str = r"embeddings_qwen.pth"
model_name: str = tokenizer_name

tokenizer: PreTrainedTokenizerBase = cast(
    PreTrainedTokenizerBase,
    AutoTokenizer.from_pretrained(tokenizer_name),  # pyright: ignore[reportUnknownMemberType]
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # pyright: ignore[reportUnknownMemberType]
pretrained_model: PreTrainedModel = cast(
    PreTrainedModel,
    AutoModel.from_pretrained(model_name),  # pyright: ignore[reportUnknownMemberType]
)
embeddings_layer: nn.Module = pretrained_model.get_input_embeddings()
print(f"Extracted Embeddings Layer for {model_name}: {embeddings_layer}")
state_dict: dict[str, Any] = embeddings_layer.state_dict()
torch.save(state_dict, embeddings_filename)

# Need to reload the tokenizer
tokenizer = cast(
    PreTrainedTokenizerBase,
    AutoTokenizer.from_pretrained(tokenizer_name),  # pyright: ignore[reportUnknownMemberType]
)
model: EmbeddingModel = EmbeddingModel(vocab_size, dimensions)
saved_embeddings: dict[str, torch.Tensor] = cast(
    dict[str, torch.Tensor],
    torch.load(embeddings_filename),  # pyright: ignore[reportUnknownMemberType]
)
if "weight" not in saved_embeddings:
    raise KeyError("The saved embeddings file does not contain 'weight' key.")
embeddings_tensor: torch.Tensor = saved_embeddings["weight"]
if embeddings_tensor.size() != (vocab_size, dimensions):
    raise ValueError(
        f"The dimensions of the loaded embeddings do not match the model's expected dimensions ({vocab_size}, {dimensions})."
    )
model.embedding.weight.data = embeddings_tensor
model.eval()

print_tokens_and_embeddings("HTML coders are not considered programmers")