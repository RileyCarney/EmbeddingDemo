# Source code used from: https://huggingface.co/spaces/hesamation/primer-llm-embedding

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    def forward(self, input_ids):
        return self.embedding(input_ids)

def find_similar_embeddings(target_embedding, n=10):
    """
    Find the n most similar embeddings to the target embedding using cosine similarity
    Args:
        target_embedding: The embedding vector to compare against
        n: Number of similar embeddings to return (default 3)
    Returns:
        List of tuples containing (word, similarity_score) sorted by similarity
    """
    if not isinstance(target_embedding, torch.Tensor):
        target_embedding = torch.tensor(target_embedding)
    all_embeddings = model.embedding.weight
    similarities = torch.nn.functional.cosine_similarity(
        target_embedding.unsqueeze(0),
        all_embeddings
    )
    top_n_similarities, top_n_indices = torch.topk(similarities, n)
    results = []
    for idx, score in zip(top_n_indices, top_n_similarities):
        word = tokenizer.decode(idx)
        results.append((word, score.item()))
    return results

def prompt_to_embeddings(prompt:str):
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']
    outputs = model(input_ids)
    embeddings = outputs
    token_id_list = tokenizer.encode(prompt, add_special_tokens=True)
    token_str = [tokenizer.decode(t_id, skip_special_tokens=True) for t_id in token_id_list]
    return token_id_list, embeddings, token_str

def print_tokens_and_embeddings(prompt: str):
    token_id_list, embeddings, token_str = prompt_to_embeddings(prompt)
    print(f"Tokens & IDs: {list(zip(token_str, token_id_list))}")
    print(f"Embeddings shape: {embeddings.shape}")
    return token_id_list, embeddings, token_str

vocab_size = 151936
dimensions = 1536
tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
embeddings_filename = r"embeddings_qwen.pth"
model_name = tokenizer_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModel.from_pretrained(model_name)
embeddings = model.get_input_embeddings()
print(f"Extracted Embeddings Layer for {model_name}: {embeddings}")
torch.save(embeddings.state_dict(), "embeddings_qwen.pth") 

# Need to reload the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = EmbeddingModel(vocab_size, dimensions) and all embeddings
saved_embeddings = torch.load(embeddings_filename)
if 'weight' not in saved_embeddings:
    raise KeyError("The saved embeddings file does not contain 'weight' key.")
embeddings_tensor = saved_embeddings['weight']
if embeddings_tensor.size() != (vocab_size, dimensions):
    raise ValueError(f"The dimensions of the loaded embeddings do not match the model's expected dimensions ({vocab_size}, {dimensions}).")
model.embedding.weight.data = embeddings_tensor
model.eval()

print_tokens_and_embeddings("HTML coders are not considered programmers")