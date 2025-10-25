import torch


def sine_pe_2d(
    embedding_dim: int, height: int, width: int, temperature: int = 10_000
) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings.

    Creates 2D positional embeddings using sine and cosine functions with different
    frequencies for both spatial dimensions. The first half of the embedding dimension
    encodes the y-coordinate (height), the second half encodes the x-coordinate (width).

    Args:
        embedding_dim (int): The dimension of the positional embedding.
            Must be divisible by 4.
        height (int): The height of the spatial grid.
        width (int): The width of the spatial grid.
        temperature (int, optional): The base for computing frequency scales.
            Defaults to 10_000.

    Returns:
        torch.Tensor: A tensor of shape (embedding_dim, height, width) containing the
            2D positional embeddings.

    Raises:
        ValueError: If embedding_dim is not divisible by 4.

    Example:
        >>> pe = sine_pe_2d(256, 32, 32)
        >>> pe.shape
        torch.Size([256, 32, 32])
    """
    if embedding_dim % 4 != 0:
        raise ValueError("Embedding dimension must be divisible by 4.")

    half_embedding = embedding_dim // 2
    quarter_embedding = half_embedding // 2

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )

    freqs = temperature ** (
        -torch.arange(0, quarter_embedding, dtype=torch.float32) / quarter_embedding
    )

    y_freqs = torch.einsum("i,jk->ijk", freqs, y_coords)
    x_freqs = torch.einsum("i,jk->ijk", freqs, x_coords)

    pe = torch.empty(embedding_dim, height, width, dtype=torch.float32)

    pe[0:half_embedding:2] = torch.sin(y_freqs)
    pe[1:half_embedding:2] = torch.cos(y_freqs)
    pe[half_embedding::2] = torch.sin(x_freqs)
    pe[half_embedding + 1 :: 2] = torch.cos(x_freqs)

    return pe
