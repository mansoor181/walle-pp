from antiberty import AntiBERTyRunner
import torch

# pip install antiberty

# Initialize the AntiBERTy runner
antiberty = AntiBERTyRunner()

def embed_antiberty(protein_sequence: str) -> torch.Tensor:
    """
    Generate AntiBERTy embeddings for a given protein sequence.

    Args:
        protein_sequence (str): The protein sequence to embed.

    Returns:
        torch.Tensor: A tensor of shape (L, 512) where L is the length of the sequence.
    """
    # Ensure the sequence is in a list format as required by AntiBERTy
    sequences = [protein_sequence]

    # Get the embeddings from AntiBERTy
    embeddings = antiberty.embed(sequences)

    # The embeddings are returned as a list of tensors, each of shape [(Length + 2) x 512]
    # We need to remove the first and last tokens (CLS and SEP tokens) to get the residue embeddings
    residue_embeddings = embeddings[0][1:-1, :]  # Remove CLS and SEP tokens

    return residue_embeddings

# Example usage
if __name__ == "__main__":
    sequence_examples = "PRTEINO"
    embeddings = embed_antiberty(sequence_examples)
    print(f"Embeddings shape: {embeddings.shape}")