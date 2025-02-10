import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import esm

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the ESM-2 model and alphabet
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

# Move the model to the GPU (if available)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

def embed_esm2(protein_sequence: str) -> torch.Tensor:
    """
    Generate ESM-2 embeddings for a given protein sequence.

    Args:
        protein_sequence (str): The protein sequence to embed.

    Returns:
        torch.Tensor: A tensor of shape (L, d) where L is the length of the sequence
                      and d is the embedding dimension (1280 for esm2_t33_650M_UR50D).
    """
    # Prepare the input data
    data = [("protein", protein_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # Move tokens to the same device as the model (GPU if available)
    batch_tokens = batch_tokens.to(device)
    
    # Get the embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    # Extract the embeddings for the last layer
    token_embeddings = results["representations"][33]
    
    # Remove the batch dimension and the start/end tokens
    residue_embeddings = token_embeddings[0, 1:-1, :]
    
    return residue_embeddings

# Example usage
if __name__ == "__main__":
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    sequence="IVMTQSPKFMSTSIGDRVNITCKATQNVRTAVTWYQQKPGQSPQALIFLASNRHTGVPARFTGSGSGTDFTLTINNVKSEDLADYFCLQHWNYPLTFGSGTKLEIKRAD"
    embeddings = embed_esm2(sequence)
    print(f"Embeddings shape: {embeddings.shape}")