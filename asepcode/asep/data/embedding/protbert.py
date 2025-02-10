from transformers import T5Tokenizer, T5EncoderModel
import torch, re

# Load the ProtBERT model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model.eval()


def embed_protbert(protein_sequence: str) -> torch.Tensor:
    """
    Generate ProtBERT embeddings for a given protein sequence.

    Args:
        protein_sequence (str): The protein sequence to embed.

    Returns:
        torch.Tensor: A tensor of shape (L, 1024) where L is the length of the sequence.
    """
    # print(f"Input sequence: {protein_sequence}")  # Debugging
    len_seq = len(protein_sequence)

    protein_sequence = [protein_sequence]
    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    protein_sequence = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_sequence][0]

    # Tokenize the sequence
    inputs = tokenizer(
        protein_sequence, 
        return_tensors="pt", 
        padding=True, 
        truncation=False, 
        max_length=1024  # Set a large enough max_length to avoid truncation
    )
    # print(f"Tokenizer output: {inputs}")  # Debugging
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings for the last layer
    residue_embeddings = outputs.last_hidden_state[0, :len_seq]
    
    # print(f"Embeddings shape before filtering: {residue_embeddings.shape}")  # Debugging
    return residue_embeddings


# Example usage
if __name__ == "__main__":
    # sequence = "ACDEF"
    sequence_examples = "PRTEINO"
    embeddings = embed_protbert(sequence_examples)
    print(f"Embeddings shape: {embeddings.shape}")