from transformers import AutoTokenizer
import torch

# Load a tokenizer (example with GPT-2)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Example token IDs
token_ids = [15496, 59, 65, 198]  # Example IDs

# Method 1: Standard decode (may interpret escape sequences)
decoded_standard = tokenizer.decode(token_ids)
print(f"Standard decode: {repr(decoded_standard)}")

# Method 2: Decode without special token processing
decoded_clean = tokenizer.decode(token_ids, skip_special_tokens=True)
print(f"Clean decode: {repr(decoded_clean)}")

# Method 3: Convert tokens to strings individually (most raw approach)
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(f"Individual tokens: {tokens}")

# Method 4: Manual reconstruction to preserve literal characters
raw_string = tokenizer.convert_tokens_to_string(tokens)
print(f"Raw string: {repr(raw_string)}")


# Method 5: For handling specific cases like \b
def decode_raw_preserving_escapes(tokenizer, token_ids):
    """
    Decode token IDs while preserving literal escape sequences
    """
    # Convert to tokens first
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Convert back to string using tokenizer's method
    decoded = tokenizer.convert_tokens_to_string(tokens)

    # If you need to ensure \b stays as literal \b and not backspace:
    # The tokenizer usually preserves this correctly, but if not:
    # decoded = decoded.replace('\b', '\\b')

    return decoded


# Example with potential \b sequence
example_text = "Hello\\bWorld"  # Text containing literal \b
encoded = tokenizer.encode(example_text)
print(f"\nOriginal text: {repr(example_text)}")
print(f"Encoded: {encoded}")

# Decode using different methods
decoded_1 = tokenizer.decode(encoded)
decoded_2 = decode_raw_preserving_escapes(tokenizer, encoded)

print(f"Standard decode: {repr(decoded_1)}")
print(f"Raw decode: {repr(decoded_2)}")


# For batch processing
def batch_decode_raw(tokenizer, batch_token_ids):
    """
    Decode a batch of token ID sequences to raw strings
    """
    if isinstance(batch_token_ids, torch.Tensor):
        batch_token_ids = batch_token_ids.tolist()

    decoded_strings = []
    for token_ids in batch_token_ids:
        # Remove padding tokens if present
        if tokenizer.pad_token_id is not None:
            token_ids = [tid for tid in token_ids if tid != tokenizer.pad_token_id]

        decoded = decode_raw_preserving_escapes(tokenizer, token_ids)
        decoded_strings.append(decoded)

    return decoded_strings


# Example batch processing
batch_ids = [[15496, 59, 65], [18435, 11, 995]]
batch_decoded = batch_decode_raw(tokenizer, batch_ids)
print(f"\nBatch decoded: {batch_decoded}")

# Additional options for specific tokenizers
print(f"\nTokenizer special tokens:")
print(f"BOS token: {repr(tokenizer.bos_token)}")
print(f"EOS token: {repr(tokenizer.eos_token)}")
print(f"PAD token: {repr(tokenizer.pad_token)}")
print(f"UNK token: {repr(tokenizer.unk_token)}")


# Clean decode without any special tokens
def decode_completely_raw(tokenizer, token_ids):
    """
    Most aggressive raw decoding - removes all special tokens
    """
    # Filter out special tokens
    special_token_ids = set(
        [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        ]
    )

    # Remove None values and filter special tokens
    special_token_ids = {tid for tid in special_token_ids if tid is not None}
    filtered_ids = [tid for tid in token_ids if tid not in special_token_ids]

    return tokenizer.decode(filtered_ids, skip_special_tokens=True)
