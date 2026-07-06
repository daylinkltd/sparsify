"""Efficient perplexity and accuracy evaluator using MLX."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn.losses as losses


# A fallback high-quality evaluation text (approx 15,000 characters)
# containing standard English structures, tech topics, and logic.
_DEFAULT_EVAL_TEXT = """
The Llama model is a family of autoregressive large language models released by Meta.
Language modeling is the use of various statistical and probabilistic techniques to determine
the probability of a given sequence of words occurring in a sentence. Language models analyze
body of text data to provide a basis for their word predictions. They are used in natural
language processing (NLP) applications, particularly those that generate text.
An operating system (OS) is system software that manages computer hardware, software resources,
and provides common services for computer programs. Time-sharing operating systems schedule
tasks for efficient use of the system and may also include accounting software for cost allocation
of processor time, mass storage, printing, and other resources.
For hardware functions such as input and output and memory allocation, the operating system acts
as an intermediary between programs and the computer hardware.
A compiler is a computer program that translates computer code written in one programming
language (the source language) into another language (the target language). The name "compiler"
is primarily used for programs that translate source code from a high-level programming language
to a lower-level language (e.g. assembly language, object code, or machine code) to create an
executable program.
In database management systems, a query is a request for data or information from a database table
or combination of tables. This data may be generated as results returned by Structured Query Language
(SQL) or as pictorials, graphs, or complex analyses.
Attention is a mechanism that mimics cognitive attention. The effect enhances the important parts
of the input data while fading out the rest. The design allows the network to focus on specific aspects
of the input context, mimicking how human cognition selectively filters information.
Machine learning infrastructure refers to the software stack, compute engines, storage, and networking
required to train, evaluate, and run inference on machine learning models at scale.
Sparsity in machine learning refers to the condition where a significant fraction of parameters,
weights, or activations in a model are zero or near-zero. Exploiting sparsity can lead to faster
execution speeds and lower memory consumption during both training and inference.
 commodity hardware is computer hardware that is affordable and easy to obtain. It is typically
compatible with other hardware products and does not require custom manufacturing or special components.
Fanless systems are computer systems that do not use active cooling fans. They rely on passive heat sinks
and natural convection to dissipate heat, making them silent and less prone to mechanical failure.
Edge devices are physical pieces of hardware that control data flow at the boundary between two networks.
They serve as entry or exit points to enterprise or service provider core networks.
Consumer desktops are personal computers intended for use at a single location on or near a desk or table
due to their size and power requirements. They are typically used for gaming, productivity, and web browsing.
"""


def compute_cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """Compute cross-entropy loss between logits and targets."""
    # Logits shape: (B, L, V)
    # Targets shape: (B, L)
    # returns average cross entropy per token
    return mx.mean(losses.cross_entropy(logits, targets))


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    text: str | None = None,
    seq_len: int = 256,
    max_tokens: int = 8192,
    batch_size: int = 4,
) -> float:
    """Calculate the perplexity of the model over the evaluation text.

    Uses batching and Metal acceleration.
    """
    eval_text = text or _DEFAULT_EVAL_TEXT
    tokens = tokenizer.encode(eval_text)
    
    # Cap token count
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        
    # We need at least seq_len + 1 tokens
    if len(tokens) <= seq_len:
        seq_len = len(tokens) - 2
        if seq_len <= 0:
            return float("nan")

    # Construct input and target sequences
    # A sequence of length S gives input [0:S-1] and target [1:S]
    num_sequences = (len(tokens) - 1) // seq_len
    if num_sequences == 0:
        return float("nan")

    input_data = []
    target_data = []
    
    for i in range(num_sequences):
        start = i * seq_len
        input_data.append(tokens[start : start + seq_len])
        target_data.append(tokens[start + 1 : start + seq_len + 1])

    # Convert to mx.arrays
    inputs = mx.array(input_data)
    targets = mx.array(target_data)

    total_loss = 0.0
    total_tokens = 0
    
    # Process in batches
    for start_idx in range(0, num_sequences, batch_size):
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_inputs = inputs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]

        # Forward pass
        # mlx_lm models return logits from calling model(inputs)
        logits = model(batch_inputs)
        
        # Calculate loss
        loss = compute_cross_entropy_loss(logits, batch_targets)
        
        # mlx.core.eval compiles and executes the graph
        mx.eval(loss)
        
        # Accumulate loss scaled by number of tokens in the batch
        batch_tokens = (end_idx - start_idx) * seq_len
        total_loss += float(loss) * batch_tokens
        total_tokens += batch_tokens

    if total_tokens == 0:
        return float("nan")
        
    mean_loss = total_loss / total_tokens
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")
        
    return perplexity


def evaluate_accuracy_probe(
    model: Any,
    tokenizer: Any,
    text: str | None = None,
    max_tokens: int = 2048,
) -> float:
    """Measure next-token prediction accuracy (top-1 accuracy rate)."""
    eval_text = text or _DEFAULT_EVAL_TEXT
    tokens = tokenizer.encode(eval_text)
    
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        
    if len(tokens) <= 1:
        return 0.0

    inputs = mx.array(tokens[:-1]).reshape(1, -1)
    targets = mx.array(tokens[1:])

    logits = model(inputs)
    predictions = mx.argmax(logits, axis=-1).squeeze(0)
    
    mx.eval(predictions)
    
    correct = mx.sum(predictions == targets)
    accuracy = float(correct) / len(tokens[1:])
    return accuracy
