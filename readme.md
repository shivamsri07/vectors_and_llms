
# Product Quantization (PQ)
🧩 Compress blocks of the vector, using codebooks.

🔍 How it works:
1. Divide the vector into m equal parts (say, 128-dim → 8 sub-vectors of 16 dims each).
2. Each sub-vector is quantized using its own KMeans codebook.
3. The full vector is then represented as a list of codebook indices.

------------

# Modern LLM Architecture

1. The Foundation: The Attention Mechanism
Problem: Older models like RNNs struggled to remember relationships between words that were far apart in a sequence.
Solution: A mechanism that allows every token to directly look at and score its relationship with every other token in the sequence. It creates context-rich representations by weighing the importance of other words.
New Problem It Created: The calculation is computationally expensive, with its complexity growing quadratically (O(N2)) with the sequence length (N). This makes it very slow and memory-hungry for long texts.
-------
2. The Speed Boost: FlashAttention
Problem It Solves: Standard attention is "memory-bound." It wastes most of its time reading and writing a massive intermediate attention matrix to the GPU's slow main memory (HBM).
The Core Idea (Kernel Fusion & Tiling):
- Break the large attention calculation into smaller tiles.
- Load one tile of inputs into the GPU's tiny, ultra-fast on-chip memory (SRAM).
- Perform the entire chain of operations (scores, softmax, multiplication) for that tile inside SRAM, without ever writing the intermediate matrix back to slow memory.
- Write only the final, small output for that tile back to HBM.
Key Benefit: Dramatically reduces slow memory access, making attention much faster and changing the memory usage from O(N2) to O(N).
-------
3. The Memory Manager: PagedAttention
Problem It Solves: When generating text, we must store a large KV Cache. For multiple users, managing this cache is a nightmare, leading to massive wasted memory (fragmentation), just like a poorly managed parking garage.
The Core Idea (Virtual Memory for GPUs):
- It borrows the concept of "paging" from operating systems.
- The entire GPU memory for the KV cache is pre-divided into small, fixed-size blocks.
- A sequence's KV cache is stored in these blocks, which can be scattered all over memory (they are not contiguous).
- A "block table" for each sequence acts as an index to keep track of where its data is.
Key Benefit: Nearly eliminates memory waste, allowing for much higher throughput (more users served at once) and efficient sharing of memory for common prompts.
------
4. The Scalable Architecture: Mixture of Experts (MoE)
Problem It Solves: Simply making a "dense" model bigger is unsustainable. The computational cost scales directly with the number of parameters, making giant models too slow and expensive.
The Core Idea (A Team of Specialists):
- Instead of one massive Feed-Forward Network (FFN), an MoE layer has many smaller "Expert" FFNs.
- A small, fast "Gating Network" (or Router) looks at each token and dynamically chooses which 2 (or so) experts are best suited to process it.
Key Benefit: You get the knowledge capacity of a model with a massive number of total parameters, but the computational cost of a much smaller model, because only a fraction of the parameters are "active" for any given token.
------
5. The Learning Process: How Weights are Forged
The Dataset: There is one single, massive dataset (e.g., a huge portion of the public internet). There are no separate datasets for Q, K, and V.
The Goal: The model is trained on one simple, self-supervised task: Predict the Next Word.
The Unified Process:
- All weights in the entire network (W_Q, W_K, W_V, FFN weights, etc.) start as random numbers.
- The model takes a text sequence, makes a (garbage) prediction for the next word.
- The prediction is compared to the actual correct word to calculate an error (loss).
- A backpropagation signal travels backward through the entire network, simultaneously nudging all weights in a direction that would have made the error smaller.
This process is repeated billions/trillions of times, molding the random weights into sophisticated matrices that encode the patterns of human language.

-----------------------------------------------
