🧬 1. Product Quantization (PQ)
🧩 Compress blocks of the vector, using codebooks.

🔍 How it works:
1. Divide the vector into m equal parts (say, 128-dim → 8 sub-vectors of 16 dims each).
2. Each sub-vector is quantized using its own KMeans codebook.
3. The full vector is then represented as a list of codebook indices.

------------
