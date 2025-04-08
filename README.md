# 🧠 Abstractor: Attention-Powered Summarizer

This project introduces a clean and minimal attention-based model for abstractive summarization. It ditches complex structures like RNNs and CNNs, focusing entirely on attention to understand and generate concise versions of input text.

Unlike traditional sequence models, this one simplifies the architecture to make training and interpretation easier, while still capturing meaningful dependencies across long inputs.

---

## 🔍 How It Works

The model maps sequences through a series of attention layers that learn which parts of the input text to focus on when creating each word of the summary.

It relies on:

- Query, key, and value vector representations
- Weighted combinations to derive contextual understanding
- A modular, layer-based implementation

---

## 📝 Notes

* Uses compact GloVe embeddings  
* Smaller number of layers for faster prototyping  
* Built for mini-batch training  
* Current version is untrained (demo stage only)  
* Hyperparameters are minimal and customizable  

---

## ✅ Tested With:

* Python 2.7.1  
* TensorFlow 1.3  
* NumPy 1.13.3  

---
