# Attentive Seq2Seq
*Hybrid LSTM + Self-Attention Seq2Seq model with training loss visualization*

---

## Description
This project implements a custom **sequence-to-sequence model** that combines **LSTM recurrence** with **transformer-style self-attention**.  
It is designed to retain sequential information while also leveraging context-aware attention, making it useful for tasks like toy translation experiments and exploring attention mechanisms.

---

## Features
- Custom **Self-Attention layer** (implemented from scratch in Keras)
- **Hybrid Encoder–Decoder** with LSTM for sequence retention
- **Additive Attention** for decoder–encoder context
- Training pipeline with **loss visualization**
- Ready-to-use in **Google Colab**

---

## Installation
Clone the repo:
```bash
git clone https://github.com/your-username/attentive-seq2seq.git
cd attentive-seq2seq

Usage:
Check out the notebooks/ folder for training in Colab.
Run the src/model.py script to build and train the model.
Modify input/target sentences in the script for experimentation.

Repository structure:
attentive-seq2seq/
├── src/
│   └── model.py       # Main model code
├── notebooks/
│   └── training.ipynb # Colab training notebook
├── requirements.txt
└── README.md

Notes:
This is a learning project, not production-ready.
Contributions and suggestions are welcome!
