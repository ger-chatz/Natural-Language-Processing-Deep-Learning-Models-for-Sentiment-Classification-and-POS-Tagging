# Neural Models for Part-of-Speech Tagging

A comparative NLP project that follows the progression from a fixed-window Multi-Layer Perceptron to a fine-tuned BERT Transformer for token-level Part-of-Speech tagging.

**Category:** Natural Language Processing & Deep Learning  
**Models:** Window MLP · BiLSTM · Stacked CNN · BERT  
**Dataset:** Universal Dependencies — English EWT  
**Frameworks:** PyTorch · Hugging Face Transformers

<p align="center">
  <img src="assets/pos-tagging-neural-models.png" alt="Comparison of MLP, BiLSTM, CNN and BERT for POS tagging" width="100%">
</p>

---

## Project Overview

The project compares four neural architectures on the same sequence-labelling task: given an English sentence, predict one Universal Part-of-Speech tag for every word.

The four notebooks were developed progressively:

1. a **window-based MLP** that uses a fixed local context;
2. a **bidirectional LSTM** that reads the complete sentence in both directions;
3. a **stacked CNN** that learns local n-gram patterns at every word position;
4. a fine-tuned **BERT** model that produces contextual representations from pre-trained Transformer layers.

The goal is not only to identify the strongest model, but also to show how different neural architectures represent linguistic context.

---

## Task

For an input sentence such as:

```text
The quick brown fox jumps over the lazy dog .
```

the model predicts one UPOS tag per word:

```text
DET ADJ ADJ NOUN VERB ADP DET ADJ NOUN PUNCT
```

This is a **token-classification** problem with 17 Universal POS tags:

`ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `SYM`, `VERB`, `X`.

---

## Dataset

The experiments use the **Universal Dependencies English Web Treebank (UD English EWT)** with its official train, development and test splits.

| Split | Sentences | Tokens | Average sentence length |
|---|---:|---:|---:|
| Train | 12,544 | 204,578 | 16.31 |
| Development | 2,001 | 25,148 | 12.57 |
| Test | 2,077 | 25,094 | 12.08 |

From the CoNLL-U files, the notebooks retain the word form and its UPOS tag. Multi-word token lines and empty nodes are skipped.

---

## Models

### 1. Window-based MLP

[Open notebook](notebooks/01_window_mlp_pos_tagger.ipynb)

The first model predicts the tag of the centre word from a fixed five-word window:

```text
two previous words + current word + two next words
```

Main configuration:

- 50-dimensional pre-trained GloVe embeddings;
- embeddings fine-tuned during training;
- 5-word input window;
- hidden layers of 256 and 128 units;
- ReLU activations;
- dropout of 0.30;
- Adam optimiser with learning rate 0.001;
- best checkpoint selected by development loss.

A small development-set exploration compared window radius, hidden size and dropout. The final full run reached its best development loss around epoch 8.

### 2. Bidirectional LSTM

[Open notebook](notebooks/02_bilstm_pos_tagger.ipynb)

The BiLSTM reads the complete sentence in both directions. Each word representation therefore contains left and right context before the output layer predicts its POS tag.

Main configuration:

- learned 80-dimensional word embeddings;
- two bidirectional LSTM layers;
- hidden size 128 per direction;
- dropout of 0.25;
- packed padded sequences;
- Adam optimiser with learning rate 0.001;
- gradient clipping;
- padding tags ignored by the loss;
- best checkpoint selected by development loss.

A small dropout comparison showed that 0.25 performed better than 0.10 in the saved experiments.

### 3. Stacked CNN

[Open notebook](notebooks/03_stacked_cnn_pos_tagger.ipynb)

The CNN retains one representation for every word position rather than using global pooling. This makes it suitable for sequence labelling.

Main configuration:

- learned 80-dimensional word embeddings;
- projection to 128 channels;
- two stacked CNN blocks;
- convolutional filter sizes 2, 3 and 4;
- residual connections;
- layer normalisation;
- dropout of 0.25;
- position-wise output layer;
- Adam optimiser with learning rate 0.001;
- best checkpoint selected by development loss.

The multiple filter sizes allow the model to learn local bigram, trigram and four-token patterns around each position.

### 4. Fine-tuned BERT

[Open notebook](notebooks/04_bert_pos_tagger.ipynb)

The final model uses `bert-base-cased` for token classification.

Because BERT may divide a word into multiple WordPieces, the gold POS tag is assigned only to the first WordPiece. The remaining subword positions receive label `-100` and are ignored by the loss.

Three fine-tuning settings were compared:

| Setting | Trainable BERT blocks | Best epoch | Development Macro-F1 |
|---|---:|---:|---:|
| Last 4 blocks | 4 | 4 | 0.9120 |
| Last 8 blocks | 8 | 3 | 0.9354 |
| Full BERT | 12 | 4 | **0.9385** |

Additional settings:

- learning rate `2e-5`;
- classifier dropout `0.10`;
- weight decay `0.01`;
- training batch size `16`;
- evaluation batch size `32`;
- maximum length `512` WordPieces;
- model selection based on development Macro-F1.

Full fine-tuning produced the strongest development result and was used for the final test evaluation.

---

## Evaluation

The primary metric is **Macro-F1**, which gives equal importance to frequent and rare tags.

The notebooks also report:

- Macro precision;
- Macro recall;
- Macro PR-AUC;
- per-tag precision, recall, F1 and PR-AUC;
- training and development loss curves;
- qualitative tagging examples;
- error analysis for easier, harder, known and unknown words.

The test set is evaluated only after model selection on the development set.

---

## Test Results

| Model | Macro precision | Macro recall | Macro-F1 | Macro PR-AUC |
|---|---:|---:|---:|---:|
| Most-frequent-tag baseline | 0.8252 | 0.7800 | 0.7924 | 0.6938 |
| Window MLP | 0.8689 | 0.8463 | **0.8563** | 0.8983 |
| BiLSTM | 0.8562 | 0.8071 | 0.8278 | 0.8709 |
| Stacked CNN | 0.8523 | 0.8214 | 0.8348 | 0.8770 |
| Fine-tuned BERT | **0.9538** | **0.9419** | **0.9463** | **0.9655** |

BERT improved test Macro-F1 by **0.0900** over the Window MLP, which was the strongest non-Transformer model in these experiments.

The results also show that greater architectural complexity does not automatically guarantee a better score. In these specific runs, the Window MLP outperformed both the BiLSTM and CNN, while BERT's pre-trained contextual representations produced a clear improvement over all earlier models.

---

## Optional LLM Experiment

The BERT notebook also contains a small bonus experiment using Gemini 2.5 Flash-Lite as a prompted POS tagger on 10 short test sentences.

It achieved:

- token accuracy: `0.91`;
- Macro-F1: `0.78`.

This result is only indicative because it was calculated on a very small sample and is not directly comparable with the full supervised test evaluation.

---

## Repository Structure

```text
pos-tagging-neural-models/
│
├── README.md
├── requirements.txt
│
├── assets/
│   └── pos-tagging-neural-models.png
│
└── notebooks/
    ├── 01_window_mlp_pos_tagger.ipynb
    ├── 02_bilstm_pos_tagger.ipynb
    ├── 03_stacked_cnn_pos_tagger.ipynb
    └── 04_bert_pos_tagger.ipynb
```

The four notebooks should be kept in one repository because they form a single comparative study rather than four unrelated projects.

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/ger-chatz/pos-tagging-neural-models.git
cd pos-tagging-neural-models
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

Open one of the notebooks locally or in Google Colab and run the cells in order.

The notebooks download the UD English EWT data automatically. The BERT notebook benefits substantially from a GPU runtime.

---

## Main Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- GloVe embeddings
- pandas and NumPy
- Matplotlib
- Google Colab

---

## Key Takeaways

- A fixed context window already provides a strong improvement over word-level frequency lookup.
- BiLSTM and CNN models represent sentence context in different ways: recurrent sequence processing versus local convolutional patterns.
- Model complexity alone did not determine performance in these runs.
- Development-based checkpoint selection was important for controlling overfitting.
- Fine-tuned BERT was clearly the strongest model, particularly because POS tagging depends heavily on sentence context.
- Rare and irregular tags remained more difficult than frequent, structurally consistent tags.

---

## Academic Context

Developed for the **Text Analytics** course of the MSc in Artificial Intelligence & Data Science at the Athens University of Economics and Business.

**Instructor:** Ion Androutsopoulos  
**Repository curated by:** Gerasimos Chatzopoulos
