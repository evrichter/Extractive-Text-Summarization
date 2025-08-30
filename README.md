# 📝 Extractive Text Summarization  

This project was implemented as the **final project for the Computational Linguistics lecture at Saarland University (WS 2020/21)**.  
It combines **topic modeling (LDA with Gibbs Sampling)**, **sentence clustering**, and **TextRank** to perform extractive summarization on German texts.  

The work was complemented by a [report](./Extractive_Text_Summarization_Eva_Richter.pdf) discussing methodology, challenges, and evaluation.  

---

## 📂 Project Structure  

```text
├── data/               # CSV files (train/test documents)
├── output/             # Saved LDA model & dictionary
├── textrankmaster/     # Source files for TextRank algorithm
├── data_analysis/      # Data exploration & statistics
├── documentSummaries.py # Extract summaries from documents
├── instructions.txt
├── requirements.txt    # Dependencies
├── rogue_r.py          # ROUGE evaluation script
├── run_evaluation.sh   # Evaluate on seen/unseen data
├── run_training.sh     # Train LDA model
├── run_inference.sh    # Generate summary for a single document
├── topic_modelling/    # Code for LDA topic modeling
└── README.md
```

---

## 📥 Data  

- The project uses the **SwissText 2019 dataset** (100k German Wikipedia articles + reference summaries).  
- Due to size (~500 MB), the training file is hosted externally:  
  👉 [Download here](https://disk.yandex.com/d/RIsIKoi3JrYNTg)  
- Pretrained models are already included in the `output/` folder, so you can run evaluation and inference without retraining.  

---

## ⚙️ Execution Guide  

### 1. Set up environment  
```bash
virtualenv <env>
source <env>/bin/activate
pip install -r requirements.txt
```

### 2. Train topic model (optional)  
```bash
./run_training.sh
```
⚠️ Training on 99,000 docs took ~1.5h. Pretrained models are already provided.  

### 3. Evaluate model  
```bash
./run_evaluation.sh
```
Evaluates on 100 seen + 100 unseen test docs using ROUGE metrics.  

### 4. Run inference  
```bash
./run_inference.sh
```
Summarizes a document (default: `data/text_inference.txt`).  

---

## 📊 Results (ROUGE Scores)  

### Seen Documents  
```text
ROUGE-1:  P: 16.87  R: 31.44  F1: 19.96
ROUGE-2:  P:  3.49  R:  7.55  F1:  4.30
ROUGE-3:  P:  0.39  R:  2.45  F1:  1.23
ROUGE-4:  P:  0.36  R:  1.09  F1:  0.50
ROUGE-L:  P: 18.01  R: 31.45  F1: 21.37
ROUGE-W:  P: 10.29  R: 11.27  F1:  9.30
```

### Unseen Documents  
```text
ROUGE-1:  P: 21.34  R: 24.75  F1: 20.99
ROUGE-2:  P:  4.11  R:  4.47  F1:  3.94
ROUGE-3:  P:  1.29  R:  1.46  F1:  1.25
ROUGE-4:  P:  0.47  R:  0.51  F1:  0.46
ROUGE-L:  P: 20.90  R: 24.55  F1: 21.14
ROUGE-W:  P: 11.70  R:  6.99  F1:  7.68
```

**Example extractive summary:**  
> “Hofmann ist Mitglied der Jury des Verbandes Liberaler Akademiker zur Vergabe des Arno-Esch-Preises. 1961 wurde Hofmann Persönlicher Referent von Walter Scheel...”  

**Reference summary:**  
> “Harald Hofmann ist ein deutscher Jurist und ehemaliger Diplomat, der u.a. Bundesgeschäftsführer der FDP und Botschafter in Dänemark, Venezuela, Norwegen und Schweden war.”  

---

## 🚀 Key Learning Outcomes  
- Implemented a full **NLP pipeline** (data preprocessing → topic modeling → sentence ranking → evaluation).  
- Gained practical experience with **LDA topic modeling** and **graph-based ranking (TextRank)**.  
- Learned limitations of extractive summarization (coherence, dependence on reference type) vs. abstractive approaches.  
