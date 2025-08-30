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

### Unseen Documents  
```text
ROUGE-1 F1: 20.99
ROUGE-L F1: 21.14
ROUGE-2/3/4: ~3–4 F1 (expected for higher n-grams)
```

### Seen Documents  
```text
ROUGE-1 F1: 19.96
ROUGE-L F1: 21.37
Similar trends to unseen data (robust generalization).
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
