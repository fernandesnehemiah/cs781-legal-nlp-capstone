# cs781-legal-nlp-capstone

# In-House Legal Intelligence System
### A Self-Hosted Legal NLP Framework for Indian Legal Documents

An end-to-end legal natural language processing (NLP) project developed as the **CS781 Capstone Project at the Indian Institute of Technology Kanpur**.

The project explores how a compact, domain-specific language-model stack can support multiple legal-document understanding tasks while keeping model training and inference under the system owner's control. It builds foundation language models from scratch on Indian legal text and adapts them to six downstream tasks: legal named entity recognition, rhetorical role prediction, court judgment prediction, legal statute identification, abstractive summarisation, and prior-case retrieval.

The system was developed under constrained compute, using a single Kaggle NVIDIA Tesla T4 GPU (16 GB). No externally pretrained language-model checkpoint or additional external legal corpus was used to train the foundation models.

---

## Contents

- [1. Project overview](#1-project-overview)
- [2. Problem statement and objectives](#2-problem-statement-and-objectives)
- [3. System architecture](#3-system-architecture)
- [4. Phase 1: Foundation language models](#4-phase-1-foundation-language-models)
- [5. Downstream tasks](#5-downstream-tasks)
  - [Phase 2: Legal Named Entity Recognition](#phase-2-legal-named-entity-recognition)
  - [Phase 3: Rhetorical Role Prediction](#phase-3-rhetorical-role-prediction)
  - [Phase 4: Court Judgment Prediction](#phase-4-court-judgment-prediction)
  - [Phase 5: Legal Statute Identification](#phase-5-legal-statute-identification)
  - [Phase 6: Abstractive Legal Summarisation](#phase-6-abstractive-legal-summarisation)
  - [Phase 7: Prior Case Retrieval](#phase-7-prior-case-retrieval)
- [6. Model adaptation and transfer learning](#6-model-adaptation-and-transfer-learning)
- [7. Data and preprocessing](#7-data-and-preprocessing)
- [8. Experimental setup](#8-experimental-setup)
- [9. Repository structure](#9-repository-structure)
- [10. Setup and execution](#10-setup-and-execution)
- [11. Outputs and artefacts](#11-outputs-and-artefacts)
- [12. Evaluation methodology](#12-evaluation-methodology)
- [13. Design observations](#13-design-observations)
- [14. Limitations and future work](#14-limitations-and-future-work)
- [15. References](#15-references)
- [16. Author](#16-author)

---

## 1. Project overview

Legal documents are long, information-dense, and structured around facts, arguments, statutes, precedents, issues, reasoning, and judicial decisions. Extracting and organising this information manually can be time-consuming. At the same time, legal documents may contain privileged or confidential information that should not be sent to third-party language-model APIs.

This project develops an **in-house legal intelligence framework** around language models trained specifically on Indian legal text. The foundation models are trained from scratch and then adapted to different legal NLP tasks.

The project is organised into seven phases:

| Phase | Task | Objective |
|---|---|---|
| 1 | Legal language-model pretraining | Train domain-specific foundation models from scratch |
| 2 | Legal Named Entity Recognition | Identify legal entities and their exact text spans |
| 3 | Rhetorical Role Prediction | Classify the rhetorical function of each sentence |
| 4 | Court Judgment Prediction | Predict whether an appeal is accepted or rejected |
| 5 | Legal Statute Identification | Identify relevant statutes from a case description |
| 6 | Abstractive Legal Summarisation | Generate summaries of long legal judgments |
| 7 | Prior Case Retrieval | Retrieve relevant earlier cases for a query judgment |

Each downstream phase is an independent task-specific system initialised from a Phase-1 checkpoint. The project does not form a sequential pipeline in which one task's predictions are automatically passed to the next task.

### Project objectives

- Build foundation language models without relying on externally pretrained checkpoints.
- Train custom byte-level BPE tokenizers suited to legal text.
- Compare different Transformer architecture families under limited compute.
- Reuse foundation checkpoints across multiple legal NLP tasks.
- Investigate parameter-efficient fine-tuning and knowledge distillation.
- Design task-specific architectures for classification, sequence labelling, generation, and retrieval.
- Evaluate models using the metrics specified for each task.
- Analyse architectural trade-offs, model errors, memory usage, and inference efficiency.

---

## 2. Problem statement and objectives

The main engineering challenge is to build a useful legal NLP stack with a relatively small training corpus and limited GPU resources.

The tasks have different output structures:

- **Token/span prediction:** Legal NER must identify entities and recover exact character offsets.
- **Sequence classification:** Rhetorical-role prediction must assign a role to each sentence while considering document context.
- **Document classification:** Judgment prediction must produce an accepted/rejected decision from a long judgment.
- **Multi-label classification:** Statute identification must predict several applicable statutes rather than exactly one class.
- **Conditional text generation:** Summarisation must generate a concise representation of a long source judgment.
- **Information retrieval:** Prior-case retrieval must rank candidate judgments by relevance.

A single generic classification head would not be suitable for all these tasks. The project therefore shares pretrained language representations while using task-specific heads, objectives, decoding strategies, and input-selection methods.

---

## 3. System architecture

### 3.1 Overall architecture

```text
                  Official Indian Legal Text Corpus
               Acts | High Courts | Supreme Court | Tribunals
                                  |
                                  v
                       Custom Byte-Level BPE
                         Tokenizer Training
                                  |
                                  v
                    PHASE 1: FOUNDATION TRAINING
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
          v                       v                       v
    Decoder-only             Transformer-XL          Encoder-Decoder
     RoPE + GQA              Segment Memory           RoPE + GQA
          |                       |                       |
          +-----------------------+-----------------------+
                                  |
                                  v
                     Multi-resolution Memory Decoder
                                  |
                                  v
                     Saved Checkpoints and Configs
                                  |
                    Reload checkpoint per downstream task
                                  |
           +----------------------+----------------------+
           |                      |                      |
           v                      v                      v
       PHASE 2                 PHASE 3                 PHASE 4
      Legal NER             Rhetorical Roles        Judgment Prediction
       BiLSTM-CRF            BiLSTM-CRF             Chunk Attention
           |                      |                      |
           +----------------------+----------------------+
                                  |
           +----------------------+----------------------+
           |                      |                      |
           v                      v                      v
       PHASE 5                 PHASE 6                 PHASE 7
    Statute Identification   Summarisation           Case Retrieval
    Multi-label Head         Text Generation         BM25 + Dense
```

The diagram summarises the model families and task-specific systems. In implementation, the downstream notebooks independently load the relevant Phase-1 model checkpoint, reconstruct its architecture, and train a task-specific descendant.

### 3.2 Transfer protocol

The common transfer workflow is:

1. Load the saved Phase-1 model configuration and checkpoint.
2. Reconstruct the same model architecture in PyTorch.
3. Load the pretrained weights and verify compatibility.
4. Attach the task-specific prediction or generation head.
5. Select an adaptation strategy, such as PEFT, partial fine-tuning, or distillation.
6. Train using the relevant task dataset and objective.
7. Select the best checkpoint using the development metric.
8. Evaluate and generate predictions in the task's required format.

The checkpoint configuration and tokenizer must remain consistent. A mismatch between tokenizer vocabulary IDs and the pretrained embedding matrix would cause the model to interpret token IDs incorrectly.

---

## 4. Phase 1: Foundation language models

### 4.1 Corpus

The foundation models were trained on Indian legal text from four sources:

- Acts
- High Court judgments
- Supreme Court judgments
- Tribunal documents

The project uses custom byte-level Byte Pair Encoding (BPE) tokenizers. The tokenizers are trained as part of the project rather than imported from an externally pretrained language model.

Two vocabulary sizes were used:

- **10,000 tokens:** decoder-only Transformer and Transformer-XL.
- **32,768 tokens:** encoder–decoder and multi-resolution memory model.

Byte-level tokenisation is useful for legal text because it can represent unusual punctuation, section symbols, citations, names, and text from different scripts without depending on a fixed external vocabulary.

### 4.2 Architecture families

Four foundation-model families were implemented and trained from scratch.

#### A. Decoder-only Transformer with RoPE and GQA

<details>
<summary><strong>Architecture details</strong></summary>

| Component | Configuration |
|---|---|
| Parameters | 85.65M |
| Transformer layers | 12 |
| Hidden dimension | 768 |
| Feed-forward dimension | 3072 |
| Query heads | 12 |
| Key/value heads | 6 |
| Attention | Grouped-Query Attention (GQA) |
| Position encoding | Rotary Position Embeddings (RoPE) |
| Normalisation | LayerNorm |
| FFN activation | GELU |
| Embeddings | Tied input/output embeddings |
| Vocabulary | 10,000 |
| Training context | 1,600 tokens |

The model uses causal self-attention for next-token prediction. RoPE encodes positional information through rotations applied to query and key representations. GQA allows multiple query heads to share key/value heads, reducing key/value computation and storage compared with standard multi-head attention.

</details>

#### B. Transformer-XL

Transformer-XL extends the usable context through segment-level recurrence. Hidden representations from previous segments are retained as memory for subsequent segments, with gradients detached through the memory. Relative positional information is used to support attention across segments.

| Component | Configuration |
|---|---|
| Parameters | 99.80M |
| Layers | 12 |
| Hidden dimension | 768 |
| Feed-forward dimension | 3072 |
| Attention heads | 12 |
| Position mechanism | Relative sinusoidal positions with learned biases |
| Context | 256-token segments with recurrent memory |

#### C. Encoder–Decoder Transformer

The encoder–decoder model is designed for conditional generation. The encoder processes the source document, while the decoder generates the continuation or summary while attending to the encoded source.

| Component | Configuration |
|---|---|
| Parameters | 148.22M |
| Encoder / decoder layers | 7 / 7 |
| Hidden dimension | 768 |
| Feed-forward dimension | 2752 |
| Attention | GQA in self- and cross-attention |
| Position encoding | RoPE |
| Normalisation | RMSNorm |
| FFN activation | SwiGLU |
| Additional component | Pointer-style copy head |
| Vocabulary | 32,768 |

The copy head combines a vocabulary-generation distribution with a distribution over source tokens obtained through cross-attention. This provides a mechanism for copying names, citations, and other source-specific expressions.

#### D. Multi-resolution memory decoder

This model retains recent context exactly while compressing older context into a smaller memory representation.

| Component | Configuration |
|---|---|
| Parameters | 103.09M |
| Layers | 12 |
| Hidden dimension | 768 |
| Feed-forward dimension | 3072 |
| Attention heads | 12 |
| Position encoding | RoPE |
| Normalisation | RMSNorm |
| Recent context | 256 exact tokens |
| Compressed memory | Up to 2,048 older tokens compressed into 256 slots |

Older tokens are grouped and mean-pooled into memory slots. The design aims to preserve useful historical context without attending to every previous token individually.

### 4.3 Training objective

The decoder-based foundation models use next-token prediction with cross-entropy loss:

\[
\mathcal{L}_{LM}
=
-\sum_{t=1}^{T}
\log P(x_t\mid x_{<t})
\]

The encoder–decoder model is trained on prompt/continuation pairs so that its training objective resembles the conditional-generation task.

The training pipeline includes:

- Custom BPE tokenisation.
- Source-aware corpus sampling.
- Train/validation separation.
- AdamW optimisation.
- Learning-rate warmup followed by cosine decay.
- Mixed-precision training.
- Gradient clipping.
- Checkpoint selection using validation loss and generation evaluation.

---

## 5. Downstream tasks

### Phase 2: Legal Named Entity Recognition

**Objective:** identify legal entities in judgments and return their exact character-offset spans.

The task contains 12 entity types, including:

- APP
- RESP
- A.COUNSEL
- R.COUNSEL
- JUDGE
- WIT
- AUTH
- COURT
- STAT
- PREC
- DATE
- CASENO

#### Architecture

```text
Raw legal document
       |
       v
Punctuation-aware pre-tokenisation
       |
       v
Byte-level BPE tokenisation + character offsets
       |
       v
Pretrained decoder-only Transformer
       |
       v
Token representations
       |
       v
BiLSTM
       |
       v
BIO classification head
       |
       v
CRF + Viterbi decoding
       |
       v
Character-offset entity spans
```

#### Methodology

1. Split the original document into punctuation-aware pieces while preserving character offsets.
2. Encode the pieces using the Phase-1 tokenizer.
3. Assign BIO tags to the appropriate subword positions.
4. Process long documents using overlapping 512-token windows.
5. Apply a BiLSTM and CRF to model contextual label dependencies.
6. Use Viterbi decoding to obtain valid BIO sequences.
7. Map predicted tags back to original character offsets.

The offset-preserving pre-tokenisation is important because even small changes to whitespace or punctuation can shift predicted entity boundaries. The reported round-trip check retained 98.05% of gold spans exactly with punctuation-aware pre-tokenisation.

The notebook also performs task-adaptive pretraining on legal documents before supervised NER training.

### Phase 3: Rhetorical Role Prediction

**Objective:** assign one rhetorical-role label to each sentence in a legal judgment.

The task uses 13 roles, including `Fact`, `Issue`, `ArgumentPetitioner`, `ArgumentRespondent`, `Statute`, `Dissent`, precedent-related roles, court-ruling roles, `RatioOfTheDecision`, and `None`.

#### Architecture

```text
Legal judgment as ordered sentences
              |
              v
Phase-1 tokenizer + decoder
              |
              v
Token hidden states
              |
              v
Mean pooling + last-token pooling
              |
              v
Domain embedding (CL / IT)
              |
              v
Sentence projection
              |
              v
Document-level BiLSTM
              |
              v
Linear role classifier
              |
              v
CRF + Viterbi decoding
              |
              v
One rhetorical role per sentence
```

#### Methodology

- Whole sentences are packed into 512-token windows so that neighbouring sentences are available to the Transformer.
- The mean and last hidden states of each sentence's tokens are concatenated to form its sentence representation.
- A learned domain embedding indicates whether the judgment belongs to Corporate Law (CL) or IT law.
- A document-level BiLSTM provides left and right sentence context.
- A CRF models dependencies between consecutive rhetorical roles.
- Viterbi decoding produces the final sequence of sentence labels.

The BiLSTM is particularly useful because a decoder-only Transformer has causal token attention, while rhetorical function can depend on sentences that appear later in the judgment.

### Phase 4: Court Judgment Prediction

**Objective:** predict whether a Supreme Court appeal is accepted or rejected.

#### Architecture

```text
Long judgment
     |
     v
Select four 512-token chunks (head + tail)
     |
     v
Shared pretrained Transformer encoder
     |
     v
Attention pooling within each chunk
     |
     v
Relative-position embeddings
     |
     v
Attention pooling across chunk vectors
     |
     v
Document representation
     |
     v
Binary classifier
     |
     v
Accepted / Rejected
```

#### Methodology

Long judgments cannot be processed in their entirety within the available context budget. The notebook selects the first chunk and the last three chunks, capturing introductory context and the operative portion near the end.

Each chunk is encoded independently. Attention pooling creates a chunk representation, and a second attention layer aggregates the chunk representations into a document vector.

The final classifier predicts the binary outcome. The decision threshold is tuned on development data for Macro-F1, and temperature scaling is used for probability calibration.

### Phase 5: Legal Statute Identification

**Objective:** identify applicable statutes from a fact description.

This is a 100-label multi-label classification problem. A document can be associated with multiple statutes.

#### Architecture

```text
Fact description
       |
       v
Pretrained Transformer
       |
       v
Document pooling
       |
       v
Document representation
       |
       v
100 sigmoid outputs
       |
       v
Label-correlation layer
       |
       v
Per-label thresholding
       |
       v
Predicted statute set
```

#### Methodology

- The model generates one sigmoid probability for each of the 100 statute labels.
- A label-correlation layer captures relationships between statutes that frequently co-occur.
- Asymmetric Loss (ASL) reduces the contribution of easy negative examples, which are common in multi-label classification.
- Sampling is adjusted to expose the model more often to documents containing rare statutes.
- Per-label thresholds are tuned using two-fold cross-tuning on development data to reduce threshold overfitting.

### Phase 6: Abstractive Legal Summarisation

**Objective:** generate a concise natural-language summary of a long legal judgment.

The summarisation approach uses the generation capabilities of the pretrained foundation models and adapts them using supervised source-summary pairs.

#### Methodology

1. Analyse the source judgment and identify salient sentences.
2. Select salient content using TF–IDF overlap with the full document.
3. Preserve the original order of selected sentences.
4. Construct a source input within the architecture's context budget.
5. Train using teacher forcing against reference summaries.
6. Generate summaries using the selected decoding strategy.

The source representation varies by architecture:

- Decoder-only and Transformer-XL models use a source-plus-prompt continuation setup.
- The encoder–decoder processes the source in its encoder and generates the summary with its decoder.
- The multi-resolution model represents recent tokens exactly and compresses older source content into memory slots.

The reported generation configuration uses a summary cap of 384 tokens, greedy decoding, a no-repeat 3-gram constraint, and a repetition penalty.

### Phase 7: Prior Case Retrieval

**Objective:** retrieve the seven most relevant prior cases for a query judgment.

The retrieval system combines sparse lexical matching with dense semantic representations.

#### A. BM25 sparse retrieval

BM25 ranks documents using term-frequency, inverse-document-frequency, and document-length normalisation.

The implementation uses hashed unigrams and bigrams to avoid storing a large explicit vocabulary dictionary. The BM25 parameters are tuned on development queries.

#### B. Dense bi-encoder retrieval

A shared Transformer encoder represents query and candidate judgments. Long cases are divided into up to eight 512-token chunks, which are encoded and pooled into dense vectors.

The notebook explores:

- Cosine similarity between averaged case representations.
- ColBERT-style MaxSim interaction over chunk vectors.
- Contrastive training using InfoNCE.
- In-batch negatives and BM25 hard negatives.
- A learned temperature parameter.

#### C. Hybrid retrieval

The sparse and dense scores are normalised and combined:

\[
s(q,d)=w\,z_{\text{dense}}(q,d)
+(1-w)\,z_{\text{BM25}}(q,d)
\]

The mixing weight is selected using development data. Reciprocal Rank Fusion (RRF) is also implemented as an alternative combination method.

---

## 6. Model adaptation and transfer learning

The project compares several strategies for adapting foundation models to downstream tasks. The objective is to reuse useful pretrained representations while controlling compute, memory, and the number of trainable parameters.

### 6.1 Full fine-tuning

All model parameters are updated using the downstream task objective. This provides maximum flexibility but requires more memory and can overfit when labelled data is limited.

### 6.2 Low-Rank Adaptation (LoRA)

LoRA freezes the original weight matrix \(W_0\) and learns a low-rank update:

\[
W=W_0+\frac{\alpha}{r}BA
\]

where \(A\) and \(B\) are trainable low-rank matrices, \(r\) is the rank, and \(\alpha\) controls the update scale.

### 6.3 Rank-Stabilised LoRA (rsLoRA)

rsLoRA uses a rank-stabilised scaling factor:

\[
\frac{\alpha}{\sqrt r}
\]

This changes how the update magnitude scales with the adapter rank.

### 6.4 Multiplicative LoRA (LoRMA+)

LoRMA+ applies a multiplicative low-rank transformation:

\[
W=\left(I+\frac{\alpha}{r}BA\right)W_0
\]

With \(B\) initialised to zero, the adapter initially preserves the original pretrained transformation.

### 6.5 Other adaptation methods

| Method | Implementation idea |
|---|---|
| AdaLoRA | Adaptively allocates a low-rank parameter budget |
| QLoRA | Keeps the base model in 4-bit NF4 and trains low-rank adapters |
| IA³ | Learns scaling vectors for internal activations |
| Houlsby / Pfeiffer adapters | Adds small trainable bottleneck modules |
| Prefix tuning | Learns trainable prefix representations for attention |
| BitFit | Updates bias parameters |
| Partial fine-tuning | Unfreezes selected upper layers |
| Knowledge distillation | Transfers teacher predictions and representations to a smaller student |

### 6.6 Knowledge distillation

A teacher model is used to train a smaller student model. The student learns from the teacher's softened output distribution in addition to the ground-truth labels. Selected hidden representations can also be matched.

A typical combined objective is:

\[
\mathcal{L}
=
\lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KL}}
+
\lambda_{\mathrm{hard}}\mathcal{L}_{\mathrm{task}}
+
\lambda_{\mathrm{hidden}}\mathcal{L}_{\mathrm{hidden}}
\]

The exact loss components and coefficients are defined in the corresponding phase notebook.

The project uses distillation to investigate whether a smaller model can retain the useful task-specific behaviour of a larger adapted model while reducing inference cost.

---

## 7. Data and preprocessing

### 7.1 Dataset source

The project uses the official CS781 capstone dataset:

**Hugging Face dataset:** [`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)

The dataset contains phase-specific data and official train, development, and test splits. The notebooks load only the data required for their respective tasks.

The dataset is not included in this repository. Users should consult the dataset card and competition instructions for access requirements and usage conditions.

### 7.2 Dataset characteristics

The project report records the following data characteristics:

| Phase | Data characteristics |
|---|---|
| Phase 1 | Acts, High Courts, Supreme Court, and Tribunal text |
| Phase 2 | 60 labelled training documents, a 10-document local development split, and 35 test documents |
| Phase 3 | Separate Corporate Law and IT sub-corpora; 80 train, 10 dev, and 10 test documents in total |
| Phase 4 | 37,387 training, 1,988 development, and 1,517 test examples |
| Phase 5 | 42,750 training, 10,181 development, and 13,019 test examples; 100 statute labels |
| Phase 6 | 6,880 training, 150 development, and 100 test examples |
| Phase 7 | 827 training, 118 development, and 237 test queries |

### 7.3 Tokenisation

All models use custom byte-level BPE tokenizers developed for the project.

- Text is not lowercased.
- The decoder and Transformer-XL use a 10k vocabulary.
- The encoder–decoder and multi-resolution model use a 32k vocabulary.
- The tokenizer configuration is saved with the corresponding model checkpoint.

### 7.4 Long-document handling

The context length of a Transformer is limited, while legal judgments can be very long. The project uses task-specific document handling:

| Phase | Long-document strategy |
|---|---|
| Phase 2 | Overlapping token windows with offset reconstruction |
| Phase 3 | Whole-sentence packing into token windows and sentence-level chunks |
| Phase 4 | Four selected chunks using a head-and-tail strategy |
| Phase 5 | Evenly spread document chunks |
| Phase 6 | Salient-sentence selection under a source-token budget |
| Phase 7 | Up to eight chunks per case for dense retrieval |

These strategies preserve relevant document information while keeping computation within the available context and GPU limits.

---

## 8. Experimental setup

### 8.1 Hardware

The experiments were conducted on:

- **Platform:** Kaggle Notebooks
- **GPU:** NVIDIA Tesla T4
- **GPU memory:** 16 GB
- **Precision:** FP16 mixed precision where supported

The project was designed to fit within a single-GPU environment.

### 8.2 Training configuration

The notebooks use the following techniques where appropriate:

- PyTorch model implementations.
- AdamW optimisation.
- Linear learning-rate warmup followed by cosine decay.
- Mixed-precision training.
- Gradient clipping (maximum norm 1.0).
- Gradient accumulation for larger effective batches.
- Checkpoint saving after epochs.
- Early stopping based on development performance.
- Experiment registries to resume interrupted runs.

Phase-1 pretraining runs took approximately 6–9 hours each in the reported environment. Downstream experiments were generally constrained to approximately 35–60 minutes per experiment. Actual runtime varies with the selected configuration and hardware availability.

### 8.3 Reproducibility

The notebooks set random seeds (42 in the reported setup), save model configurations and experiment results, and retain checkpoints for resuming runs. GPU operations may still introduce some nondeterminism.

The project report provides the phase-wise hyperparameters and experiment comparisons. The notebook configuration cells are the authoritative place to inspect or change the settings used for a particular run.

---

## 9. Repository structure

```text
.
├── README.md
├── report/
│   └── CS781-CP-Nehemiah-Fernandes-261110040-report.pdf
└── code/
    ├── phase1/
    │   └── phase1_decoder_only_rope_gqa.ipynb
    ├── phase2/
    │   └── phase2_decoder_only_lner.ipynb
    ├── phase3/
    │   └── phase3_decoder_only_rr.ipynb
    ├── phase4/
    │   └── phase4_decoder_only_cjpe.ipynb
    ├── phase5/
    │   └── phase5_decoder_only_lsi.ipynb
    ├── phase6/
    │   └── phase6_decoder_only_summ.ipynb
    └── phase7/
        └── phase7_decoder_only_pcr.ipynb
```

### Notebook guide

| Notebook | Main contents |
|---|---|
| `phase1_decoder_only_rope_gqa.ipynb` | Decoder-only foundation model, tokenizer/checkpoint setup, pretraining, validation and generation |
| `phase2_decoder_only_lner.ipynb` | Legal NER preprocessing, offset alignment, BiLSTM-CRF, adaptation, evaluation and prediction |
| `phase3_decoder_only_rr.ipynb` | Sentence representations, domain embeddings, document BiLSTM-CRF and rhetorical-role evaluation |
| `phase4_decoder_only_cjpe.ipynb` | Long-document chunk encoding, attention pooling, binary classification and calibration |
| `phase5_decoder_only_lsi.ipynb` | Multi-label statute prediction, asymmetric loss, label correlations and threshold tuning |
| `phase6_decoder_only_summ.ipynb` | Salient source selection, conditional generation, fine-tuning and summary evaluation |
| `phase7_decoder_only_pcr.ipynb` | BM25, dense bi-encoder training, hybrid ranking and retrieval evaluation |

The PDF report provides the consolidated methodology, experiment analysis, results, limitations, and bibliography.

---

## 10. Setup and execution

The notebooks were developed for Kaggle GPU notebooks. They can be adapted to another compatible environment, but paths, data access, and available GPU memory may need to be changed.

### 10.1 Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 10.2 Open the notebooks

Open the relevant `.ipynb` file from the `code/` directory in Kaggle or Jupyter.

For the intended Kaggle setup:

1. Create a Kaggle notebook.
2. Enable a GPU accelerator.
3. Import or upload the relevant project notebook.
4. Attach the official capstone dataset or configure the notebook's dataset download.
5. Run the notebook's setup and configuration cells.

### 10.3 Configure dataset access

The notebooks use the official dataset ID:

```python
DATASET_ID = "Exploration-Lab/CS-781-Capstone"
```

If access requires authentication, configure a Hugging Face access token through Kaggle Secrets or an equivalent secure environment variable. Do not hard-code credentials or commit access tokens to the repository.

Some notebooks may also support local dataset paths. In that case, attach the relevant official phase files as notebook input data and update the path in the configuration cell.

### 10.4 Run Phase 1

Start with:

```text
code/phase1/phase1_decoder_only_rope_gqa.ipynb
```

This notebook trains the decoder-only RoPE+GQA foundation model and creates the checkpoint and tokenizer artefacts required by the downstream decoder-based notebooks.

The downstream notebooks expect a compatible Phase-1 checkpoint, model configuration, and tokenizer. Make these outputs available as Kaggle notebook inputs or place them in the locations specified by the downstream notebook's checkpoint-discovery code.

> The repository archive contains the Phase-1 decoder-only notebook. The other three Phase-1 architecture families are documented in the report, but their separate training notebooks are not part of this repository snapshot.

### 10.5 Run a downstream task

After the required Phase-1 artefacts are available, select the corresponding notebook:

```text
code/phase2/phase2_decoder_only_lner.ipynb
code/phase3/phase3_decoder_only_rr.ipynb
code/phase4/phase4_decoder_only_cjpe.ipynb
code/phase5/phase5_decoder_only_lsi.ipynb
code/phase6/phase6_decoder_only_summ.ipynb
code/phase7/phase7_decoder_only_pcr.ipynb
```

Each notebook contains its own:

1. Configuration and experiment definitions.
2. Dataset loading and preprocessing.
3. Phase-1 model reconstruction and checkpoint loading.
4. Task-specific architecture and adaptation method.
5. Training and validation loop.
6. Evaluation and error analysis.
7. Prediction generation.

The downstream tasks are independently runnable once their required data and checkpoint artefacts are available. They do not need to be run in numerical order after Phase 1.

### 10.6 Select experiments

Each notebook defines experiment configurations and a run list. Inspect these cells before starting training.

Depending on the phase, the notebook may include full fine-tuning, PEFT variants, partial fine-tuning, or distillation. Some experiments depend on a previously trained teacher model. Run the required teacher experiment first if the selected configuration uses distillation.

The notebooks also save experiment state and may skip completed runs when corresponding result files are present. If you modify a configuration, use a clean output directory or update the run identifier to avoid reusing stale checkpoints.

### 10.7 Runtime and storage considerations

- Use a GPU accelerator; CPU-only execution will be substantially slower.
- Monitor GPU memory when changing batch size, context length, or model architecture.
- Use gradient accumulation rather than increasing the microbatch beyond available memory.
- Preserve checkpoints and tokenizer files between Kaggle sessions.
- Kaggle working storage is temporary; download or save important artefacts as notebook outputs before the session ends.

---

## 11. Outputs and artefacts

Depending on the phase and experiment, notebooks generate:

- Trained model checkpoints (`.pt` / `.pth`).
- Tokenizer JSON files.
- Saved model configuration.
- Experiment registry and result JSON files.
- Development predictions and evaluation metrics.
- Training curves and diagnostic plots.
- Task-specific prediction files.
- Packaged prediction outputs for external evaluation.

Exact filenames and output directories are defined in each notebook's configuration and final execution cells.

The uploaded source repository contains the report and notebooks. It does **not** bundle the full training corpus, trained checkpoints, tokenizer artefacts, cached predictions, or generated model outputs. These must be obtained or produced by following the notebook workflow.

---

## 12. Evaluation methodology

Each task is evaluated using its designated primary metric.

| Phase | Primary metric | Evaluation purpose |
|---|---|---|
| 1. Language-model pretraining | BERTScore-F1 | Semantic similarity of generated continuations |
| 2. Legal NER | Strict span macro-F1 | Exact entity type and character-boundary prediction |
| 3. Rhetorical roles | Macro-F1 | Balanced performance across rhetorical-role classes |
| 4. Judgment prediction | Macro-F1 | Performance on accepted/rejected classification |
| 5. Statute identification | Macro-F1 | Multi-label statute prediction quality |
| 6. Summarisation | BERTScore-F1 | Semantic similarity between generated and reference summaries |
| 7. Prior-case retrieval | Micro-F1@7 | Relevance of the seven retrieved cases |

Macro-F1 is useful for imbalanced classification because it gives each class equal weight in the final average. Strict span-F1 for NER requires exact boundary and label agreement. Retrieval metrics evaluate the set of relevant cases returned among the top seven results.

### Important evaluation distinction

The report contains both local development results and official benchmark results. These are not always directly comparable: some local validation sets are small, and some local model-selection metrics differ from the official metric. The final report should be consulted for the exact evaluation split and metric used in each phase.

---

## 13. Design observations

Several practical observations emerged from the experiments:

- **Data diversity matters:** increasing model width without a corresponding increase in unique training data produced only a small improvement in the initial decoder experiments.
- **Context selection matters:** long legal judgments require task-aware chunk selection or compression because the complete document often exceeds the model's context window.
- **Task-specific heads matter:** token tagging, sentence sequence labelling, multi-label prediction, generation, and retrieval require different output architectures and objectives.
- **PEFT can be competitive:** low-rank and activation-scaling methods can adapt a foundation model without updating every pretrained parameter.
- **Distillation is useful for deployment trade-offs:** a smaller student can retain strong task performance while improving inference throughput.
- **Classical retrieval remains valuable:** BM25 was a strong baseline and an important component of the hybrid prior-case retrieval system.
- **Metrics need interpretation:** semantic similarity metrics do not establish legal correctness, and exact span metrics may penalise boundary differences that a human reader might consider minor.

For the phase-wise results, ablations, error analysis, and detailed discussion, see the project report.

---

## 14. Limitations and future work

The project is an experimental academic system and has several limitations:

1. **Compute constraints:** all experiments were designed around a single 16 GB T4 GPU, limiting model size, batch size, context length, and training duration.
2. **Limited training data:** the legal corpus is relatively small for foundation-model pretraining compared with large general-purpose language models.
3. **Long-document coverage:** several tasks process selected chunks or salient sentences rather than every token in the full judgment.
4. **Small development sets in some phases:** local validation estimates can be sensitive to individual documents.
5. **Rare labels:** some legal entity and rhetorical-role categories have limited examples, making robust prediction difficult.
6. **Retrieval quality:** dense-only retrieval did not outperform BM25 in the reported experiments.
7. **Legal validation:** the system was evaluated using benchmark metrics, not by a formal review process conducted by practising legal professionals.

Potential next steps include training memory-based models on more of the available corpus, adding right-context information to causal NER representations, improving rare-label learning, refining long-document selection, and training dense retrieval specifically on cases missed by BM25.

**This project is not a substitute for legal advice or professional legal review.** Predictions and generated summaries should be checked by a qualified human before being used in legal or operational decisions.

---

## 15. References

The complete bibliography is provided in the project report. Key foundations include:

- Joshi et al. (2024), *IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning*.
- Vaswani et al. (2017), *Attention Is All You Need*.
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*.
- Dai et al. (2019), *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*.
- Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*.
- Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*.
- Hinton et al. (2015), *Distilling the Knowledge in a Neural Network*.
- Lafferty et al. (2001), *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*.
- Robertson and Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*.

For the full set of references, see:

`report/CS781-CP-Nehemiah-Fernandes-261110040-report.pdf`

---

## 16. Author

**Nehemiah Fernandes**  
CS781 Capstone Project  
Indian Institute of Technology Kanpur

---

## Disclaimer

This repository is an academic research and engineering project. It is intended for experimentation and educational use. It does not provide legal advice and should not be used as an autonomous legal decision-making system.
