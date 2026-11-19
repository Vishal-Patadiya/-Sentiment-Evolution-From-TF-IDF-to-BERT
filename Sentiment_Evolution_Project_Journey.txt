SENTIMENT EVOLUTION: FROM TF-IDF TO BERT A PROJECT JOURNEY, ENGINEERING
LOG, AND LEARNING DOCUMENTATION

Author: Vishal Patadiya Project: IMDb Sentiment Analysis Repository:
Sentiment-Evolution-From-TF-IDF-to-BERT

====================================================================== 1.
WHY THIS PROJECT EXISTS
======================================================================

This project started as an IMDb sentiment-analysis exercise, but it
deliberately grew into something larger: a practical journey through the
evolution of Natural Language Processing models.

The original objective was not simply to obtain the highest possible
accuracy. The objective was to understand why NLP systems evolved from
sparse, manually engineered representations toward neural sequence
models, attention mechanisms, Transformers, and finally pretrained
language models such as BERT.

The progression used in this project was:

    TF-IDF
        ↓
    Logistic Regression
        ↓
    Dense Neural Network
        ↓
    LSTM
        ↓
    Transformer Encoder
        ↓
    BERT

Each stage was treated as an experiment rather than as a replacement for
the previous model. The central question was:

    "What limitation did we encounter, and what does the next architecture give us that the previous one did not?"

That question shaped the entire notebook.

The project was also intentionally engineering-oriented. Whenever
something failed, became too slow, consumed too much memory, produced an
unexpected result, or behaved differently from what was expected, the
problem was investigated instead of simply being hidden or bypassed.

This document records that process.

It is not intended to pretend that every experiment was perfect. In
fact, some of the most useful lessons came from experiments that did not
improve the score.

======================================================================
2. THE ORIGINAL PLAN: A LOCAL VS CODE + JUPYTER NOTEBOOK PROJECT
======================================================================

The project was initially designed to run locally in a VS Code Jupyter
Notebook environment.

The working environment was a local Python/Conda setup with the typical
data-science stack: Python, Jupyter, Pandas, NumPy, scikit-learn,
TensorFlow/Keras, and later PyTorch and Hugging Face Transformers.

The original intention was straightforward:

    load IMDb
        ↓
    explore the dataset
        ↓
    build a classical NLP baseline
        ↓
    gradually introduce neural architectures
        ↓
    finish with BERT
        ↓
    compare the models

At the beginning, there was no reason to assume that the project would
need a cloud GPU.

The early phases were perfectly reasonable on a normal laptop. TF-IDF,
Logistic Regression, and even the first neural-network experiments were
manageable.

That assumption changed when we reached BERT.

The project therefore became an exercise in not only machine learning,
but also in deciding when local computing was no longer the right
engineering environment.

======================================================================
3. DATASET: IMDb MOVIE REVIEWS
======================================================================

The project uses the IMDb movie-review dataset.

The labeled training data contains:

    25,000 total reviews
    12,500 positive reviews
    12,500 negative reviews

The classes are perfectly balanced:

    Positive: 50%
    Negative: 50%

The raw dataset was organized into positive and negative text files. The
notebook loaded those files and constructed a Pandas DataFrame
containing two columns:

    review
    sentiment

where:

    1 = positive
    0 = negative

The project deliberately worked with the raw review text rather than
relying on a prebuilt numerical representation from the beginning. This
allowed the representation-learning journey to remain visible.

The raw dataset archive was obtained through a Hugging Face mirror after
the original direct-download approach was not practical. The archive was
extracted into the standard “aclImdb” directory structure.

The notebook also inspected the actual directory structure and opened
example positive and negative reviews before beginning model training.

This was important because the project was being built as a learning
exercise. Understanding what the data actually looked like came before
blindly feeding it into a model.

======================================================================
4. DATA LOADING AND DATASET CONSTRUCTION
======================================================================

The raw reviews were loaded by iterating through the “pos” and “neg”
directories.

Each text file was opened with UTF-8 encoding and appended to a list of
reviews. The corresponding sentiment label was appended to a second
list.

The two lists were then converted into a Pandas DataFrame.

The resulting dataset was:

    Shape: (25000, 2)

Class distribution:

    Positive: 12500
    Negative: 12500

This confirmed that the dataset was balanced.

One important lesson here was that machine-learning work often begins
with mundane engineering work: locating files, verifying directory
structures, reading text correctly, checking labels, and confirming that
the resulting dataset actually matches expectations.

======================================================================
5. THE TRAIN/VALIDATION SPLIT
======================================================================

For the main experiments, the labeled data was divided into:

    Training:   17,500 reviews
    Validation:  7,500 reviews

The split used:

    test_size = 0.30
    random_state = 42
    stratify = sentiment

Stratification was important because the dataset is balanced and we
wanted the training and validation subsets to preserve that balance.

The split was performed at the raw-text level for the main NLP
experiments.

This became particularly important when TF-IDF was introduced.

======================================================================
6. AN IMPORTANT LESSON ABOUT DATA LEAKAGE
======================================================================

One of the most valuable engineering lessons in the project was
understanding that a vectorizer is itself something that learns from
data.

TF-IDF is not just a passive conversion function. When we call:

    vectorizer.fit_transform(training_text)

the vectorizer learns vocabulary and document-frequency statistics from
the training corpus.

Therefore, the correct pipeline is:

    raw training text
        ↓
    fit TF-IDF
        ↓
    transform training text

and:

    raw validation text
        ↓
    transform using the already-fitted TF-IDF
        ↓
    validation features

The validation data must not be used to fit the vectorizer.

During the early exploration, TF-IDF was also applied to the full
DataFrame as a learning demonstration. We then deliberately
reconstructed the correct pipeline using the raw text split and fitted
TF-IDF only on the training text.

The final modeling workflow therefore used:

    train_text
        ↓
    vectorizer.fit_transform(train_text)

and:

    val_text
        ↓
    vectorizer.transform(val_text)

This was not just a coding correction. It was a conceptual correction.

A model evaluation is only meaningful if information from the validation
set does not leak into the training process.

======================================================================
7. PHASE 1 — TF-IDF
======================================================================

The first representation was TF-IDF.

TF-IDF converts text into a sparse numerical representation where each
feature corresponds to a vocabulary term and the value represents the
importance of that term in a document relative to the corpus.

The first exploratory TF-IDF representation produced approximately:

    74,849 features

After rebuilding the correct train/validation pipeline, the training
matrix contained:

    17,500 reviews
    64,881 features

The validation matrix contained:

    7,500 reviews
    64,881 features

This was the first point where the project made the distinction between:

    "Which column represents this word?"

and:

    "How much does this word matter in this document?"

The vocabulary mapping answers the first question.

The TF-IDF value answers the second.

The representation is extremely high-dimensional but sparse.

For the training matrix:

    Shape: (17500, 64881)
    Non-zero values: 2,411,664

The theoretical number of cells is enormous, but most of them are zero.

This led to another important engineering lesson:

    Sparse data should remain sparse when possible.

Converting this matrix into a dense array would have created a huge
memory burden for no useful reason.

======================================================================
8. PHASE 2 — LOGISTIC REGRESSION
======================================================================

The first actual sentiment classifier was Logistic Regression.

The model was:

    LogisticRegression(max_iter=1000)

It was trained directly on the sparse TF-IDF representation.

Validation accuracy:

    87.95%

This was a strong baseline.

More importantly, Logistic Regression gave us something interpretable.

Because the model learns a coefficient for each TF-IDF feature, we could
inspect which words were associated with positive and negative
predictions.

Strong positive-associated terms included examples such as:

    great
    excellent
    best
    perfect
    wonderful
    amazing
    fun
    loved
    love

Strong negative-associated terms included examples such as:

    worst
    bad
    awful
    waste
    boring
    poor
    terrible
    worse

This gave us an intuitive view of what the model had learned.

The model was not “understanding movies.”

It was learning statistical associations between words and sentiment
labels.

======================================================================
9. ERROR ANALYSIS: WHY 87.95% IS NOT THE END
======================================================================

The baseline produced approximately 904 validation mistakes.

Instead of immediately changing the model, we inspected mistakes.

One of the observed failures involved a positive review containing mixed
or context-dependent language that could be interpreted negatively by a
bag-of-words system.

This revealed a fundamental limitation.

TF-IDF represents a document largely as a collection of weighted terms.
It does not naturally understand the full compositional structure of
language.

For example:

    "not good"

contains the word “good,” but the meaning of the phrase is negative.

Likewise:

    "I expected much more, but unfortunately this film was disappointing."

requires understanding how the sentence is composed rather than simply
counting sentiment-associated words.

Sarcasm, negation, contrast, word order, and long-range relationships
are difficult for a basic bag-of-words representation.

This led to the first major project principle:

    We do not upgrade the model because deep learning is "cool."
    We upgrade the model because we identified a limitation.

======================================================================
10. LOGISTIC REGRESSION COEFFICIENT ANALYSIS
======================================================================

The coefficient analysis was one of the more educational parts of the
baseline.

The model coefficients were mapped back to the vocabulary so that
individual words could be examined.

This showed that the model was learning useful sentiment signals, but it
also exposed weaknesses.

Some highly weighted terms were not inherently positive or negative.

Examples included words such as:

    it
    and
    today
    also

These words can receive strong coefficients because of correlations and
dataset-specific patterns rather than because the words themselves carry
universal sentiment.

This demonstrated an important machine-learning idea:

    A model can discover predictive correlations without understanding why those correlations exist.

======================================================================
11. BIGRAM EXPERIMENT
======================================================================

The next experiment tested a specific hypothesis.

Hypothesis:

    If single words lose local context, perhaps adding word pairs
    will help the model understand phrases such as "not good."

The TF-IDF representation was therefore expanded from unigrams to:

    ngram_range = (1, 2)

This increased the feature space dramatically.

Feature dimensionality increased from approximately:

    64,881

to approximately:

    1,171,987

However, validation accuracy did not improve.

Results:

    Unigram TF-IDF + Logistic Regression: 87.95%
    Unigram + Bigram TF-IDF:              87.83%

The model became more complex in representation but slightly worse in
validation accuracy.

This became one of the most important lessons in the project:

    More features does not automatically mean more understanding.

The bigram experiment also taught us that adding local context is not
equivalent to building a model capable of understanding context.

The project therefore moved forward.

======================================================================
12. WHY A DENSE NEURAL NETWORK?
======================================================================

The next question was:

    Can a neural network learn a more flexible decision boundary
    over the TF-IDF representation?

The TF-IDF representation remained sparse, but Keras was configured to
accept a sparse input.

The first dense architecture was conceptually simple:

    Sparse TF-IDF input
        ↓
    Dense(128, ReLU)
        ↓
    Dense(1, Sigmoid)

The model contained approximately:

    8.3 million parameters

This was a useful moment for understanding what a neural network was
actually doing.

The first layer had to connect tens of thousands of TF-IDF features to
128 hidden units.

That immediately explained why the parameter count became so large.

The model was learning combinations of lexical signals rather than
simply applying one coefficient per feature.

======================================================================
13. DENSE NEURAL NETWORK RESULTS
======================================================================

The dense neural network was trained for five epochs.

A key observation was the difference between training and validation
performance.

The model eventually approached almost perfect training accuracy while
validation accuracy stopped improving.

The best validation accuracy was:

    89.92%

This was better than the Logistic Regression baseline:

    87.95%

However, the training curve showed clear overfitting.

This was an important turning point.

The model had more capacity and could fit the training data extremely
well, but generalization did not improve proportionally.

The lesson was:

    Increasing model capacity can improve training performance
    without improving performance on unseen data.

======================================================================
14. DROPOUT EXPERIMENT
======================================================================

Because overfitting was visible, we tested a common regularization
technique: Dropout.

The revised architecture inserted:

    Dropout(0.5)

between the hidden layer and output layer.

The hypothesis was:

    Randomly dropping hidden activations during training may
    reduce over-reliance on particular features and improve
    generalization.

The result was educational rather than spectacular.

Training behavior changed, but validation performance remained around
the same region.

The recorded comparison was approximately:

    Original Dense NN
        Training → ~100%
        Validation → ~88.84% at the end
        Best validation → 89.92%

    Dense NN + Dropout
        Training → ~98%
        Validation → ~88.9%
        Best validation → ~89.9%

The important conclusion was:

    Dropout did not meaningfully improve this particular
    TF-IDF + Dense architecture.

That is still a successful experiment.

A machine-learning experiment does not have to produce a better score to
be valuable.

It only needs to answer a question.

======================================================================
15. THE TRANSITION FROM WORD FEATURES TO SEQUENCES
======================================================================

At this point, the project had explored a major limitation of TF-IDF:

    It represents text as features, not as a sequence of language.

The next architecture therefore needed to preserve word order.

This led to the LSTM.

Before building the LSTM, the text had to be converted into token IDs.

Keras TextVectorization was used rather than relying on older tokenizer
workflows.

The vocabulary was limited to:

    10,000 tokens

Each review was converted into a sequence of integer token IDs.

The raw review lengths were examined before selecting a fixed sequence
length.

Observed statistics included approximately:

    Minimum length: 10
    Maximum length: 2470
    Mean length:    233.8
    Median length:  174

Several sequence-length thresholds were tested conceptually.

A maximum length of:

    500 tokens

was selected.

Only about:

    8.07%

of reviews were longer than 500 words/tokens in the inspected length
distribution.

This was a practical tradeoff between preserving useful context and
controlling computational cost.

======================================================================
16. EMBEDDINGS
======================================================================

The token IDs were passed into an embedding layer.

The embedding configuration used:

    Vocabulary size: 10,000
    Embedding dimension: 64

The conceptual transformation was:

    token ID
        ↓
    dense 64-dimensional vector

This was a major conceptual change from TF-IDF.

TF-IDF assigns a scalar importance value to a feature.

An embedding instead represents a token using a learned vector.

That vector can participate in learned relationships with other tokens.

The project had now moved from manually calculated word importance
toward learned representations.

======================================================================
17. LSTM: THE FIRST MAJOR SURPRISE
======================================================================

The initial LSTM architecture was:

    Input
        ↓
    Embedding(10000, 64)
        ↓
    LSTM(64)
        ↓
    Sigmoid output

The model contained approximately:

    673,089 parameters

The first LSTM attempt performed unexpectedly poorly, hovering around
random-guess behavior.

This was not treated as proof that LSTM was useless.

Instead, we investigated the representation.

The important issue was padding.

When sequences are forced to length 500, shorter reviews contain padding
tokens.

The LSTM should not interpret padding as meaningful language.

The solution was to use:

    mask_zero=True

in the Embedding layer.

This tells the downstream sequence model that token ID 0 represents
padding and should not be treated as ordinary information.

A small 500-sample experiment was used as a sanity check before
retraining the full model.

This was a good example of an engineering principle used throughout the
project:

    Before spending a large amount of compute on a full experiment,
    validate the suspected fix on a small controlled experiment.

======================================================================
18. FINAL LSTM RESULT
======================================================================

After correcting the padding behavior with masking, the LSTM was trained
on the full dataset.

The five-epoch result was:

    Epoch 1:
        Train Accuracy: 71.93%
        Validation Accuracy: 84.73%

    Epoch 2:
        Train Accuracy: 88.97%
        Validation Accuracy: 86.96%

    Epoch 3:
        Train Accuracy: 92.13%
        Validation Accuracy: 87.03%

    Epoch 4:
        Train Accuracy: 92.45%
        Validation Accuracy: 85.72%

    Epoch 5:
        Train Accuracy: 95.20%
        Validation Accuracy: 85.41%

Best validation accuracy:

    87.03%

This was lower than the Dense TF-IDF network’s 89.92%.

That result was not ignored.

It demonstrated that a theoretically more sophisticated architecture
does not automatically outperform a strong baseline.

The LSTM also showed increasing separation between training and
validation performance, suggesting overfitting.

======================================================================
19. ATTENTION: UNDERSTANDING THE NEXT GENERATION
======================================================================

Before implementing a Transformer, the project slowed down again to
understand attention.

Rather than immediately importing a Transformer architecture, attention
was manually explored.

The basic attention flow was studied through:

    Q = Query
    K = Key
    V = Value

The conceptual calculation was:

    scores = Q × Kᵀ

followed by scaling:

    scores / sqrt(d_k)

then:

    softmax(scores)

and finally:

    attention weights × V

This helped establish what attention is actually doing.

Instead of processing language strictly one step at a time, attention
allows tokens to assign different importance to other tokens in the
sequence.

This is one of the key ideas that eventually leads to Transformer
architectures.

======================================================================
20. MULTI-HEAD ATTENTION AND AN IMPORTANT COMPUTATIONAL LIMIT
======================================================================

The project then used Keras MultiHeadAttention.

The embedding dimension was:

    64

and the number of attention heads was:

    4

A single review had the shape:

    (1, 500, 64)

The attention output retained:

    (1, 500, 64)

Attention scores were observed with a shape equivalent to:

    (1, 4, 500, 500)

That last part is important.

For every head, attention creates relationships between every token and
every other token.

With sequence length 500, this means:

    500 × 500

attention relationships per head.

When we attempted to process the entire training set at once, the
attention tensor became enormous.

A conceptual shape encountered was approximately:

    [17500, 4, 500, 500]

This caused an out-of-memory problem.

The solution was not to abandon attention.

Instead, attention was tested on a single review or a small batch.

This was a direct demonstration of the computational cost of
self-attention.

The lesson was:

    A model can be mathematically valid and still be computationally
    impossible to run at an inappropriate batch size.

======================================================================
21. BUILDING A TRANSFORMER ENCODER FROM SCRATCH
======================================================================

The next stage implemented a small Transformer encoder block.

The architecture contained:

    Multi-Head Self-Attention
        ↓
    Residual connection + Layer Normalization
        ↓
    Feed-Forward Network
        ↓
    Residual connection + Layer Normalization

The feed-forward network used:

    Dense(128, ReLU)
        ↓
    Dense(64)

The encoder used:

    embedding dimension = 64
    number of heads = 4
    feed-forward dimension = 128

The resulting Transformer classifier had approximately:

    673,537 parameters

This was intentionally small compared with modern pretrained language
models.

The objective was not to reproduce BERT.

The objective was to understand the building blocks that make BERT
possible.

======================================================================
22. TRANSFORMER RESULTS
======================================================================

The Transformer classifier was trained for five epochs.

Results:

    Epoch 1:
        Train Accuracy: 80.10%
        Validation Accuracy: 87.32%

    Epoch 2:
        Train Accuracy: 91.30%
        Validation Accuracy: 86.92%

    Epoch 3:
        Train Accuracy: 94.31%
        Validation Accuracy: 87.57%

    Epoch 4:
        Train Accuracy: 96.22%
        Validation Accuracy: 87.13%

    Epoch 5:
        Train Accuracy: 97.46%
        Validation Accuracy: 86.85%

Best validation accuracy:

    87.57%

Again, this was lower than the Dense TF-IDF model’s best validation
accuracy.

This was a valuable result because it prevented the project from making
the simplistic claim:

    "Transformer = automatically better."

A Transformer architecture implemented from scratch with a small
embedding size, limited capacity, and only a basic encoder is not the
same thing as a large pretrained language model.

The project therefore moved to BERT.

======================================================================
23. A TRANSFORMER MASK WARNING
======================================================================

During the custom Transformer experiment, Keras produced a warning
indicating that the custom TransformerEncoder layer received an input
with a mask but did not explicitly support mask propagation.

The warning was essentially:

    The layer was passed an input with a mask attached,
    but the custom layer does not support masking and therefore
    the mask information may be destroyed.

This warning was not a fatal runtime error.

The model still trained.

At this stage, we made a deliberate engineering decision:

    Do not stop the entire project to redesign the Transformer
    masking implementation when the immediate educational goal
    is to understand the architecture and proceed to BERT.

A more complete Transformer implementation could explicitly support
masking and add positional encoding. Those are valid future
improvements.

For this project, the warning is documented rather than hidden.

======================================================================
24. THE MOVE TO BERT
======================================================================

BERT was the final architecture in the planned progression.

The original plan was to use TensorFlow/Keras and Hugging Face’s
TensorFlow model implementation.

The initial import attempted to use:

    BertTokenizer
    TFBertForSequenceClassification

However, the installed Transformers environment did not provide the
expected TensorFlow model class in the current setup.

Instead of spending excessive time fighting framework compatibility, the
project switched to the PyTorch implementation:

    AutoTokenizer
    AutoModelForSequenceClassification

The selected pretrained model was:

    google-bert/bert-base-uncased

This was an important engineering decision.

The goal was not to force one framework.

The goal was to use the appropriate implementation available in the
environment and continue the experiment.

======================================================================
25. THE FIRST BERT PROBLEM: LOCAL CPU PERFORMANCE
======================================================================

BERT was immediately different from the previous models.

The local environment was using CPU.

A first benchmark showed that training was far too slow for practical
iteration.

The original BERT dataset class tokenized reviews inside getitem.

That meant tokenization could happen repeatedly during data loading.

This was inefficient.

The pipeline was optimized by tokenizing the complete training and
validation sets once:

    train_encodings = tokenizer(
        train_text.tolist(),
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

and similarly for validation.

The resulting tensors were:

    Training:   [17500, 256]
    Validation: [7500, 256]

The optimized local benchmark was:

    10 training batches:
        74.75 seconds

Estimated time per epoch:

    approximately 272.6 minutes

That is more than four and a half hours for one epoch.

This was the moment where the original local-computing assumption became
invalid.

======================================================================
26. THE ENGINEERING DECISION: MIGRATE TO KAGGLE
======================================================================

A hard practical rule was established:

    If BERT training requires more than approximately 1–2 hours,
    do not waste time running the full experiment locally.

Instead of lowering the dataset size simply because the laptop could not
handle BERT efficiently, the project was migrated to Kaggle.

This was not a change in the scientific objective.

It was a change in compute infrastructure.

The entire notebook workflow was moved so that BERT could access
stronger GPU resources and more practical RAM/compute availability.

The Kaggle environment provided NVIDIA T4 GPUs.

The benchmark strategy was deliberately conservative:

    1. Verify CUDA.
    2. Verify the GPU.
    3. Run a small number of training batches.
    4. Estimate the full epoch runtime.
    5. Only run the full dataset if the estimate is acceptable.

This avoided committing to a multi-hour experiment without knowing how
long it would take.

======================================================================
27. A WINDOWS-SPECIFIC PROBLEM DURING MIGRATION
======================================================================

The original notebook contained a download command using:

    curl.exe

This worked in the Windows-oriented local environment but failed in
Kaggle’s Linux environment.

The error was effectively:

    /bin/bash: line 1: curl.exe: command not found

This was a useful reminder that notebooks are not automatically portable
across operating systems.

A command that works in Windows is not necessarily valid in a Linux
cloud environment.

The solution was to replace the Windows-specific download approach with
a platform-independent Python-based download method.

This was one of the small engineering problems that would be invisible
in a polished final project but is important in the actual development
journey.

======================================================================
28. ANOTHER KAGGLE PROBLEM: SPARSE TF-IDF + GPU/XLA
======================================================================

When earlier TensorFlow experiments were attempted on Kaggle GPU, the
sparse TF-IDF neural network encountered an XLA-related error involving:

    SparseFillEmptyRows

and:

    XLA_GPU_JIT

The important lesson was not to densify the TF-IDF matrix.

The TF-IDF representation was intentionally sparse, and converting it
into a dense matrix would create a severe memory problem.

The attempted engineering solution was to disable JIT compilation for
the affected model, for example:

    jit_compile=False

when compiling the model.

If necessary, TensorFlow’s global JIT setting could also be disabled.

The broader lesson was:

    GPU acceleration does not automatically mean every operation
    becomes GPU-compatible.

Some sparse operations and graph compilation paths have different
support characteristics.

This is exactly the type of issue that appears in real ML engineering.

======================================================================
29. BERT MODEL INITIALIZATION WARNINGS
======================================================================

When the BERT sequence-classification model was loaded, the Transformers
library displayed a load report containing:

    UNEXPECTED:
        cls.predictions.*
        cls.seq_relationship.*

and:

    MISSING:
        classifier.weight
        classifier.bias

This looked alarming at first, but it was expected.

The pretrained BERT checkpoint contains components used during BERT’s
original pretraining tasks.

For sentiment classification, we are using a different task head.

Therefore:

    the original pretraining-specific heads are not needed

and:

    the new sentiment classifier is initialized for the downstream task.

The missing classifier parameters are supposed to be newly initialized.

They are then trained during fine-tuning.

So the correct interpretation is:

    BERT encoder = pretrained
    Sentiment classification head = newly initialized and trained

This warning should not be treated as a failure.

======================================================================
30. BERT DATA PIPELINE OPTIMIZATION
======================================================================

The optimized BERT pipeline used pre-tokenized tensors rather than
tokenizing every review inside the DataLoader.

The tokenizer was configured with:

    truncation=True
    padding="max_length"
    max_length=256

The final training and validation tensor shapes were:

    Training input IDs:   [17500, 256]
    Validation input IDs: [7500, 256]

The DataLoaders used a batch size of 8 in the benchmarked setup.

This produced:

    Training batches:   2,188
    Validation batches:   938

The optimizer used was:

    AdamW

with:

    learning rate = 2e-5

This was a conventional fine-tuning configuration for BERT.

======================================================================
31. THE KAGGLE BENCHMARK
======================================================================

Before committing to the full dataset, the project benchmarked BERT on
the Kaggle GPU.

The benchmark indicated an estimated:

    approximately 16.4 minutes per epoch

This was comfortably below the project’s two-hour limit.

Therefore, the decision was:

    GO.

The full 17,500 training examples were retained.

The project did not reduce the dataset simply to make the benchmark
easier.

That was an important decision.

The objective was to make the hardware adapt to the experiment, not to
make the experiment artificially smaller just because the original
hardware was insufficient.

======================================================================
32. BERT TRAINING — EPOCH 1
======================================================================

The first full BERT epoch produced:

    Train Loss: 0.2833
    Train Accuracy: 0.8793

    Validation Loss: 0.2416
    Validation Accuracy: 0.9119

Therefore:

    BERT Epoch 1 validation accuracy = 91.19%

This immediately surpassed the previous best model.

The earlier best validation accuracy was:

    Dense TF-IDF Neural Network = 89.92%

So BERT improved the validation result by:

    1.27 percentage points

At this point, BERT had already become the strongest model in the
project.

======================================================================
33. BERT SANITY CHECKS
======================================================================

Before treating the model as the final result, several simple
human-readable reviews were passed through the trained model.

Example:

    "This movie was absolutely fantastic. I loved every minute of it."

Prediction:

    Positive
    Confidence: 99.14%

Example:

    "The acting was terrible and the story was painfully boring."

Prediction:

    Negative
    Confidence: 99.77%

Example:

    "I expected much more, but unfortunately this film was disappointing."

Prediction:

    Negative
    Confidence: 99.22%

These examples were intentionally simple sanity checks.

They are useful for confirming that the pipeline works and that the
model responds sensibly to obvious sentiment.

However, these predictions were not treated as proof of model quality.

The real comparison was based on validation performance.

======================================================================
34. BERT TRAINING — EPOCH 2
======================================================================

Because the first epoch was already fast enough to make additional
experimentation practical, a second epoch was intentionally tested.

The model was not reinitialized.

Instead, training continued from the already fine-tuned first-epoch
state.

Epoch 2 produced:

    Train Loss: 0.1481
    Validation Loss: 0.2214
    Validation Accuracy: 0.9181

Therefore:

    BERT Epoch 2 validation accuracy = 91.81%

This improved the first-epoch result:

    91.19% → 91.81%

Improvement:

    +0.62 percentage points

The validation loss also decreased:

    0.2416 → 0.2214

This was a good sign because the additional epoch did not merely improve
training performance while validation deteriorated.

======================================================================
35. FINAL MODEL LEADERBOARD
======================================================================

The final validation comparison was:

    TF-IDF + Logistic Regression
        Validation Accuracy: 87.95%

    LSTM
        Validation Accuracy: 87.03%

    Transformer Encoder
        Validation Accuracy: 87.57%

    TF-IDF + Dense Neural Network
        Best Validation Accuracy: 89.92%

    BERT
        Validation Accuracy: 91.81%

Final winner:

    BERT — 91.81% validation accuracy

BERT exceeded the previous best Dense Neural Network by:

    91.81 - 89.92
    = 1.89 percentage points

This was a meaningful improvement.

More importantly, the result made sense in the context of the
architectural progression.

======================================================================
36. WHAT THE FINAL RESULTS ACTUALLY TELL US
======================================================================

The project does not prove that every more advanced architecture will
always outperform every simpler model.

In fact, the results demonstrate the opposite.

The LSTM and small Transformer did not beat the Dense TF-IDF network.

This is useful.

Model quality depends on many factors:

    architecture
    representation
    parameter count
    training procedure
    sequence length
    masking
    positional information
    initialization
    pretraining
    hyperparameters
    regularization
    data
    compute

The biggest leap in this project came with BERT because BERT is not
simply “another Transformer.”

BERT brings a large pretrained language representation to the downstream
sentiment task.

The model begins with linguistic knowledge learned during pretraining
and then adapts that representation to sentiment classification.

That is a fundamentally different starting point from the small
Transformer trained from scratch in this project.

======================================================================
37. WHY THE SMALL TRANSFORMER DID NOT WIN
======================================================================

It would be misleading to look at the results and conclude:

    "Transformers are worse than LSTMs."

That is not what the experiment demonstrates.

The Transformer implemented here was intentionally small.

It used:

    64-dimensional embeddings
    4 attention heads
    128-dimensional feed-forward layer
    a simple encoder
    GlobalAveragePooling1D
    no full positional-encoding experiment in the final progression

It was trained from scratch on the IMDb sentiment task.

BERT, by contrast, was pretrained on a much broader corpus before
fine-tuning.

Therefore, the comparison is useful as an educational architectural
progression, but it is not a controlled scientific comparison of every
possible Transformer configuration.

======================================================================
38. WHAT WAS LEARNED ABOUT SPARSE MATRICES
======================================================================

One of the recurring engineering lessons was memory efficiency.

TF-IDF generated a matrix with tens of thousands of possible features.

Yet only a small fraction of the theoretical cells contained non-zero
values.

The project explicitly inspected:

    type(X_train)

and observed a SciPy CSR sparse matrix.

The project also inspected:

    X_train.shape
    X_train.nnz

This made the sparsity concrete rather than theoretical.

A dense conversion was deliberately avoided.

This principle carried forward into the GPU debugging.

When the sparse TensorFlow model encountered an XLA GPU operation
problem, the solution was to address the compilation path rather than
destroying the sparse representation.

======================================================================
39. WHAT WAS LEARNED ABOUT OVERFITTING
======================================================================

Overfitting appeared repeatedly.

The Dense Neural Network reached nearly perfect training accuracy while
validation accuracy plateaued.

The LSTM also showed training accuracy continuing to rise while
validation performance peaked and then declined.

The Transformer showed the same broad pattern:

    training accuracy increased
    validation accuracy peaked earlier
    validation loss eventually worsened

This reinforced a central principle:

    Training accuracy is not the same thing as generalization.

A model can memorize the training distribution without becoming better
at unseen data.

The project therefore consistently focused on validation accuracy rather
than celebrating training accuracy alone.

======================================================================
40. WHAT WAS LEARNED ABOUT DEBUGGING
======================================================================

The project encountered several classes of problems:

    1. Data-pipeline issues
    2. Data leakage concerns
    3. Sparse matrix memory concerns
    4. Neural-network overfitting
    5. Padding/masking problems
    6. Attention memory explosion
    7. Framework compatibility issues
    8. CPU performance limitations
    9. Operating-system-specific commands
    10. GPU/XLA compatibility issues
    11. Model checkpoint warnings
    12. Notebook state and variable-definition problems

A particularly practical example was the repeated occurrence of:

    NameError: name 'train_text' is not defined

and later:

    NameError: name 'df' is not defined

These happened after migration because a cloud notebook kernel does not
automatically inherit the state of a previous local notebook session.

The solution was not to patch the error blindly.

The required upstream variables were recreated:

    df
        ↓
    train_text / val_text
        ↓
    tokenization
        ↓
    DataLoader
        ↓
    BERT

This reinforced another practical rule:

    Notebook execution state matters.

A notebook is not just a document. It is also a stateful program.

======================================================================
41. WHY SMALL EXPERIMENTS MATTERED
======================================================================

Several times, the project avoided committing to expensive full runs
immediately.

Examples included:

    LSTM masking:
        test the corrected architecture on a small subset

    Attention:
        test one review instead of the entire dataset

    Transformer:
        test a small batch before full training

    BERT:
        benchmark a small number of batches before full training

This was one of the strongest engineering habits developed during the
project.

Instead of asking:

    "Will this work?"

we repeatedly asked:

    "What is the cheapest experiment that can tell us whether
     this will probably work?"

That is a much more practical way to develop machine-learning systems.

======================================================================
42. WHY THE PROJECT DID NOT CHASE EVERY POSSIBLE IMPROVEMENT
======================================================================

There were many possible side experiments.

For example:

    better Transformer masking
    positional encodings
    more Transformer layers
    different attention dimensions
    different LSTM sizes
    larger vocabularies
    longer sequence lengths
    more BERT epochs
    learning-rate schedules
    larger batch sizes
    mixed precision
    multiple GPUs
    more extensive hyperparameter search

Some of these would probably improve the system.

However, the project had a specific educational objective and a
practical time constraint.

The goal was not:

    "Perform an exhaustive hyperparameter search on IMDb."

The goal was:

    "Understand the evolution from classical NLP features to
     pretrained Transformers while documenting the engineering
     problems encountered along the way."

That distinction prevented the project from turning into an endless
optimization loop.

======================================================================
43. COMPUTE WAS TREATED AS AN ENGINEERING CONSTRAINT
======================================================================

One of the biggest lessons was that model selection cannot be separated
from hardware.

A model that is theoretically suitable may be impractical on a
particular machine.

The local BERT benchmark demonstrated this clearly:

    ~272.6 minutes per epoch on CPU

The Kaggle GPU benchmark changed the situation:

    ~16.4 minutes per epoch

That is a dramatic difference.

The model did not change.

The dataset did not change.

The task did not change.

The compute environment changed.

This is an important real-world machine-learning lesson:

    Performance engineering is part of machine learning.

======================================================================
44. WHY KAGGLE WAS NOT A “CHEAT”
======================================================================

The move from local VS Code/Jupyter to Kaggle was not a shortcut around
learning.

The project had already been developed locally through the classical and
neural stages.

Kaggle was used because BERT fine-tuning was computationally expensive.

The migration preserved:

    the dataset
    the train/validation split
    the model objective
    the sequence length
    the training data
    the evaluation approach

Only the execution environment changed.

The migration therefore represents a realistic engineering decision:

    use local compute when it is sufficient;
    use cloud GPU compute when the workload requires it.

======================================================================
45. REPRODUCIBILITY NOTE
======================================================================

Anyone running this repository should understand that the notebook was
developed across different environments.

The original development environment was local VS Code/Jupyter.

The BERT phase was migrated to Kaggle for GPU access.

Because of this, a user running the entire notebook in a different
environment may encounter:

    dependency-version differences
    framework warnings
    GPU availability differences
    CPU/GPU performance differences
    operating-system-specific command differences
    notebook state differences
    library API differences

The results shown in this documentation are the results obtained during
the actual development journey.

They should not be interpreted as a guarantee that every machine will
reproduce identical timing or identical floating-point results.

======================================================================
46. EXPECTED WARNINGS WHEN RUNNING THE NOTEBOOK
======================================================================

There may be a few warnings when someone runs this notebook.

They should not automatically be interpreted as errors.

One known example is the custom Transformer mask warning.

The custom TransformerEncoder did not explicitly propagate the Keras
mask, so Keras warned that mask information could be lost at that layer.

Another known example is the BERT loading report.

The report may show unexpected pretraining-head weights and missing
classification-head weights.

This is expected when loading a general BERT checkpoint into a
sequence-classification architecture.

In short:

    WARNING ≠ ERROR

However, this does not mean every warning in every future environment
should blindly be ignored.

The recommendation is:

    If the warning matches the documented behavior described here,
    it can be treated as known and non-fatal.

    If a new warning is accompanied by incorrect results, a crash,
    missing data, or a shape/device error, it should be investigated.

======================================================================
47. KNOWN LIMITATIONS OF THE PROJECT
======================================================================

This project has several limitations, and documenting them is important.

1.  The dataset is a classic benchmark dataset.

IMDb sentiment classification is useful for learning, but it does not
represent every type of modern social-media, customer-review, or
conversational sentiment.

2.  The evaluation uses a train/validation split.

The project did not perform an independent external test against a new
real-world dataset.

3.  The models were not exhaustively hyperparameter tuned.

The purpose was architectural understanding rather than leaderboard
optimization.

4.  The Transformer was intentionally small.

It was not designed to compete with pretrained Transformer models.

5.  The custom Transformer did not implement a complete
    masking/positional-encoding treatment.

This is a known simplification.

6.  BERT was fine-tuned for only two epochs.

This was a deliberate compute/time decision.

7.  BERT used a maximum sequence length of 256.

Reviews longer than the limit were truncated.

8.  Confidence values from BERT are not guaranteed to represent
    calibrated probabilities.

A high confidence score should not automatically be interpreted as
certainty.

9.  Results may vary slightly between environments.

Different library versions, hardware, random states, and numerical
implementations can influence training.

10. The project is primarily educational and engineering-oriented.

It should not be interpreted as a production-ready sentiment-analysis
service.

======================================================================
48. WHAT THE AUTHOR ACTUALLY DID
======================================================================

The strongest part of this project is not simply the final 91.81%
number.

The strongest part is the sequence of decisions that produced it.

The author did not begin with BERT.

The author started with a simple baseline and progressively challenged
its limitations.

The author:

    • Loaded and inspected the raw dataset.
    • Verified the class balance.
    • Built a TF-IDF representation.
    • Learned the difference between vocabulary and TF-IDF values.
    • Trained Logistic Regression.
    • Inspected errors instead of hiding them.
    • Inspected model coefficients.
    • Tested bigrams as a specific hypothesis.
    • Learned that more features do not automatically improve performance.
    • Investigated sparse matrices and memory.
    • Built a dense neural network.
    • Observed overfitting.
    • Tested Dropout.
    • Learned that regularization does not guarantee improvement.
    • Moved from word features to token sequences.
    • Examined review-length distributions before choosing sequence length.
    • Built embeddings.
    • Built an LSTM.
    • Debugged a severe masking problem.
    • Validated the masking fix on a smaller experiment.
    • Trained the corrected LSTM.
    • Manually explored attention.
    • Examined Q, K, V and scaled dot-product attention.
    • Investigated attention score shapes.
    • Encountered an out-of-memory problem with full-batch attention.
    • Reduced attention experiments to manageable samples.
    • Built a Transformer encoder from scratch.
    • Observed and documented a masking warning.
    • Decided to move forward rather than endlessly optimize the intermediate Transformer.
    • Attempted TensorFlow-based BERT.
    • Encountered framework compatibility limitations.
    • Switched to PyTorch/Hugging Face.
    • Optimized BERT tokenization.
    • Benchmarked local CPU performance.
    • Determined that local BERT training was impractical.
    • Migrated the project to Kaggle.
    • Adapted Windows-specific code for a Linux cloud environment.
    • Verified GPU availability.
    • Benchmarked BERT before committing to full training.
    • Kept the complete 17,500-example training set.
    • Fine-tuned BERT for one epoch.
    • Evaluated the result.
    • Chose to run a second epoch because the runtime was acceptable.
    • Improved validation accuracy from 91.19% to 91.81%.
    • Performed human-readable prediction sanity checks.
    • Stopped once the result was strong enough for the project's objective.

That is the actual journey.

======================================================================
49. THE ENGINEERING MINDSET DEVELOPED THROUGH THE PROJECT
======================================================================

The project gradually shifted from:

    "Can I make this model run?"

to:

    "Why is this model behaving this way?"

and eventually:

    "What is the cheapest reliable experiment that can tell me
     what to do next?"

That progression matters.

Machine learning is not only about knowing APIs.

It involves:

    problem formulation
    data validation
    representation choices
    model selection
    debugging
    computational reasoning
    experiment design
    evaluation
    tradeoffs
    documentation

The project was therefore as much about engineering judgment as it was
about model architecture.

======================================================================
50. FINAL TECHNICAL SCORECARD
======================================================================

Dataset:

    IMDb
    25,000 labeled training reviews
    12,500 positive
    12,500 negative

Split:

    17,500 training
    7,500 validation

Final model results:

    TF-IDF + Logistic Regression
        Validation Accuracy: 87.95%

    TF-IDF + Dense Neural Network
        Best Validation Accuracy: 89.92%

    LSTM
        Best Validation Accuracy: 87.03%

    Transformer Encoder
        Best Validation Accuracy: 87.57%

    BERT — Epoch 1
        Validation Accuracy: 91.19%

    BERT — Epoch 2
        Validation Accuracy: 91.81%

Final best:

    BERT
    91.81% validation accuracy

BERT improvement over the best earlier model:

    +1.89 percentage points

======================================================================
51. THE BIGGEST LESSONS
======================================================================

Lesson 1:

    Start with a baseline.

Without Logistic Regression, the later improvements would have had no
meaningful reference point.

Lesson 2:

    Analyze errors before changing the model.

The errors explained why a bag-of-words approach had limitations.

Lesson 3:

    More features do not necessarily mean more understanding.

The bigram experiment increased the feature space dramatically but
slightly reduced validation accuracy.

Lesson 4:

    Sparse data should remain sparse when possible.

Densifying TF-IDF would have created unnecessary memory pressure.

Lesson 5:

    High training accuracy can be misleading.

Several neural models learned the training data extremely well while
validation performance stopped improving.

Lesson 6:

    Small experiments can save hours.

The LSTM masking issue, attention memory issue, and BERT runtime were
all investigated with smaller tests before full-scale runs.

Lesson 7:

    Padding is data too.

If padding is not handled correctly, a sequence model can learn from
meaningless tokens.

Lesson 8:

    Attention has a real computational cost.

The quadratic relationship between sequence length and attention
relationships matters.

Lesson 9:

    Pretraining changes the starting point.

The small Transformer had to learn language patterns from the sentiment
dataset. BERT arrived with a pretrained language representation.

Lesson 10:

    Hardware is part of the engineering problem.

The local CPU was not a practical environment for BERT. Kaggle GPU
compute changed the feasibility of the experiment.

Lesson 11:

    Warnings should be understood, not feared.

The important distinction is between a known non-fatal warning and an
actual failure.

Lesson 12:

    A failed experiment can still be a successful experiment.

Dropout did not improve validation performance. Bigram TF-IDF did not
improve validation performance. LSTM and the small Transformer did not
beat the dense baseline.

Each result still taught us something.

======================================================================
52. ACKNOWLEDGEMENT OF AI-ASSISTED TECHNICAL SUPPORT
======================================================================

This project was developed with significant technical assistance from AI
during the problem-solving process.

The AI assistance was used as a technical companion rather than as a
replacement for the author’s decision-making.

The author remained responsible for:

    • deciding the project direction
    • choosing the progression of models
    • deciding when to move forward
    • interpreting results
    • testing proposed solutions
    • deciding which experiments were worth running
    • setting practical compute limits
    • choosing to migrate to Kaggle
    • deciding when additional experimentation was unnecessary

AI assistance was particularly useful for:

    • explaining machine-learning concepts
    • interpreting errors and warnings
    • suggesting debugging experiments
    • explaining sparse matrices
    • reasoning about attention tensor sizes
    • diagnosing padding and masking behavior
    • improving the BERT data pipeline
    • interpreting framework compatibility issues
    • estimating computational feasibility
    • suggesting engineering tradeoffs
    • helping document the journey

The most valuable part of the collaboration was not simply receiving
code.

It was the iterative process:

    problem
        ↓
    investigation
        ↓
    explanation
        ↓
    small experiment
        ↓
    result
        ↓
    next decision

The AI provided technical guidance and explanations, while the author
tested, questioned, accepted, rejected, modified, and implemented the
decisions.

This distinction is important.

The project should be understood as an AI-assisted engineering and
learning project, not as a claim that every line was independently
written without assistance.

======================================================================
53. A NOTE ON LEARNING THROUGH THE PROCESS
======================================================================

One of the goals of this project was to avoid treating machine-learning
libraries as black boxes.

Whenever possible, the notebook paused to inspect:

    shapes
    vocabulary
    token IDs
    embeddings
    sparse matrices
    model parameters
    attention scores
    training curves
    validation curves
    prediction errors

This created a different relationship with the code.

Instead of:

    "Run this because the tutorial says so."

the project tried to ask:

    "What object exists right now?"
    "What shape does it have?"
    "What information does it contain?"
    "Why does the next layer expect this shape?"
    "What changed after this operation?"
    "What does this result tell us?"

That mindset is arguably more valuable than memorizing a particular API.

======================================================================
54. WHY THE FINAL PROJECT IS CALLED “SENTIMENT EVOLUTION”
======================================================================

The name reflects the actual structure of the project.

It is not simply:

    "IMDb Sentiment Analysis using BERT."

It is an evolution:

    sparse lexical features
        ↓
    linear classification
        ↓
    nonlinear feature combinations
        ↓
    sequential representations
        ↓
    attention
        ↓
    Transformer architecture
        ↓
    pretrained language representation

Each stage exists because the previous stage exposed a limitation or
raised a new question.

That is the central story of this repository.

======================================================================
55. WHAT THIS PROJECT IS — AND WHAT IT IS NOT
======================================================================

This project is:

    • an NLP learning journey
    • an engineering experiment
    • an architecture comparison
    • a debugging record
    • a practical demonstration of model evolution
    • an exploration of computational tradeoffs
    • an AI-assisted learning project

This project is not:

    • an exhaustive benchmark of every sentiment model
    • a production-ready sentiment API
    • a claim that BERT always wins
    • a fully optimized Transformer implementation
    • a comprehensive hyperparameter search
    • a replacement for understanding the underlying concepts

The repository intentionally preserves the journey.

======================================================================
56. FINAL REFLECTION
======================================================================

The most satisfying part of this project is not that BERT reached
91.81%.

It is that the final result has a story behind it.

We started with a simple question:

    "Can we classify movie reviews as positive or negative?"

A straightforward TF-IDF + Logistic Regression model answered that
question at 87.95%.

Then we asked:

    "Can we capture more complex relationships?"

A dense neural network pushed the best result to 89.92%, but exposed
overfitting.

We tried adding local context with bigrams.

The score slightly decreased.

We moved to sequence modeling.

The LSTM introduced embeddings and recurrent memory, but the first
implementation exposed a padding/masking problem. After fixing it, the
model reached 87.03%.

We then stepped into attention.

We manually explored Q, K, V and attention weights, and eventually built
a Transformer encoder.

The Transformer reached 87.57%, but also exposed the computational cost
of attention and a masking warning.

Then came BERT.

The first local attempt revealed that the laptop was not a practical
environment for fine-tuning it.

Instead of giving up, the project moved to Kaggle.

The code had to be adapted.

The environment had to be rebuilt.

The data had to be reloaded.

The BERT input pipeline had to be optimized.

The GPU was benchmarked before committing to the full experiment.

The result was:

    91.19% after one epoch

and then:

    91.81% after two epochs.

That final number is useful.

But the journey is more useful.

Because the real lesson was not:

    "BERT gets 91.81%."

The real lesson was:

    "When a model fails, investigate why.
     When an experiment is slow, benchmark it.
     When a representation is limited, understand the limitation.
     When a warning appears, interpret it.
     When hardware becomes the bottleneck, change the environment.
     And when an experiment does not improve the score,
     learn from what it tells you."

That is the engineering mindset this project was meant to develop.

======================================================================
57. REPRODUCTION / RUNNING NOTE
======================================================================

If you are running the notebook yourself:

    1. Make sure the IMDb dataset is available.
    2. Make sure the required Python libraries are installed.
    3. Run the notebook in a sensible order so variables are defined.
    4. If using Kaggle, adapt any local/Windows-specific commands.
    5. For BERT, GPU access is strongly recommended.
    6. Expect different runtimes depending on hardware.
    7. Expect a few known warnings documented above.
    8. Do not confuse those known warnings with fatal errors.
    9. If a new error appears, inspect the environment, tensor shapes,
       device placement, and notebook execution order before changing
       the model itself.

The notebook is a record of an actual development journey, so it may
contain exploratory cells, intermediate experiments, and diagnostic
outputs.

That is intentional.

======================================================================
58. CLOSING ACKNOWLEDGEMENT
======================================================================

This project represents a combination of persistence, experimentation,
engineering judgment, and technical assistance.

The author deserves credit for continuing through the frustrating parts:

    the slow downloads
    the environment differences
    the sparse-matrix issues
    the overfitting
    the LSTM failure
    the masking problem
    the attention memory explosion
    the Transformer warning
    the framework compatibility issue
    the extremely slow local BERT training
    the Kaggle migration
    the missing notebook variables
    the GPU/XLA complications
    and the many small problems that appear when code moves
    from one environment to another.

The final result exists because the project kept moving.

AI assistance also played a meaningful role by helping explain technical
concepts, debug failures, compare options, and reason through the
engineering decisions.

The author, however, remained the person making the decisions, running
the experiments, interpreting the results, and deciding what the project
should become.

The repository therefore represents something more valuable than a final
model:

    a documented learning process.

And that process is the real project.

======================================================================
END OF PROJECT JOURNEY DOCUMENTATION
======================================================================
