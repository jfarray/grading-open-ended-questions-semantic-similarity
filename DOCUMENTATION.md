# Sentence Transformer Evaluation for Semantic Similarity

## Overview

This project explores the use of Sentence Transformers to estimate the semantic similarity between a teacher reference answer and a student answer, and to compare that semantic similarity score with the teacher-assigned grade. The goal is to evaluate whether sentence embeddings and cosine similarity can be used as a proxy for quality assessment in short educational responses.

The project is implemented through a combination of Jupyter notebooks and Python scripts that cover:

- data preparation from JSON files
- model experimentation with multiple transformer checkpoints
- training and evaluation using cosine similarity objectives
- metric and correlation analysis
- a simple Gradio-based demo for testing new inputs

## Project files

- `SentenceTransformer_Training.ipynb` — training workflow and model benchmarking
- `SentenceTransformer_Testing.ipynb` — evaluation pipeline and export of metrics
- `SentenceTransformer_Testing_GetCorrelations.ipynb` — correlation and regression analysis
- `PreparingDataSet.ipynb` — dataset construction and preliminary validation
- `sentencetransformer_training.py` — training script converted from the notebook
- `sentencetransformer_testing.py` — testing and metrics script
- `preparingdataset.py` — Gradio interface for processing JSON files and comparing text similarity

## Objectives

The core objective of the project is to answer the following question:

- Can a sentence embedding model reliably quantify how semantically close a student response is to a teacher reference response?

To evaluate this, the project:

1. loads dataset entries containing a teacher answer and a student answer
2. assigns a numeric grade or score to each pair
3. trains and compares multiple SentenceTransformer checkpoints
4. measures how well the model-predicted semantic similarity aligns with the actual marks
5. analyzes statistical correlations between grades and similarity values

## Data format

The dataset is stored in JSON and loaded with the Hugging Face `datasets` library. Each sample contains fields such as:

- `hashed_id`
- `mark` or `nota`
- `responseStudent` or `respuesta`
- `responseTeacher` or `metadata.minipreguntas[i].minirespuesta`

The training pipeline converts each sample into an `InputExample` object with:

- `texts=[responseTeacher, responseStudent]`
- `label=mark`

This structure allows the model to learn a semantic relationship between two texts and a score indicating their similarity or alignment.

## Data preparation process

The data preparation stage focuses on building a training dataset from JSON files and normalizing the structure so it can be consumed by SentenceTransformers.

### Main steps

- load the JSON dataset
- iterate over each item in the training set
- collect teacher text and student text
- build a pair of texts for sentence comparison
- store the teacher grade as a regression label
- convert each example into `InputExample`

The preparation script also includes a Gradio interface for quick experimentation with a chosen model and JSON file. It:

- reads a dataset file
- converts the file into sample pairs
- runs the selected model against the examples
- returns a table with the student ID, score, and semantic similarity

## Training workflow

The training notebook and script use a sequence of SentenceTransformer models as candidate encoders. The code tests multiple checkpoints, including:

- `distiluse-base-multilingual-cased-v1`
- `paraphrase-multilingual-MiniLM-L12-v2`
- `paraphrase-multilingual-mpnet-base-v2`
- `all-distilroberta-v1`
- `bert-base-multilingual-uncased`
- `dccuchile/bert-base-spanish-wwm-uncased`

Some runs start from a pretrained checkpoint, while others are trained from scratch with additional transformer and pooling layers.

### Training logic

The project uses:

- `SentenceTransformer(checkpoint)` for transfer learning
- custom `SentenceTransformer` modules when training from scratch
- `DataLoader` for batching
- `losses.CosineSimilarityLoss` as the optimization objective
- `EmbeddingSimilarityEvaluator` to evaluate progress on a validation set

The script also defines:

- `CreateModelEvaluationData()`
- `TrainModel(checkpoint, evaluator, epochs, fromScratch)`

This allows the model to be trained and saved with a local path such as:

- `Model_<checkpoint>/<epochs>_Epochs`

The code then uploads the model to the Hugging Face Hub using a repository name derived from the checkpoint and epoch count.

## Evaluation strategy

The testing workflow compares model outputs with human-assigned marks. For each model and each epoch setting, the process:

1. loads the trained model
2. encodes both teacher and student text embeddings
3. computes cosine similarity between the pairs
4. stores the similarity score for each sample
5. calculates error and performance metrics
6. saves the results to CSV files

### Example metrics used

The pipeline computes the following regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Root Mean Squared Log Error (RMSLE) when valid
- Mean Absolute Percentage Error (MAPE)
- R-squared (R2)

These metrics are saved as CSV outputs such as:

- `Model_<checkpoint>_Metrics_Results.csv`
- `Model_<checkpoint>_Scores_Results.csv`

## Correlation analysis

The notebook `SentenceTransformer_Testing_GetCorrelations.ipynb` focuses on statistical validation of the relationship between teacher grades and semantic similarity scores.

It reads the CSV result files and converts the grade and similarity columns to numeric values. Then it computes:

- Pearson correlation
- Spearman rank correlation
- Kendall rank correlation
- linear regression between grade and similarity score

The notebook plots each model's data points and overlays the regression line, producing visual summaries of how well semantic scores track teacher grades.

The generated plots help answer whether the similarity metric is linearly or monotonically aligned with the marks.

## Semantic similarity scoring

The model encodes each sentence into a vector embedding and compares them using cosine similarity. In the code, this is done as follows:

- `embeddings1 = model.encode(sentences1, convert_to_tensor=True)`
- `embeddings2 = model.encode(sentences2, convert_to_tensor=True)`
- `cosine_scores = util.cos_sim(embeddings1, embeddings2)`

For each item, the diagonal value of the cosine similarity matrix is selected:

- `cosine_scores[i][i]`

This corresponds to the similarity between the teacher answer and the student answer for that specific pair.

## Gradio demo

The file `preparingdataset.py` includes a small web interface built with Gradio. The interface allows a user to:

- select a trained model
- upload a JSON file
- run the comparison pipeline
- view the resulting similarity table

The tool returns a DataFrame with columns like:

- `Hashed_id`
- `Nota`
- `Similitud Semántica`

This is useful for testing new examples without re-running the full training pipeline.

## Experimental configuration

The project runs training and evaluation over multiple model checkpoints and several epoch values:

- epochs tested: `1, 5, 10, 30, 50, 100`
- models tested: several multilingual and Spanish-language Transformers

The purpose is to identify which checkpoint-epoch combination yields the best trade-off between learning capacity and predictive quality for this educational similarity task.

## Workflow summary

The complete workflow can be summarized as:

1. Prepare a JSON dataset with teacher/student pairs and grades.
2. Convert each pair to an `InputExample`.
3. Train one or more SentenceTransformer models.
4. Evaluate them on validation or test data using semantic similarity objectives.
5. Measure similarity between teacher and student texts.
6. Compare the resulting similarity scores with teacher marks.
7. Compute error metrics and correlations to determine model quality.
8. Save the trained models and CSV reports for later comparison.

## Expected use cases

This project is appropriate for tasks such as:

- automatic similarity assessment of short educational answers
- grading support tools based on semantic closeness
- content comparison between reference and student responses
- benchmarking multilingual sentence embedding models on educational text

## Limitations and considerations

Although the project demonstrates a valid evaluation workflow, some considerations should be kept in mind:

- the quality of the score depends strongly on the dataset and annotation consistency
- cosine similarity is a semantic approximation, not a perfect substitute for human grading
- some models may overfit or underperform depending on the number of epochs and training volume
- the text domain matters: education-specific language may require domain-adapted data

## Conclusion

This project demonstrates an end-to-end pipeline for evaluating sentence embedding models in a semantic text comparison task. By combining multiple transformer checkpoints, cosine-similarity training, evaluation metrics, and correlation analysis, it provides a reproducible framework for measuring whether semantic similarity can approximate academic grading.

The code and notebooks show both the experimental research approach and the practical implementation of a model comparison workflow for educational text similarity.
