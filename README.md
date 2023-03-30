# Plagiarism-Checker

Based on tf-idf values, cosine similarity, jaccard distance similarity, edit distance similarity and ranked retrieval

# Prerequisites

- nltk
- numpy

Install using

<pre><code>pip install <package_name></code></pre>

# How to run

1. First we will create train_vocab file from DATASET, it contains data by command:

   ```
   python train.py
   ```

   2 . Run the main.py with arguments from cl are the test file we want to check plagiarism and type algorithm that we want to use (Choose one)

```
python main.py TEST/[filename] ['Cosine'/ 'Jaccard'/ 'Edit']
```
