```bash
│   .dvcignore
│   Data_pipeline.py
│   dvc.lock
│   dvc.yaml
│   params.yaml
│   README.md
│   requirements.txt
│   setup.py
│
├───.dvc
│   │   .gitignore
│   │   config
│   │
│   └───tmp
│       │  
│       │   lock
│       │   rwlock
│       ├───links
│       │       cache.db
│       └───md5s
│               cache.db
│
├───configs
│       .gitignore
│       config.yaml
│       credentials.yaml
│
├───data
│   │   .gitignore
│   │   test.csv
│   │   test.csv.dvc
│   │   test_ranked_output.csv
│   │   test_ranked_output.csv.dvc
│   │   test_withoutlabel.csv
│   │   test_withoutlabel.csv.dvc
│   │
│   ├───extracted_data
│   │       .gitignore
│   │
│   ├───feature_processed
│   │       .gitignore
│   │       feature.csv
│   │       feature.csv.dvc
│   │       rank.csv
│   │       rank.csv.dvc
│   │
│   └───processed_data
│           .gitignore
│           processed.csv
│           processed.csv.dvc
│
├───logs
│       running_logs.log
│
├───models
│       gib_model.pki
│       random_forest.joblib
│
├───report
│       my_report.html
│       params.json
│       scores.json
│
├───src
│   │   stage_01_get_data.py
│   │   stage_02_preprocessing.py
│   │   stage_03_feature_engineering.py
│   │   stage_04_model_training.py
│   │   __init__.py
│   │
│   ├───DictionaryUtils
│   │       company_tags.txt
│   │       english_profanity_google.txt
│   │       service_tagger.txt
│   │
│   ├───utils
│   │   │   common.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           common.cpython-38.pyc
│   │           __init__.cpython-38.pyc
│   │
│   └───__pycache__
│           stage_02_preprocessing.cpython-38.pyc
│           stage_03_feature_engineering.cpython-38.pyc
│           stage_04_model_training.cpython-38.pyc
│           __init__.cpython-38.pyc
│
├───src.egg-info
│       dependency_links.txt
│       PKG-INFO
│       SOURCES.txt
│       top_level.txt
│
└───__pycache__
        secrets.cpython-38.pyc

```
DESCRIPTION
```bash
-> E-Commerce applications provide an added advantage to customers to buy a product with added suggestions in the form of reviews. Obviously, reviews are useful and impactful for customers who are going to buy the products. But these enormous amounts of reviews also create problems for customers as they are not able to segregate useful ones. Regardless, these immense proportions of reviews make an issue for customers as it becomes very difficult to filter informative reviews. This proportional issue has been attempted in this project. The approach that we discuss in detail later ranks reviews based on their relevance with the product and rank down irrelevant reviews.

-> This work has been done in four phases- data preprocessing/filtering (which includes Language Detection, Gibberish Detection, Profanity Detection), feature extraction, pairwise review ranking, and classification. The outcome will be a list of reviews for a particular product ranking on the basis of relevance using a pairwise ranking approach.
```
DATASET
```bash
-> Download the dataset for custom training
-> https://drive.google.com/drive/folders/1_z9MaY3zIqZdnhIifnF8N63wemet8Q2w?usp=sharing
```
STEPS-
```bash
-> Step 1: Clone the repository
-> Step 2: Create conda environment
-> Step 3: Conda activate environment
-> Step 4: Install the requirements.txt
-> Step 5: Initializing the dvc
-> Step 6: Run dvc repro command
```