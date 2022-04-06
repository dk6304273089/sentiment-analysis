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