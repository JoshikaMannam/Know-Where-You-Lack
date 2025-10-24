# Know Where You Lack

A machine learning project for predicting and analyzing student performance across multiple datasets.

## Datasets Used
- AI Course Dataset: Student performance data from AI courses
- UCI Dataset: Student performance data from Portuguese schools (Mathematics and Portuguese language courses)
- OU Dataset: Open University Learning Analytics Dataset

## Project Structure
```
.
 data/
    processed/    # Processed and cleaned datasets
    raw/         # Original source datasets
 models/          # Trained model files
 reports/         # Analysis reports and visualizations
 results/         # Final results and comparisons
 src/            # Source code
     preprocessor/  # Data preprocessing modules
```

## Features
- Multi-dataset analysis of student performance
- Ensemble modeling (Random Forest + XGBoost)
- Performance prediction across different academic contexts
- Feature importance analysis
- Comparative analysis with literature results

## Setup
1. Clone the repository:
```bash
git clone https://github.com/JoshikaMannam/Know-Where-You-Lack.git
cd Know-Where-You-Lack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the datasets:
- AI Course Dataset: [Link to be added]
- UCI Dataset: https://archive.ics.uci.edu/ml/datasets/Student+Performance
- OU Dataset: [Link to be added]

Place the downloaded datasets in their respective folders under `data/raw/`.

## Usage
1. Preprocess the data:
```bash
python src/run_preprocessing.py
```

2. Train the models:
```bash
python src/train_models.py
```

3. Evaluate and analyze results:
```bash
python src/evaluate_model.py
```

## Results
- Achieved high accuracy in predicting student performance levels
- Identified key factors influencing academic success
- Generated comprehensive comparison with existing literature

## License
MIT License
