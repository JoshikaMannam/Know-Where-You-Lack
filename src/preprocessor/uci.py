from .base import BasePreprocessor
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

class UCIPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__("UCI")
    
    def load_data(self):
        """Load UCI Student Performance Dataset"""
        self.logger.info("Loading UCI dataset...")
        
        # Load math and portuguese datasets
        uci_dir = self.raw_dir / 'uci_data'
        math_path = uci_dir / "student-mat.csv"
        por_path = uci_dir / "student-por.csv"
        
        math_df = pd.read_csv(math_path, sep=';')
        por_df = pd.read_csv(por_path, sep=';')
        
        # Add subject identifier
        math_df['subject'] = 'math'
        por_df['subject'] = 'portuguese'
        
        # Combine datasets
        self.raw_data = pd.concat([math_df, por_df], ignore_index=True)
        self.report['original_shape'] = self.raw_data.shape
        self.report['features_before'] = self.raw_data.columns.tolist()
        
        return self.raw_data
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features for UCI dataset"""
        self.logger.info("Performing feature engineering...")
        
        # Average of early test scores (only G1, G2)
        df['avg_score'] = df[['G1', 'G2']].mean(axis=1)
        
        # Score trend (using only early scores)
        df['score_trend'] = (df['G2'] - df['G1'])
        
        # Performance consistency
        df['performance_consistency'] = df[['G1', 'G2', 'G3']].std(axis=1)
        
        # At risk flag
        df['at_risk_flag'] = (df['avg_score'] < 10).astype(int)
        
        # Study time efficiency
        df['time_efficiency'] = df['avg_score'] / df['studytime']
        
        return df
    
    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess UCI dataset following the defined steps"""
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 4: Handle outliers
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df = self.handle_outliers(df, numeric_columns)
        
        # Step 5: Create target variable
        df = self.create_weakness_level(df, 'avg_score')
        
        # Step 6: Encode categorical variables
        df = self.encode_categorical(df)
        
        # Step 7: Train-test split
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['weakness_level']
        )
        
        # Step 8: Scale features
        numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
        train_df, test_df = self.scale_features(train_df, test_df, numeric_columns)
        
        # Update report
        self.report['final_shape'] = df.shape
        self.report['features_after'] = df.columns.tolist()
        
        # Class distribution
        self.report['class_distribution'] = {
            'train': train_df['weakness_level'].value_counts().to_dict(),
            'test': test_df['weakness_level'].value_counts().to_dict()
        }
        
        # Save processed data and report
        self.save_data(train_df, test_df)
        self.save_report()
        
        # Plot distributions
        plot_columns = ['avg_score', 'score_trend', 'performance_consistency']
        self.plot_distributions(
            self.raw_data,
            train_df,
            plot_columns,
            self.reports_dir / f"{self.dataset_name}_distributions.png"
        )
        
        return train_df, test_df