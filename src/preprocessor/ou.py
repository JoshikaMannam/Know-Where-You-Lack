from .base import BasePreprocessor
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

class OUPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__("OU")
    
    def load_data(self):
        """Load OU Analyse Dataset"""
        self.logger.info("Loading OU dataset...")
        
        ou_dir = self.raw_dir / 'ou_data'
        
        # Load essential columns only
        student_info = pd.read_csv(
            ou_dir / 'studentInfo.csv',
            usecols=['id_student', 'gender', 'region', 'highest_education']
        )
        
        assessments = pd.read_csv(
            ou_dir / 'studentAssessment.csv',
            usecols=['id_student', 'id_assessment', 'score']
        )
        
        vle_data = pd.read_csv(
            ou_dir / 'studentVle.csv',
            usecols=['id_student', 'sum_click']
        ).groupby('id_student')['sum_click'].sum().reset_index()
        
        # Merge datasets
        self.raw_data = student_info.merge(assessments, on='id_student', how='left')
        self.raw_data = self.raw_data.merge(vle_data, on='id_student', how='left')
        
        self.report['original_shape'] = self.raw_data.shape
        self.report['features_before'] = self.raw_data.columns.tolist()
        
        return self.raw_data
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features for OU dataset"""
        self.logger.info("Performing feature engineering...")
        
        # Average score per student
        score_stats = df.groupby('id_student')['score'].agg(['mean', 'std', 'count']).reset_index()
        score_stats.columns = ['id_student', 'avg_score', 'performance_consistency', 'attempts_per_topic']
        
        # Merge score statistics back
        df = df.merge(score_stats, on='id_student', how='left')
        
        # Score trend based on assessment order
        student_scores = df.groupby('id_student')['score']
        first_scores = student_scores.first()
        last_scores = student_scores.last()
        attempts = student_scores.count()
        
        score_trends = (last_scores - first_scores) / attempts
        score_trends = score_trends.reset_index()
        score_trends.columns = ['id_student', 'score_trend']
        
        df = df.merge(score_trends, on='id_student', how='left')
        
        # Time efficiency (score per VLE interaction)
        df['time_efficiency'] = df['score'] / df.groupby('id_student')['sum_click'].transform('sum')
        
        # At risk flag
        df['at_risk_flag'] = (df['avg_score'] < 50).astype(int)
        
        return df
    
    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess OU dataset following the defined steps"""
        
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