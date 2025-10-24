from .base import BasePreprocessor
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

class AIPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__("AI")
    
    def load_data(self):
        """Load AI Course Performance Dataset"""
        self.logger.info("Loading AI Course dataset...")
        
        ai_dir = self.raw_dir / 'ai_course_data'
        ai_file = ai_dir / 'Student Performance Dataset in AI course' / 'Stu_Performance_dataset.csv'
        
        self.raw_data = pd.read_csv(ai_file)
        self.report['original_shape'] = self.raw_data.shape
        self.report['features_before'] = self.raw_data.columns.tolist()
        
        return self.raw_data
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features for AI Course dataset"""
        self.logger.info("Performing feature engineering...")
        
        # Calculate average score across all assessments
        assessment_columns = ['Quiz ', 'Midterm', 'Assignment_1', 'Assignment_2', 'Assignment_3', 
                            'Project', 'Presentation', 'Final_Exam']
        
        df['avg_score'] = df[assessment_columns].mean(axis=1)
        
        # Score trend (difference between final exam and early assessments)
        early_assessments = ['Quiz ', 'Midterm', 'Assignment_1']
        df['early_score'] = df[early_assessments].mean(axis=1)
        df['score_trend'] = (df['Final_Exam'] - df['early_score']) / len(assessment_columns)
        
        # Performance consistency
        df['performance_consistency'] = df[assessment_columns].std(axis=1)
        
        # At risk flag
        df['at_risk_flag'] = (df['avg_score'] < 50).astype(int)
        
        # Count of assignments submitted (assuming NaN means not submitted)
        df['attempts_per_topic'] = df[assessment_columns].notna().sum(axis=1)
        
        # Time efficiency (not available in this dataset, but we'll create a proxy)
        max_scores = df[assessment_columns].max()
        df['time_efficiency'] = df[assessment_columns].sum(axis=1) / (max_scores.sum())
        
        return df
    
    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess AI Course dataset following the defined steps"""
        
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