import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split 
from typing import List

def partition_data(df, target_column, test_ratio=0.2, seed=42): 
    features = df.drop(columns=target_column) 
    targets = df[target_column] 
     
    train_features, val_features, train_labels, val_labels = train_test_split( 
        features, targets, test_size=test_ratio, random_state=seed, stratify=targets 
    ) 
     
    return train_features, val_features, train_labels, val_labels 
 
def remove_columns(df, columns_to_exclude): 
    return df.drop(columns=columns_to_exclude) 
 
def transform_features( 
    df: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str], apply_scaling: bool, transformer: ColumnTransformer = None): 
 
    if transformer is None: 
        numeric_pipeline = Pipeline(steps=[('scaler', MinMaxScaler())]) if apply_scaling else 'passthrough' 
        categorical_pipeline = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) 
 
        transformer = ColumnTransformer( 
            transformers=[ 
                ('num', numeric_pipeline, numeric_columns), 
                ('cat', categorical_pipeline, categorical_columns) 
            ] 
        ) 
        transformed_data = transformer.fit_transform(df) 
    else: 
        transformed_data = transformer.transform(df) 
     
    one_hot_feature_names = transformer.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_columns) 
    all_column_names = numeric_columns + list(one_hot_feature_names) 
     
    return transformed_data, all_column_names, transformer 
 
def prepare_data(raw_df, apply_scaling): 
    target_column = 'Exited' 
    columns_to_exclude = ["Surname", "CustomerId"] 

    train_features, val_features, train_labels, val_labels = partition_data(raw_df, target_column) 

    train_df = pd.concat([train_features, train_labels], axis=1) 
    val_df = pd.concat([val_features, val_labels], axis=1) 
     
    train_df = remove_columns(train_df, columns_to_exclude) 
    val_df = remove_columns(val_df, columns_to_exclude) 
     
    input_columns = list(train_df.columns)[1:-1] 
    train_inputs, train_labels = train_df[input_columns], train_df[target_column] 
    val_inputs, val_labels = val_df[input_columns], val_df[target_column] 
     
    numeric_columns = train_inputs.select_dtypes(include=np.number).columns.tolist() 
    categorical_columns = train_inputs.select_dtypes(include='object').columns.tolist() 
     
    train_inputs_transformed, all_column_names, transformer = transform_features( 
        train_inputs, numeric_columns, categorical_columns, apply_scaling 
    ) 
     
    val_inputs_transformed, _, _ = transform_features( 
        val_inputs, numeric_columns, categorical_columns, apply_scaling, transformer 
    ) 
     

    scaler = transformer.named_transformers_['num']['scaler'] if apply_scaling else None 
    encoder = transformer.named_transformers_['cat']['onehot'] 
     

    train_inputs_df = pd.DataFrame(train_inputs_transformed, columns=all_column_names) 
    val_inputs_df = pd.DataFrame(val_inputs_transformed, columns=all_column_names) 
     
    train_labels_df = pd.DataFrame(train_labels).reset_index(drop=True) 
    val_labels_df = pd.DataFrame(val_labels).reset_index(drop=True) 
     
    return train_inputs_df, train_labels_df, val_inputs_df, val_labels_df, input_columns, scaler, encoder 
 
def process_new_data(new_df: pd.DataFrame, input_columns: List[str], scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame: 
    new_df = remove_columns(new_df, ["Surname", "CustomerId"]) 
    inputs = new_df[input_columns] 
     
    numeric_columns = inputs.select_dtypes(include=np.number).columns.tolist() 
    categorical_columns = inputs.select_dtypes(include='object').columns.tolist() 
     
    numeric_pipeline = Pipeline(steps=[('scaler', scaler)]) if scaler else 'passthrough' 
    categorical_pipeline = Pipeline(steps=[('onehot', encoder)]) 
     
    transformer = ColumnTransformer( 
        transformers=[ 
            ('num', numeric_pipeline, numeric_columns), 
            ('cat', categorical_pipeline, categorical_columns) 
        ] 
    ) 
     
    transformed_inputs = transformer.fit_transform(inputs) 
    one_hot_feature_names = encoder.get_feature_names_out(categorical_columns) 
    all_column_names = numeric_columns + list(one_hot_feature_names) 
     
    transformed_inputs_df = pd.DataFrame(transformed_inputs, columns=all_column_names) 
     
    return transformed_inputs_df
