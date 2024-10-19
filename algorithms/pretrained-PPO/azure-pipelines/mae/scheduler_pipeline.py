import argparse
import glob
import os
from pathlib import Path

import pandas as pd
from azure.ai.ml import MLClient, Input, pipeline
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from mldesigner import command_component, Input, Output
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """start scheduler MAE training pipeline
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-job_data", type=str, help="Path to input job_data", default='<YOUR_INPUT_DATA_PATH>')
    parser.add_argument('-id', type=str, help="Azure ML workspace id", default='<YOUR_SUBSCRIPTION_ID>')
    parser.add_argument('-rg', type=str, help="Azure ML resource group", default='<YOUR_RESOURCE_GROUP>')
    parser.add_argument('-name', type=str, help="Azure ML workspace name", default='<YOUR_WORKSPACE_NAME>')
    args = parser.parse_args()

    subscription_id = args.id
    resource_group = args.rg
    workspace_name = args.name

    credential = DefaultAzureCredential()

    ml_client = MLClient(TokenCredential(credentials), subscription_id, resource_group, workspace_name)


    @command_component(
        name="prep_data",
        version="1",
        display_name="Prep Data",
        description="Convert job_data to CSV file, and split to training and test job_data",
        environment=dict(
            conda_file=Path(__file__).parent / "env.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        ),
    )
    def preprocess_component(input_data: Input(type="uri_folder"), preprocessed_data: Output(type="uri_folder")):
        raw_data_files = glob.glob(os.path.join(input_data, "*.swf"))
        all_data = []

        # Load and preprocess SWF job_data from all files
        for raw_data_path in raw_data_files:
            with open(raw_data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith(';'):
                        continue
                    values = line.split()
                    all_data.append({
                        'job_num': int(values[0]),
                        'submit_time': int(values[1]),
                        'wait_time': int(values[2]),
                        'run_time': int(values[3]),
                        'num_procs': int(values[4]),
                        'average_cpu_time': int(values[5]),
                        'used_memory': int(values[6]),
                        'req_procs': int(values[7]),
                        'req_time': int(values[8]),
                        'req_memory': int(values[9]),
                        'status': int(values[10]),
                        'user_id': int(values[11]),
                        'group_id': int(values[12]),
                        'exec_id': int(values[13]),
                        'queue_num': int(values[14]),
                        'partition_num': int(values[15]),
                        'preceding_job_nums': int(values[16]),
                        'think_time_from_preceding_job': int(values[17])
                    })

        df = pd.DataFrame(all_data)
        df.dropna(subset=['wait_time', 'run_time'], inplace=True)  # Drop jobs with missing times
        df['duration'] = df['run_time'] - df['wait_time']
        features = df[['job_num', 'submit_time', 'wait_time', 'run_time', 'num_procs', 'average_cpu_time', 'used_memory',
                       'req_procs', 'req_time', 'req_memory', 'status', 'user_id', 'group_id', 'exec_id', 'queue_num',
                       'partition_num', 'preceding_job_nums', 'think_time_from_preceding_job', 'duration']]

        # Save the combined dataset for training
        train_df, test_df = train_test_split(features, test_size=0.2, random_state=42)

        train_data_path = os.path.join(preprocessed_data, "train_traces.csv")
        test_data_path = os.path.join(preprocessed_data, "test_traces.csv")

        train_df.to_csv(train_data_path, index=False)
        test_df.to_csv(test_data_path, index=False)

        return train_data_path, test_data_path

    @command_component(
        name="train_mae",
        version="1",
        display_name="Train MAE",
        description="Training Masked Autoencoder Model on job-scheduling job_data",
        environment=dict(
            conda_file=Path(__file__).parent / "env.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        ),
    )
    def train_mae_component(preprocessed_data: Input(type="uri_folder"), trained_model: Output(type="uri_folder")):
        train_mae_path = Path(__file__).parent.parent / "mae" / "submitit_pretrain.py"
        os.system(
            f"python {train_mae_path} --job_dir {preprocessed_data} --nodes 2 --use_volta32 --batch_size 64 "
            f"--model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 1000 --blr 1.5e-4 --weight_decay 0.05 --data_path {preprocessed_data}"
        )
        return trained_model


    @pipeline(name="scheduler_mae_pipeline")
    def mae_pipeline(input_data_path: str):
        train_path, test_path = preprocess_component(input_data=Input(path=input_data_path))
        train_step = train_mae_component(preprocessed_data=Input(path=train_path))
        return {"trained_model": train_step.outputs.trained_model}


    pipeline_job = mae_pipeline(input_data_path=args.data)
    pipeline_job = ml_client.jobs.create_or_update(pipeline_job)

    print(f"Pipeline submitted successfully. Job ID: {pipeline_job.name}")
