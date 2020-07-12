import json
import os
from argparse import ArgumentParser

from azureml.core import Datastore, Environment, Workspace
from azureml.core.authentication import MsiAuthentication
from azureml.core.dataset import Dataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep


def parse_args():
    # Parse command line arguments
    ap = ArgumentParser()

    ap.add_argument('--subscription_id', required=True)
    ap.add_argument('--resource_group', required=True)
    ap.add_argument('--workspace_name', required=True)
    ap.add_argument('--compute_name', required=True)
    ap.add_argument('--pipeline_name', required=True)
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--pipeline_version', required=True)
    ap.add_argument('--input_dataset_name', required=True)
    ap.add_argument('--output_datastore_name', required=True)
    ap.add_argument('--environment_specification', required=True)
    ap.add_argument('--pipeline_metadata_file', default=None)
    ap.add_argument('--environment_name', default='score_env')
    ap.add_argument('--ai_connection_string', default='')
    ap.add_argument('--msi_auth', default=False)

    args, _ = ap.parse_known_args()
    return args


def main():
    try:
        args = parse_args()

        print('args', args, args.subscription_id,
              args.resource_group, args.workspace_name)

        # Retreive workspace
        workspace = Workspace.get(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            name=args.workspace_name)

        # Retreive compute cluster
        compute_target = workspace.compute_targets[args.compute_name]

        # Setup batch scoring environment from conda dependencies
        environment = Environment.from_conda_specification(
            name=args.environment_name,
            file_path=args.environment_specification)

        # Add environment variables
        environment.environment_variables = {
            'APPLICATIONINSIGHTS_CONNECTION_STRING': args.ai_connection_string
        }

        # Retreive input dataset
        input_dataset = Dataset.get_by_name(
            workspace, name=args.input_dataset_name)

        # Define input dataset
        input_dataset = DatasetConsumptionConfig(
            name='input_dataset',
            mode='mount',
            dataset=input_dataset)

        # Retreive output datastore
        datastore = Datastore(workspace, args.output_datastore_name)

        # Define output datastore
        output_folder = PipelineData(
            name='output_datastore', datastore=datastore,
            output_name=args.pipeline_name.replace('-', '_'))

        # Define model id parameter
        model_id_param = PipelineParameter(
            "model_id", default_value=args.model_id)

        # Define configuration for parallel run step
        parallel_run_config = ParallelRunConfig(
            entry_script='score.py',
            source_directory='src',
            output_action='append_row',
            append_row_file_name="predictions.txt",
            compute_target=compute_target,
            environment=environment,
            error_threshold=5,
            node_count=1)

        # Define parallel run step for batch scoring
        score_step = ParallelRunStep(
            name='score',
            inputs=[input_dataset],
            output=output_folder,
            parallel_run_config=parallel_run_config,
            arguments=['--model_id', model_id_param])

        # Define pipeline for batch scoring
        pipeline = Pipeline(workspace=workspace, steps=[score_step])

        # Publish pipeline
        published_pipeline = pipeline.publish(name=args.pipeline_name,
                                              version=args.pipeline_version)

        # Get pipeline details
        pipeline_details = {'name': published_pipeline.name,
                            'id': published_pipeline.id,
                            'endpoint': published_pipeline.endpoint}

        # Display pipeline details
        print(pipeline_details)

        if args.pipeline_metadata_file:
            # Create directory if it does not exist
            directory = args.pipeline_metadata_file.split('/')
            os.makedirs('/'.join(directory[:-1]), exist_ok=True)

            # Write pipeline details to file
            with open(args.pipeline_metadata_file, 'w') as f:
                json.dump(pipeline_details, f)

    except Exception as error:
        print('Exception:', error)
        exit(1)


if __name__ == '__main__':
    main()
