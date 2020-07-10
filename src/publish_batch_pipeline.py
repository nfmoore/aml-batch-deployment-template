from argparse import ArgumentParser

from azureml.core import Datastore, Environment, Workspace
from azureml.core.dataset import Dataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import (DataTransferStep, ParallelRunConfig,
                                    ParallelRunStep)

workspace_name = 'mlwsandboxamlplatform'
subscription_id = '68be6dca-bd5e-40f5-8590-32bbfd45bdd2'
resource_group = 'sandbox-amlplatform-rg'
compute_name = 'cpu-cluster'
conda_specification_file = 'environments/conda_dependencies_score.yml'
dataset_name = 'cardiovascular_disease_dataset'
pipeline_name = 'cardiovascular-disease-batch-score'


def parse_args():
    # Parse command line arguments
    ap = ArgumentParser()

    ap.add_argument('--subscription_id', required=True)
    ap.add_argument('--resource_group', required=True)
    ap.add_argument('--workspace_name', required=True)
    ap.add_argument('--compute_name', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--pipeline_name', required=True)
    ap.add_argument('--datastore_name', required=True)
    ap.add_argument('--environment_name', default='score_env')
    ap.add_argument('--environment_specification', required=True)

    args, _ = ap.parse_known_args()
    return args


# def make_data_path(path_name, datastore):
#     name = '{}_datapath'.format(path_name)
#     default_value = DataPath(
#         datastore=datastore,
#         path_on_datastore='{}/YYYYMMDD'.format(path_name))

#     return (
#         PipelineParameter(name=name, default_value=default_value),
#         DataPathComputeBinding(mode='mount'))


def main():
    try:
        args = parse_args()

        workspace = Workspace.get(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            name=args.workspace_name,)

        compute_target = workspace.compute_targets[args.compute_name]

        environment = Environment.from_conda_specification(
            name=args.environment_name,
            file_path=args.environment_specification)

        # dataset = Dataset.get_by_name(workspace, name=args.dataset_name)
        # input_dataset = dataset.as_named_input(args.dataset_name)

        # default_datastore = workspace.get_default_datastore()
        # output_folder = PipelineData(
        #     name='score_results', datastore=default_datastore)

        datastore = Datastore(workspace, "batch_score")  # args.datastore_name)

        # input_data_path = make_data_path('input', datastore)
        # output_data_path = make_data_path('output', datastore)

        # input_folder = PipelineData(
        #     name='{}-inputs'.format(pipeline_name), datastore=datastore)

        # output_folder = PipelineData(
        #     name='{}-outputs'.format(pipeline_name), datastore=datastore)

        # input_folder = PipelineData(
        #     name='inputs', datastore=datastore)

        input_dataset = Dataset.get_by_name(
            workspace, name='batch_score_inputs')

        # input_dataset_pipeline_param = PipelineParameter(
        #     name='input_dataset_param', default_value=input_dataset)

        input_dataset_consumption = DatasetConsumptionConfig(
            name='input_data',
            dataset=input_dataset,
            mode='mount')

        output_folder = PipelineData(
            name='output_data', datastore=datastore)

        model_id_param = PipelineParameter(
            "model_id", default_value='cardiovascular-disease-model:1')

        parallel_run_config = ParallelRunConfig(
            entry_script='score.py',
            source_directory='src',
            output_action='append_row',
            append_row_file_name="predictions.txt",
            compute_target=compute_target,
            environment=environment,
            error_threshold=5,
            node_count=1)

        score_step = ParallelRunStep(
            name='score',
            inputs=[input_dataset_consumption],
            output=output_folder,
            parallel_run_config=parallel_run_config,
            arguments=['--model_id', model_id_param])

        # blob_data_ref = DataReference(
        #     datastore=blob_datastore,
        #     data_reference_name="blob_test_data",
        #     path_on_datastore="testdata")

        transfer_adls_to_blob = DataTransferStep(
            name="transfer_adls_to_blob",
            source_data_reference=input_dataset,
            compute_target=compute_target)

        pipeline = Pipeline(workspace=workspace, steps=[
                            score_step, transfer_adls_to_blob])

        pipeline.publish(name=args.pipeline_name)

    except Exception as error:
        print('Exception:', error)
        exit(1)


if __name__ == '__main__':
    main()
