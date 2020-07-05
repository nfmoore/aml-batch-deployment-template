from azureml.core import Dataset, Environment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep

workspace_name = 'mlwsandboxamlplatform'
subscription_id = '68be6dca-bd5e-40f5-8590-32bbfd45bdd2'
resource_group = 'sandbox-amlplatform-rg'
compute_name = 'cpu-cluster'
environment_name = 'score_env'
conda_specification_file = 'environments/conda_dependencies_score.yml'
dataset_name = 'cardiovascular_disease_dataset'
pipeline_name = 'cardiovascular-disease-batch-score'
pipeline_description = 'Pipeline for batch scoring'
max_nodes_scoring = 1


def main():
    try:
        workspace = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group)

        compute_target = workspace.compute_targets[compute_name]

        environment = Environment.from_conda_specification(
            name=environment_name, file_path=conda_specification_file)

        environment.docker.enabled = True

        run_config = RunConfiguration()
        run_config.environment = environment

        dataset = Dataset.get_by_name(workspace, name=dataset_name)
        input_dataset = dataset.as_named_input(dataset_name)

        default_datastore = workspace.get_default_datastore()
        output_folder = PipelineData(
            name='score_results', datastore=default_datastore)

        model_name_param = PipelineParameter(
            "model_name", default_value='cardiovascular-disease-model')

        version_param = PipelineParameter(
            "version", default_value='latest')

        parallel_run_config = ParallelRunConfig(
            entry_script='score.py',
            source_directory='src',
            output_action='append_row',
            compute_target=compute_target,
            node_count=max_nodes_scoring,
            environment=environment,
            error_threshold=5)

        score_step = ParallelRunStep(
            name='score',
            inputs=[input_dataset],
            output=output_folder,
            arguments=['--model_name', model_name_param,
                       '--version', version_param],
            parallel_run_config=parallel_run_config)

        pipeline = Pipeline(workspace=workspace, steps=[score_step])

        pipeline.publish(name=pipeline_name)

    except Exception as error:
        print('Exception:', error)
        exit(1)


if __name__ == '__main__':
    main()
