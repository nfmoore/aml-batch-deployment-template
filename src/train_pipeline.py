from azureml.core import Dataset, Environment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

workspace_name = 'mlwsandboxamlplatform'
subscription_id = '68be6dca-bd5e-40f5-8590-32bbfd45bdd2'
resource_group = 'sandbox-amlplatform-rg'
compute_name = 'cpu-cluster'
environment_name = 'train_env'
conda_specification_file = 'environments/conda_dependencies_train.yml'
dataset_name = 'cardiovascular_disease_dataset'
pipeline_name = 'cardiovascular-disease-train'
pipeline_description = 'Pipeline for model training'


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

        train_step = PythonScriptStep(
            name='train_model',
            script_name='train.py',
            source_directory='src',
            compute_target=compute_target,
            runconfig=run_config,
            inputs=[input_dataset])

        pipeline = Pipeline(workspace=workspace, steps=[train_step])

        pipeline.publish(name=pipeline_name)

    except Exception as error:
        print(error)
        exit(1)


if __name__ == '__main__':
    main()
