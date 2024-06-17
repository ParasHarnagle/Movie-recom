import mlflow
from mlflow.tracking import MlflowClient

def deploy_model(model_uri, model_name):
    """
    Deploy yrained model to mlflow model registry
    :param model_uri:
    :param model_name:
    :return:
    """

    client = MlflowClient()
    client.create_registered_model(name=model_name)
    client.create_model_version(name=model_name,
                                source=model_uri,
                                run_id=mlflow.active_run().info.run_id)
    client.transition_model_version_stage(
        name=model_name,version=1,stage='Production'
    )