import aws_cdk as core
import aws_cdk.assertions as assertions

from ml_training_inference_workflow.ml_training_inference_workflow_stack import MlTrainingInferenceWorkflowStack

# example tests. To run these tests, uncomment this file along with the example
# resource in ml_training_inference_workflow/ml_training_inference_workflow_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MlTrainingInferenceWorkflowStack(app, "ml-train-inference-workflow")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
