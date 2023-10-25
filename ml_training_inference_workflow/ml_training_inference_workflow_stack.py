from aws_cdk import (
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    RemovalPolicy,
    Duration,
    Stack,
    Size,
)
from constructs import Construct


class MlTrainingInferenceWorkflowStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # create buckets with sample training and inference data
        source_bucket = s3.Bucket(self, "SourceBucket", removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)
        target_bucket = s3.Bucket(self, "TargetBucket", removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)
        source_deployment = s3deploy.BucketDeployment(
            self,
            "SourceDeployment",
            sources=[
                s3deploy.Source.asset("images/local_test/test_dir/input/data/training")
            ],
            destination_bucket=source_bucket,
        )
        target_deployment = s3deploy.BucketDeployment(
            self,
            "TargetDeployment",
            sources=[
                s3deploy.Source.asset("images/local_test/test_dir/input/data/inference")
            ],
            destination_bucket=target_bucket,
            destination_key_prefix="inference/",
        )

        # Define the custom image for sagemaker
        custom_image = tasks.DockerImage.from_asset(
            self,
            "TrainImage",
            directory="./images/",
            platform=ecr_assets.Platform.LINUX_AMD64,
        )
        init_state = sfn.Pass(
            self,
            "InitState",
            parameters={
                "SourceBucketUri": sfn.JsonPath.format(
                    "s3://{}/{}",
                    source_bucket.bucket_name,
                    sfn.JsonPath.string_at("$.Key"),
                ),
                "BucketName": source_bucket.bucket_name,
                "Key": sfn.JsonPath.string_at("$.Key"),
            },
        )
        # Define the training job task
        # To use enable_managed_spot_training whe have to create a custom task
        create_training_job = tasks.SageMakerCreateTrainingJob(
            self,
            "CreateTrainingJob",
            training_job_name=sfn.JsonPath.uuid(),
            algorithm_specification={
                "training_image": custom_image,
                "training_input_mode": tasks.InputMode.FILE,
            },
            input_data_config=[
                tasks.Channel(
                    channel_name="training",
                    data_source=tasks.DataSource(
                        s3_data_source=tasks.S3DataSource(
                            s3_data_type=tasks.S3DataType.S3_PREFIX,
                            s3_location=tasks.S3Location.from_json_expression(
                                "$.SourceBucketUri"
                            ),
                        )
                    ),
                )
            ],
            output_data_config=tasks.OutputDataConfig(
                s3_output_location=tasks.S3Location.from_bucket(
                    target_bucket,
                    "train_results",
                )
            ),
            resource_config={
                "instance_count": 1,
                "instance_type": ec2.InstanceType.of(
                    ec2.InstanceClass.M5,
                    ec2.InstanceSize.LARGE,
                ),  # ml.m6i.xlarge, ml.trn1.32xlarge, ml.p2.xlarge, ml.m5.4xlarge, ml.m4.16xlarge, ml.m6i.12xlarge, ml.p5.48xlarge, ml.m6i.24xlarge, ml.p4d.24xlarge, ml.g5.2xlarge, ml.c5n.xlarge, ml.p3.16xlarge, ml.m5.large, ml.m6i.16xlarge, ml.p2.16xlarge, ml.g5.4xlarge, ml.c4.2xlarge, ml.c5.2xlarge, ml.c6i.32xlarge, ml.c4.4xlarge, ml.c6i.xlarge, ml.g5.8xlarge, ml.c5.4xlarge, ml.c6i.12xlarge, ml.c5n.18xlarge, ml.g4dn.xlarge, ml.c6i.24xlarge, ml.g4dn.12xlarge, ml.c4.8xlarge, ml.g4dn.2xlarge, ml.c6i.2xlarge, ml.c6i.16xlarge, ml.c5.9xlarge, ml.g4dn.4xlarge, ml.c6i.4xlarge, ml.c5.xlarge, ml.g4dn.16xlarge, ml.c4.xlarge, ml.trn1n.32xlarge, ml.g4dn.8xlarge, ml.c6i.8xlarge, ml.g5.xlarge, ml.c5n.2xlarge, ml.g5.12xlarge, ml.g5.24xlarge, ml.c5n.4xlarge, ml.trn1.2xlarge, ml.c5.18xlarge, ml.p3dn.24xlarge, ml.m6i.2xlarge, ml.g5.48xlarge, ml.g5.16xlarge, ml.p3.2xlarge, ml.m6i.4xlarge, ml.m5.xlarge, ml.m4.10xlarge, ml.c5n.9xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, ml.m6i.8xlarge, ml.m6i.large, ml.p2.8xlarge, ml.m5.2xlarge, ml.m6i.32xlarge, ml.p4de.24xlarge, ml.p3.8xlarge, ml.m4.4xlarge
                "volume_size": Size.gibibytes(10),
            },
            result_path="$.SageMakerTrainingOutput",
            result_selector={
                "TrainingJobName": sfn.JsonPath.array_get_item(
                    sfn.JsonPath.string_split(
                        sfn.JsonPath.string_at("$.TrainingJobArn"), "/"
                    ),
                    1,
                ),
                "ModelS3Location": sfn.JsonPath.format(
                    "s3://{}/train_results/{}/output/model.tar.gz",
                    target_bucket.bucket_name,
                    sfn.JsonPath.array_get_item(
                        sfn.JsonPath.string_split(
                            sfn.JsonPath.string_at("$.TrainingJobArn"), "/"
                        ),
                        1,
                    ),
                ),
            },
        )
        create_model = tasks.SageMakerCreateModel(
            self,
            "CreateModel",
            model_name=sfn.JsonPath.uuid(),
            primary_container=tasks.ContainerDefinition(
                image=custom_image,
                mode=tasks.Mode.SINGLE_MODEL,
                model_s3_location=tasks.S3Location.from_json_expression(
                    "$.SageMakerTrainingOutput.ModelS3Location"
                ),
            ),
        )

        # Create a loop using Choice and Wait states
        wait_train_state = sfn.Wait(
            self,
            "WaitForTrainingJob",
            time=sfn.WaitTime.duration(
                Duration.seconds(60)
            ),  # Adjust the polling interval as needed
        )

        check_training_status = tasks.CallAwsService(
            self,
            "CheckTrainingJob",
            service="sagemaker",
            action="describeTrainingJob",
            iam_resources=["*"],
            parameters={
                "TrainingJobName": sfn.JsonPath.string_at(
                    "$.SageMakerTrainingOutput.TrainingJobName"
                )
            },
            result_path="$.SageMakerTrainingOutput",
            result_selector={
                "TrainingJobName": sfn.JsonPath.string_at("$.TrainingJobName"),
                "ModelS3Location": sfn.JsonPath.format(
                    "{}/{}/output/model.tar.gz",
                    sfn.JsonPath.string_at("$.OutputDataConfig.S3OutputPath"),
                    sfn.JsonPath.string_at("$.TrainingJobName"),
                ),
                "TrainingJobStatus": sfn.JsonPath.string_at("$.TrainingJobStatus"),
            },
        )

        training_complete = sfn.Choice(
            self,
            "IsTrainingComplete",
        )

        # If the training job is incomplete, loop back to the wait state
        training_complete.when(
            sfn.Condition.string_equals(
                "$.SageMakerTrainingOutput.TrainingJobStatus", "InProgress"
            ),
            wait_train_state,
        )
        # If the training job is complete, end the loop
        training_complete.when(
            sfn.Condition.string_equals(
                "$.SageMakerTrainingOutput.TrainingJobStatus", "Completed"
            ),
            create_model,
        )

        batch_inference = tasks.SageMakerCreateTransformJob(
            self,
            "CreateBatchInferenceJob",
            transform_job_name=sfn.JsonPath.uuid(),
            model_name=sfn.JsonPath.array_get_item(
                sfn.JsonPath.string_split(sfn.JsonPath.string_at("$.ModelArn"), "/"),
                1,
            ),
            model_client_options=tasks.ModelClientOptions(
                invocations_max_retries=3,  # default is 0
                invocations_timeout=Duration.minutes(5),
            ),
            transform_input={
                "content_type": "text/csv",
                "transform_data_source": {
                    "s3_data_source": {
                        "s3_uri": sfn.JsonPath.format(
                            f"s3://{target_bucket.bucket_name}" + "/inference/{}/",
                            sfn.JsonPath.array_get_item(
                                sfn.JsonPath.string_split(
                                    sfn.JsonPath.string_at("$$.Execution.Input.Key"),
                                    ".",
                                ),
                                0,
                            ),
                        ),
                        "s3_data_type": tasks.S3DataType.S3_PREFIX,
                    }
                },
            },
            transform_output=tasks.TransformOutput(
                s3_output_path=f"s3://{target_bucket.bucket_name}/inference_results/"
            ),
            transform_resources=tasks.TransformResources(
                instance_count=1,
                instance_type=ec2.InstanceType.of(
                    ec2.InstanceClass.M5, ec2.InstanceSize.LARGE
                ),
            ),  # ml.m5.4xlarge, ml.p2.xlarge, ml.m4.16xlarge, ml.m5.large, ml.p3.16xlarge, ml.p2.16xlarge, ml.c4.2xlarge, ml.c5.2xlarge, ml.c4.4xlarge, ml.c5.4xlarge, ml.g4dn.xlarge, ml.g4dn.12xlarge, ml.g4dn.2xlarge, ml.c4.8xlarge, ml.g4dn.4xlarge, ml.c5.9xlarge, ml.g4dn.16xlarge, ml.c5.xlarge, ml.c4.xlarge, ml.g4dn.8xlarge, ml.c5.18xlarge, ml.p3.2xlarge, ml.m5.xlarge, ml.m4.10xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, ml.m5.2xlarge, ml.p2.8xlarge, ml.p3.8xlarge, ml.m4.4xlarge
            result_path="$.SageMakerInferenceOutput",
            result_selector={
                "TransformJobArn": sfn.JsonPath.string_at("$.TransformJobArn"),
            },
        )

        # Create a loop using Choice and Wait states
        wait_inference_state = sfn.Wait(
            self,
            "WaitForInferenceJob",
            time=sfn.WaitTime.duration(
                Duration.seconds(60)
            ),  # Adjust the polling interval as needed
        )

        check_inference_status = tasks.CallAwsService(
            self,
            "CheckInferenceJob",
            service="sagemaker",
            action="describeTransformJob",
            iam_resources=["*"],
            parameters={
                "TransformJobName": sfn.JsonPath.array_get_item(
                    sfn.JsonPath.string_split(
                        sfn.JsonPath.string_at(
                            "$.SageMakerInferenceOutput.TransformJobArn"
                        ),
                        "/",
                    ),
                    1,
                )
            },
            result_path="$.SageMakerInferenceOutput",
            result_selector={
                "TrainingJobName": sfn.JsonPath.string_at("$.TransformJobName"),
                "ModelName": sfn.JsonPath.string_at("$.ModelName"),
                "TransformJobStatus": sfn.JsonPath.string_at("$.TransformJobStatus"),
                "TransformJobArn": sfn.JsonPath.string_at("$.TransformJobArn"),
            },
        )

        delete_model = tasks.CallAwsService(
            self,
            "DeleteModel",
            service="sagemaker",
            action="deleteModel",
            iam_resources=["*"],
            parameters={
                "ModelName": sfn.JsonPath.array_get_item(
                    sfn.JsonPath.string_split(
                        sfn.JsonPath.string_at("$.ModelArn"), "/"
                    ),
                    1,
                )
            },
        )

        inference_complete = sfn.Choice(
            self,
            "IsInferenceComplete",
        )

        # If the training job is incomplete, loop back to the wait state
        inference_complete.when(
            sfn.Condition.string_equals(
                "$.SageMakerInferenceOutput.TransformJobStatus", "InProgress"
            ),
            wait_inference_state,
        )
        # If the training job is complete, end the loop
        inference_complete.when(
            sfn.Condition.string_equals(
                "$.SageMakerInferenceOutput.TransformJobStatus", "Completed"
            ),
            delete_model,
        )

        create_training_job.add_retry(jitter_strategy=sfn.JitterType.FULL, max_attempts=10)
        batch_inference.add_retry(jitter_strategy=sfn.JitterType.FULL, max_attempts=10)
        create_model.add_retry(max_attempts=10)
        delete_model.add_retry(max_attempts=10)

        init_state.next(create_training_job)
        create_training_job.next(wait_train_state)
        wait_train_state.next(check_training_status)
        check_training_status.next(training_complete)
        create_model.next(batch_inference)
        batch_inference.next(wait_inference_state)
        wait_inference_state.next(check_inference_status)
        check_inference_status.next(inference_complete)

        state_json = {
            "Type": "Map",
            "ItemProcessor": {
                "ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "STANDARD"},
                "StartAt": "InitState",
                "States": {
                    "InitState": init_state.to_state_json(),
                    "CreateTrainingJob": create_training_job.to_state_json(),
                    "WaitForTrainingJob": wait_train_state.to_state_json(),
                    "CheckTrainingJob": check_training_status.to_state_json(),
                    "IsTrainingComplete": training_complete.to_state_json(),
                    "CreateModel": create_model.to_state_json(),
                    "CreateBatchInferenceJob": batch_inference.to_state_json(),
                    "WaitForInferenceJob": wait_inference_state.to_state_json(),
                    "CheckInferenceJob": check_inference_status.to_state_json(),
                    "IsInferenceComplete": inference_complete.to_state_json(),
                    "DeleteModel": delete_model.to_state_json(),
                },
            },
            "ItemReader": {
                "Resource": "arn:aws:states:::s3:listObjectsV2",
                "Parameters": {
                    "Bucket": source_bucket.bucket_name,
                },
            },
            "MaxConcurrency": 2,
            "Label": "S3objectkeys",
        }

        # custom state which represents a task to insert data into DynamoDB
        custom = sfn.CustomState(self, "DistributedMap", state_json=state_json)

        # Create a Step Functions state machine
        definition = sfn.Chain.start(custom)

        ml_orchestration = sfn.StateMachine(
            self,
            "MLTraininingInference",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.minutes(30),
        )
        ml_orchestration.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))
        ml_orchestration.add_to_role_policy(iam.PolicyStatement(
            actions=["states:StartExecution"],
            resources=[f"arn:aws:states:*:{self.account}:stateMachine:*"]
        ))

        source_bucket.grant_read_write(ml_orchestration)
        target_bucket.grant_read_write(ml_orchestration)
        source_bucket.grant_read_write(create_training_job.role)
        target_bucket.grant_read_write(create_training_job.role)
        target_bucket.grant_read_write(create_model.role)
