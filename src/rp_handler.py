"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


APP = predict.Predictor()
APP.setup()


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    with rp_debugger.LineTimer('prediction_step'):
        whisper_results = APP(
            audio=job_input['audio'],
            model_name=job_input['model_name'],
            language=job_input["language"],
            batch_size=job_input["batch_size"],
        )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
