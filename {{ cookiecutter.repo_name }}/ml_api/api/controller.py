from flask import Blueprint, request, jsonify
from src.predict import make_prediction
from src import __version__ as _version


from api.config import get_logger
from api.validation import validate_inputs
from api import __version__ as api_version


_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)

@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/regression', methods=['POST'])
def predict():
    """
    The second step is optional and it isn't necessary, you can comment
    that lines of code, but if you want to have a robust control of 
    the data that people post in your API you have to include in a correct
    format in the same way that the example in validation.py
    """
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = make_prediction(input_data=input_data)
        _logger.debug(f'Outputs: {result}')

        # Step 4: Convert numpy ndarray to list
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})
