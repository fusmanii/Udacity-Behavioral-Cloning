import argparse
from keras.models import load_model

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'model',
    type=str,
    help='Path to model h5 file. Model should be on the same path.'
)
args = parser.parse_args()
model = load_model(args.model)

# Save model data
model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)