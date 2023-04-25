import argparse
import os
import pickle
import tensorflow as tf
import yaml
# Define command line arguments
parser = argparse.ArgumentParser(description='Train a model.')

parser.add_argument('--train_data', type=str, default='data/train.pkl',
                    help='Path to the training data')
parser.add_argument('--valid_data', type=str, default='data/valid.pkl',
                    help='Path to the validation data')
parser.add_argument('--model_path', type=str, default='model.h5',
                    help='Path to save the trained model')

# Parse command line arguments
args = parser.parse_args()


with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

print(params['layer1_size'])
print(params['layer2_size'])



# Load the training and validation data
with open(args.train_data, 'rb') as f:
    train_data = pickle.load(f)
with open(args.valid_data, 'rb') as f:
    valid_data = pickle.load(f)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(params['layer1_size'], activation='relu', input_shape=(train_data['inputs'].shape[1],)),
    tf.keras.layers.Dense(params['layer2_size'], activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_data['inputs'], train_data['targets'], validation_data=(valid_data['inputs'], valid_data['targets']))

# Save the model
model.save(args.model_path)

