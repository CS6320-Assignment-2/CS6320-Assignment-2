import subprocess
import json
from tqdm import tqdm

# Define a list of hyperparameters
hyperparameters = [
    {'hidden_dim': 2, 'num_layers': 1, 'init': 'default'},
    {'hidden_dim': 32, 'num_layers': 1, 'init': 'default'},
    {'hidden_dim': 128, 'num_layers': 1, 'init': 'default'},
    {'hidden_dim': 2, 'num_layers': 2, 'init': 'default'},
    {'hidden_dim': 32, 'num_layers': 2, 'init': 'default'},
    {'hidden_dim': 128, 'num_layers': 2, 'init': 'default'},
    {'hidden_dim': 2, 'num_layers': 1, 'init': 'xavier'},
    {'hidden_dim': 32, 'num_layers': 1, 'init': 'xavier'},
    {'hidden_dim': 128, 'num_layers': 1, 'init': 'xavier'},
]

# Results will be stored in this list
results = []

# Loop over all combinations of hyperparameters with tqdm for progress bar
for params in tqdm(hyperparameters, desc="Running hyperparameter combinations"):
    # Build the command to run rnn.py with the current set of hyperparameters
    cmd = [
        'python', 'rnn.py',
        '--hidden_dim', str(params['hidden_dim']),
        '--num_layers', str(params['num_layers']),
        '--init', params['init'],
    ]
    
    # Run the command and collect output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Store the result along with the hyperparameters used
    results.append({
        'hyperparameters': params,
        'output': result.stdout,
        'error': result.stderr
    })

# Save results to disk
with open('rnn_results.json', 'w') as f:
    json.dump(results, f)

print("Finished running all hyperparameter combinations.")
