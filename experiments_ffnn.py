import subprocess
import json
from tqdm import tqdm

# Define a list of hyperparameters
hyperparameters = [
    {'hidden_dim': 1, 'init': 'default'},
    {'hidden_dim': 2, 'init': 'default'},
    {'hidden_dim': 5, 'init': 'default'},
    {'hidden_dim': 10, 'init': 'default'},
    {'hidden_dim': 20, 'init': 'default'},
    {'hidden_dim': 50, 'init': 'default'},
    {'hidden_dim': 100, 'init': 'default'},
    {'hidden_dim': 1, 'init': 'xavier'},
    {'hidden_dim': 2, 'init': 'xavier'},
    {'hidden_dim': 5, 'init': 'xavier'},
    {'hidden_dim': 10, 'init': 'xavier'},
    {'hidden_dim': 20, 'init': 'xavier'},
    {'hidden_dim': 50, 'init': 'xavier'},
    {'hidden_dim': 100, 'init': 'xavier'},
    {'hidden_dim': 1, 'init': 'kaiming'},
    {'hidden_dim': 2, 'init': 'kaiming'},
    {'hidden_dim': 5, 'init': 'kaiming'},
    {'hidden_dim': 10, 'init': 'kaiming'},
    {'hidden_dim': 20, 'init': 'kaiming'},
    {'hidden_dim': 50, 'init': 'kaiming'},
    {'hidden_dim': 100, 'init': 'kaiming'},
]

# Results will be stored in this list
results = []

# Loop over all combinations of hyperparameters with tqdm for progress bar
for params in tqdm(hyperparameters, desc="Running hyperparameter combinations"):
    # Build the command to run ffnn.py with the current set of hyperparameters
    cmd = [
        'python', 'ffnn.py',
        '--hidden_dim', str(params['hidden_dim']),
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
with open('ffnn_results.json', 'w') as f:
    json.dump(results, f)

print("Finished running all hyperparameter combinations.")
