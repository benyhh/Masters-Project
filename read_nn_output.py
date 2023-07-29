import pandas as pd
import re
from IPython import embed

def save_model_results(output):
    # Initialize variables to store the parameters and minimum validation loss
    params = {}
    min_val_loss = float('inf')
    min_val_loss_epoch = -1

    # Split the output into lines
    lines = output.split('\n')

    # Extract the parameters

    params = {}
    for line in lines:
        if 'with parameters:' in line:
            # Extract the parameter string and remove the leading and trailing braces
            param_str = line.split(':')[1].strip()[1:-1]
        # Split the parameter string into key-value pairs
            param_pairs = param_str.split(', ')
            # Extract the key and value for each pair and add them to the params dictionary
            test_split = line.split('{')[1].split('\'')
            test_split = [i.replace(':', ' ') for i in test_split]
            test_split = [i.replace(',', ' ') for i in test_split]
            test_split = [i.replace('}', ' ') for i in test_split]
            test_split = [i.replace('\'', ' ') for i in test_split]
            test_split = [i.strip() for i in test_split]
            test_split = [i for i in test_split if len(i) > 0]
            
            for i,val in enumerate(test_split[::2]):
                params[val] = test_split[i*2+1]
                

    # Extract the minimum validation loss and the epoch at which it occurred
    for line in lines:
        if 'Val loss:' in line:
            # Extract the validation loss and epoch
            val_loss = re.findall(r'Val loss: ([\d.]+)', line)[0]
            val_loss = float(val_loss)
            # Update the minimum validation loss and epoch if necessary
            if val_loss < min_val_loss:
                min_val_loss = val_loss

    # Save the results to the specified file
    params['min_val'] = min_val_loss
    df = pd.DataFrame(params, index=[0])
    return df


# Example usage:
# save_model_results(output, 'results.txt')

output_file = open('out1.log', 'r')
output = output_file.read()
output = output.split('Plotting sorted predictions for')
df = pd.DataFrame()
for i in range(len(output)):
    
    df = pd.concat( [ df , save_model_results( output[i] ) ] )

#df.to_csv('./Results/RandomSearch/results_nn.csv', index=False)
embed()