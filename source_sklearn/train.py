from __future__ import print_function

import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #self parsed-arguments: 
    parser.add_argument('--n_estimators', type = int, default = 150, help = 'no. of estimators (default:150)')
    parser.add_argument('--max_depth', type = int, default= 10, help = 'max depth (default:10)')
    parser.add_argument('--min_samples_split', type = int, default = 4, help = 'minimum samples to split (default: 4')
    parser.add_argument('--random_state', type = int, default = 42, help='random state (default:42)')
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    n_est = args.n_estimators
    m_d = args.max_depth
    m_s_s = args.min_samples_split
    rnd_st = args.random_state

    ## TODO: Define a model 
    model = RandomForestClassifier(n_estimators=n_est, max_depth=m_d, min_samples_split = m_s_s, random_state = rnd_st)
    
    
    ## TODO: Train the model
    model.fit(train_x,train_y)
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))