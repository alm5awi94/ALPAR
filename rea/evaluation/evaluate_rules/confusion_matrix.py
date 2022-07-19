import pandas as pd


# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/tree/master/src/evaluate_rules

def confusion_matrix(actual_data, predicted_data):
    data = {'actual': actual_data,
            'predicted': predicted_data
            }

    df = pd.DataFrame(data, columns=['actual', 'predicted'])

    matrix = pd.crosstab(df['actual'], df['predicted'], rownames=['Actual'],
                         colnames=['Predicted'])
    return matrix
