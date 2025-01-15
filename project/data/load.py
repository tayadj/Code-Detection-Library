import pandas

def load(path):

    """
    Load data from .csv file
    
    Parameters: 
        path - path to .csv file
    
    Return:
        data - dataframe with uploaded data
    """

    data = pandas.read_csv(path)

    return data
