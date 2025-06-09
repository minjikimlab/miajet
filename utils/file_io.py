import pandas as pd
import json
import numpy as np

def round_and_json(x, dp):
    """
    Round a numeric array and convert to json using dumps
    """
    x_dp = np.round(x, decimals=dp)
    return json.dumps(x_dp.tolist())

def save_csv(df_in, save_name, root, parameter_str, dp=3, exclude_rounding=["s_imagej"], convert_json=None):
    """
    Save a dataframe to a csv file 

    Parameters
    ----------
    df_in : pd.DataFrame
        The input dataframe to save
    save_name : str
        The name of the file to save the dataframe to
    root : str
        The root path to include in the file name
    parameter_str : str
        A string containing parameters to be included in the beginning of the csv file as comments
    dp : int, optional
        The number of decimal places to round numeric columns to, by default 3
    exclude_rounding : list, optional
        A list of column names to exclude from rounding, by default ["s_imagej"]
    convert_json : list, optional
        A list of column names to convert to json format, by default None
        This is useful for numpy arrays that need to be saved as json strings
    
    Returns
    -------
    None, but saves the dataframe to a csv file
    """
    df = df_in.copy()

    rounding_dict = {col: dp for col in df.select_dtypes(include="number").columns if col not in exclude_rounding}

    # round first
    # print("\tNOTE: Disabling rounding for columns")
    df = df.round(rounding_dict)

    # convert any numpy arrays to a json format string
    if convert_json is not None:
        for col in convert_json:
            df[col] = df[col].apply(lambda x : round_and_json(x, dp)) # round and convert to json

    # add # to each line of parameter_str 
    lines = parameter_str.splitlines()
    parameter_str_comment = "\n".join([f"# {line}" for line in lines])

    with open(save_name, "w") as f:
        f.write(f"# {root}\n")
        f.write(parameter_str_comment + "\n")
        df.to_csv(f, index=False)

import tifffile
import cv2 as cv
import os 

def save_tensor_as_tiff(tensor, save_path, file_name, perc_vmax):
    """
    Save a numpy tensor as a tiff file
    """
    if perc_vmax is not None:

        tensor_norm = np.clip(tensor, None, np.percentile(tensor, perc_vmax))

        print("Before normalization: min =", tensor_norm.min(), "max =", tensor_norm.max())

    else:

        tensor_norm = np.copy(tensor)

    if np.issubdtype(tensor_norm.dtype, bool):
        tensor_norm = tensor_norm.astype(int)

    # normalize first
    tensor_norm = cv.normalize(tensor_norm, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    save_name = os.path.join(save_path, file_name)

    tifffile.imwrite(save_name, tensor_norm, imagej=True, metadata={'axes': 'ZYX'})