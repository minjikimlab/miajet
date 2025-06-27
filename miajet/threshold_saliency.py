
import pandas as pd
import numpy as np

def threshold_saliency_q(df_agg, ranking, q, verbose):
    """
    Filter the ridges based on a percentile threshold of the saliency values

    Note that the percentile is computed only on the non-zero saliency values
    
    Parameters
    ----------
    df_agg : pd.DataFrame
        Summary dataframe containing the `ranking` column 
    ranking : str
        The column name containing jet saliency values to be used for filtering
    q : float
        The percentile (0-100) of the saliency values to use as a threshold,
        above which we keep
    
    Returns
    -------
    pd.DataFrame
        Updated summary dataframe with ridges above the percentile saliency value
    """

    if df_agg.empty:
        return df_agg

    # get the non-zero saliency values only
    values = df_agg[ranking].values
    non_zero_saliency = values[~np.isclose(values, 0)]

    if len(non_zero_saliency) == 0:
        if verbose:
            print("\tNo non-zero saliency values found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Get the qth percentile saliency value 
    saliency_threshold = np.percentile(non_zero_saliency, q=q)
    
    # Threshold
    df_agg_filtered = df_agg[df_agg[ranking] > saliency_threshold].reset_index(drop=True)
    
    if verbose:
        print(f"\tNumber of ridges after {q} percentile saliency thresholding: {len(df_agg_filtered)} out of {len(df_agg)}...")
    
    return df_agg_filtered







