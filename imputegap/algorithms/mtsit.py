import time

from imputegap.wrapper.AlgoPython.MTSIT.recovMTSIT import recov_mtsit


def mtsit(incomp_data, seq_len=-1, epochs=100, batch_size=-1, sliding_windows=1, d_model=16, n_heads=4, n_layers=3, d_ffn=16, masking_ratio=0.15, mean_mask_length=3, tr_ratio=0.9, seed=42, logs=True, verbose=True, **reproduction):
    """
    Perform imputation using the MTSIT algorithm (Multivariate Time Series Imputation with Transformers).

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs), shape (time steps, series).
    seq_len : int, optional
        Length of the windows processed by the transformer; -1 uses min(48, number of time steps) (default is -1).
    epochs : int, optional
        Number of training epochs (default is 100; the paper uses 400).
    batch_size : int, optional
        Number of windows per mini-batch; -1 uses min(64, number of training windows) (default is -1).
    sliding_windows : int, optional
        Stride between consecutive windows; 0 means non-overlapping windows (default is 1).
    d_model : int, optional
        Embedding width L of each time point (default is 16).
    n_heads : int, optional
        Number of attention heads per encoder block; must divide d_model (default is 4).
    n_layers : int, optional
        Number of stacked encoder blocks N (default is 3).
    d_ffn : int, optional
        Hidden width H of the position-wise feed-forward network (default is 16).
    masking_ratio : float, optional
        Expected share r of observed entries masked during training (default is 0.15).
    mean_mask_length : int, optional
        Mean length l_m of a masked segment in the geometric masks (default is 3).
    tr_ratio : float, optional
        Share of windows used for training; the rest select the best checkpoint by validation MAE (default is 0.9).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the training progress (default is True).
    **reproduction : optional
        Reproduction-mode options passed to the model (``profile``, ``normalize``,
        ``train_rows``, ``validation_rows``, ``validation_targets``); see ``recov_mtsit``.
        If omitted, the standard ImputeGAP behaviour is used.

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered; observed values are unchanged.

    Example
    -------
        >>> recov_data = mtsit(incomp_data, seq_len=48, epochs=100)
        >>> print(recov_data)

    References
    ----------
    A. Y. Yildiz, E. Koc and A. Koc, "Multivariate Time Series Imputation With Transformers," IEEE Signal Processing Letters, vol. 29, pp. 2517-2521, 2022, doi: 10.1109/LSP.2022.3224880.
    """
    start_time = time.time()  # Record start time

    recov_data = recov_mtsit(incomp_data=incomp_data, seq_len=seq_len, epochs=epochs, batch_size=batch_size, sliding_windows=sliding_windows, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, masking_ratio=masking_ratio, mean_mask_length=mean_mask_length, tr_ratio=tr_ratio, seed=seed, verbose=verbose, **reproduction)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation mtsit - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
