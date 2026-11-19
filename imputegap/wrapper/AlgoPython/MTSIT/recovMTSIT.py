"""MTSIT imputation for ImputeGAP.

Self-contained implementation of MTSIT (Yildiz, Koc and Koc, "Multivariate
Time Series Imputation With Transformers", IEEE Signal Processing Letters 29,
2022): a linear input projection with learnable positional encodings, a stack
of post-BatchNorm transformer encoder blocks, and a linear decoder, trained by
reconstructing artificially masked *observed* entries (geometric masks).

The model is written from the paper's equations; no code from the authors'
repository (which carries no licence) is included.

Data convention (ImputeGAP): ``incomp_data`` has shape ``(T, N)`` = time steps
x series, with missing values encoded as ``np.nan``. The returned matrix has
the same shape; only the ``NaN`` entries are replaced, observed values are
returned unchanged.
"""

import math
import warnings

import numpy as np
import torch
from torch import nn


class _EncoderBlock(nn.Module):
    """Post-norm block: X_BN = BN(X + MSA(X)); out = BN(X_BN + FFN(X_BN))."""

    def __init__(self, d_model, n_heads, d_ffn, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm_attention = nn.BatchNorm1d(d_model)
        self.norm_ffn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        # Bidirectional attention over time (no causal mask): a time point may
        # use context before and after a gap.
        attended, _ = self.attention(values, values, values, need_weights=False)
        values = self.norm_attention(
            (values + self.dropout(attended)).transpose(1, 2)
        ).transpose(1, 2)
        return self.norm_ffn(
            (values + self.dropout(self.ffn(values))).transpose(1, 2)
        ).transpose(1, 2)


class MTSITNetwork(nn.Module):
    """Input projection + learnable positions -> N encoder blocks -> linear decoder.

    Inputs and outputs have shape ``(batch, seq_len, n_series)``.
    """

    def __init__(self, n_series, seq_len, d_model, n_heads, n_layers, d_ffn, dropout):
        super().__init__()
        self.projection = nn.Linear(n_series, d_model)
        self.position = nn.Parameter(torch.empty(1, seq_len, d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )
        self.decoder = nn.Linear(d_model, n_series)

    def forward(self, inputs):
        hidden = self.dropout(self.projection(inputs) + self.position)
        for block in self.blocks:
            hidden = block(hidden)
        return self.decoder(hidden)


def _window_starts(n_steps, seq_len, stride):
    """Start indices covering every time step; the last window ends at T."""
    starts = list(range(0, n_steps - seq_len + 1, stride))
    if starts[-1] != n_steps - seq_len:
        starts.append(n_steps - seq_len)
    return np.asarray(starts, dtype=int)


def _geometric_mask(rng, seq_len, n_series, ratio, mean_length):
    """Independent two-state Markov chain per series along time.

    Masked segments have mean length ``mean_length`` (l_m); unmasked segments
    have mean length l_u = (1 - r) l_m / r, so the expected masked fraction is r.
    """
    hidden = np.empty((seq_len, n_series), dtype=bool)
    state = rng.random(n_series) < ratio
    leave_masked = 1.0 / mean_length
    leave_unmasked = ratio / ((1.0 - ratio) * mean_length)
    for step in range(seq_len):
        hidden[step] = state
        flip = rng.random(n_series) < np.where(state, leave_masked, leave_unmasked)
        state = state ^ flip
    return hidden


def _masked_batch(values, observed, window_ids, epoch, seed, ratio, mean_length, stream=0):
    """Model inputs and target masks for one batch of windows.

    ``stream`` separates training masks (0, fresh per epoch) from the fixed
    validation masks (1); each window has its own reproducible random stream.
    """
    masks = np.stack(
        [
            _geometric_mask(
                np.random.default_rng(
                    np.random.SeedSequence([seed, stream, epoch, int(w)])
                ),
                values.shape[1],
                values.shape[2],
                ratio,
                mean_length,
            )
            for w in window_ids
        ]
    ) & observed[window_ids]
    inputs = np.where(observed[window_ids] & ~masks, values[window_ids], 0.0)
    return inputs, masks


def recov_mtsit(
    incomp_data,
    seq_len=-1,
    epochs=100,
    batch_size=-1,
    sliding_windows=1,
    d_model=16,
    n_heads=4,
    n_layers=3,
    d_ffn=16,
    masking_ratio=0.15,
    mean_mask_length=3,
    dropout=0.1,
    lr=1e-3,
    clip_norm=4.0,
    tr_ratio=0.9,
    seed=42,
    device=None,
    verbose=False,
):
    """Impute the NaN entries of ``incomp_data`` (shape ``(T, N)``) with MTSIT.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        Matrix of shape (T, N) with missing values as NaN.
    seq_len : int
        Window length L_T in time steps; -1 uses min(48, T).
    epochs : int
        Training epochs (the paper uses 400; ImputeGAP default 100 for speed).
    batch_size : int
        Windows per mini-batch; -1 uses min(64, number of training windows).
    sliding_windows : int
        Stride between consecutive windows (default 1: all overlapping windows);
        0 means non-overlapping windows (stride = seq_len).
    d_model, n_heads, n_layers, d_ffn : int
        Embedding width L, attention heads h, encoder blocks N and FFN hidden width H
        (paper: 16, 4, 3, 16). ``d_model`` must be divisible by ``n_heads``.
    masking_ratio, mean_mask_length : float, int
        Geometric training masks: expected masked fraction r and mean masked
        segment length l_m (paper: 0.15 and 3).
    tr_ratio : float
        Share of windows used for training; the rest select the checkpoint
        (minimum validation MAE on fixed masks).
    seed : int
        Seed for windows split, masks and model initialisation.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (T, N): observed values unchanged, NaN entries imputed.
    """
    data = np.asarray(incomp_data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("MTSIT expects a 2-D matrix of shape (time steps, series)")
    recov = data.copy()
    missing = np.isnan(data)
    if not missing.any() or missing.all():
        return recov
    n_steps, n_series = data.shape
    if d_model % n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if not 0 < masking_ratio < 1 or mean_mask_length < 1:
        raise ValueError("masking_ratio must be in (0, 1) and mean_mask_length >= 1")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Step 1: per-series standardisation with observed values only.
    with warnings.catch_warnings():  # all-missing series are handled just below
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    normalized = np.where(missing, 0.0, (data - mean) / std).astype(np.float32)
    observed = ~missing

    # Windows of length seq_len over the time axis.
    seq_len = min(48, n_steps) if seq_len is None or seq_len <= 0 else min(seq_len, n_steps)
    stride = seq_len if sliding_windows is None or sliding_windows <= 0 else sliding_windows
    starts = _window_starts(n_steps, seq_len, stride)
    values = np.stack([normalized[s : s + seq_len] for s in starts])
    observed_w = np.stack([observed[s : s + seq_len] for s in starts])

    order = rng.permutation(len(starts))
    n_validation = round((1.0 - tr_ratio) * len(starts)) if len(starts) > 1 else 0
    n_validation = min(max(n_validation, 1 if len(starts) > 1 else 0), len(starts) - 1)
    validation_ids, train_ids = order[:n_validation], order[n_validation:]
    batch_size = min(64, len(train_ids)) if batch_size is None or batch_size <= 0 else batch_size

    # Fixed validation masks so that checkpoint selection is repeatable.
    validation_inputs = validation_masks = None
    if len(validation_ids):
        validation_inputs, validation_masks = _masked_batch(
            values,
            observed_w,
            validation_ids,
            0,
            seed,
            masking_ratio,
            mean_mask_length,
            stream=1,
        )
        if not validation_masks.any():
            validation_inputs = validation_masks = None

    model = MTSITNetwork(
        n_series, seq_len, d_model, n_heads, n_layers, d_ffn, dropout
    ).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    best_mae, best_state = math.inf, None

    for epoch in range(epochs):
        model.train()
        epoch_order = np.random.default_rng([seed, epoch]).permutation(train_ids)
        squared, count = 0.0, 0
        for begin in range(0, len(epoch_order), batch_size):
            batch = epoch_order[begin : begin + batch_size]
            inputs, masks = _masked_batch(
                values, observed_w, batch, epoch, seed, masking_ratio, mean_mask_length
            )
            if not masks.any():
                continue
            inputs_t = torch.from_numpy(inputs).float().to(device)
            targets_t = torch.from_numpy(values[batch]).to(device)
            masks_t = torch.from_numpy(masks).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs_t)
            # Masked MSE over the masked observed entries of the mini-batch.
            loss = (prediction[masks_t] - targets_t[masks_t]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            squared += loss.item() * int(masks.sum())
            count += int(masks.sum())

        if validation_masks is not None:
            model.eval()
            with torch.no_grad():
                prediction = model(
                    torch.from_numpy(validation_inputs).float().to(device)
                ).cpu().numpy()
            mae = float(
                np.abs(prediction - values[validation_ids])[validation_masks].mean()
            )
            if mae < best_mae:
                best_mae = mae
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            train_mse = squared / count if count else float("nan")
            print(
                f"MTSIT epoch {epoch + 1}/{epochs}: train MSE {train_mse:.5f}, "
                f"validation MAE {best_mae:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference: no artificial mask (S = 0); natural gaps are zero placeholders.
    model.eval()
    summed = np.zeros((n_steps, n_series))
    counts = np.zeros((n_steps, 1))
    with torch.no_grad():
        for begin in range(0, len(starts), max(batch_size, 1)):
            ids = np.arange(begin, min(begin + max(batch_size, 1), len(starts)))
            inputs = np.where(observed_w[ids], values[ids], 0.0)
            prediction = model(torch.from_numpy(inputs).float().to(device)).cpu().numpy()
            for row, window in zip(prediction, ids):
                summed[starts[window] : starts[window] + seq_len] += row
                counts[starts[window] : starts[window] + seq_len] += 1
    imputed = summed / np.maximum(counts, 1) * std + mean
    recov[missing] = imputed[missing]
    return recov
