"""Rolling-origin backtesting with purging.

k-fold on a time series leaks the future into the past, so splits here are always
chronological on the forecast *origin*.

Even that isn't enough for direct multi-horizon work: a training row whose origin sits
just before the boundary has its target at origin + h, which lands inside the test window
-- the model gets fitted on an outcome it's then asked to predict. Hence the purge gap of
one horizon between train and test.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lowland.utils import get_logger

log = get_logger(__name__)


@dataclass
class Fold:
    """One train/test split, expressed as boolean masks over the sample index."""

    fold_id: int
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Fold {self.fold_id}: train n={self.train_mask.sum()} up to {self.train_end.date()} | "
            f"test n={self.test_mask.sum()} {self.test_start.date()}..{self.test_end.date()}"
        )


@dataclass
class RollingOriginSplitter:
    """Expanding- or sliding-window splitter over forecast origins.

    Parameters
    ----------
    n_folds:
        Number of evaluation windows.
    test_days:
        Length of each test window in days.
    horizon:
        Forecast horizon in hours; also the size of the purge gap.
    expanding:
        ``True`` grows the training set each fold (more data, closer to production
        retraining); ``False`` keeps a fixed-length sliding window (tests robustness to
        regime change, since the model never sees the distant past).
    min_train_days:
        Minimum training length before the first fold is emitted.
    max_train_days:
        Cap on training length when ``expanding`` is ``False``.
    """

    n_folds: int = 6
    test_days: int = 30
    horizon: int = 24
    expanding: bool = True
    min_train_days: int = 730
    max_train_days: int = 1095

    def split(self, origin_index: pd.DatetimeIndex) -> Iterator[Fold]:
        """Yield folds over the supplied forecast-origin timestamps."""
        origins = pd.DatetimeIndex(origin_index)
        if origins.empty:
            return

        t_end = origins.max()
        test_len = pd.Timedelta(days=self.test_days)
        purge = pd.Timedelta(hours=self.horizon)

        for k in range(self.n_folds):
            # Fold 0 is the most recent window; later folds step further back in time.
            back = self.n_folds - 1 - k
            test_stop = t_end - back * test_len
            test_start = test_stop - test_len
            train_stop = test_start - purge

            if (train_stop - origins.min()) < pd.Timedelta(days=self.min_train_days):
                log.debug("skipping fold %s: insufficient training history", k)
                continue

            train_start = origins.min()
            if not self.expanding:
                train_start = max(train_start, train_stop - pd.Timedelta(days=self.max_train_days))

            train_mask = (origins >= train_start) & (origins <= train_stop)
            test_mask = (origins > test_start) & (origins <= test_stop)
            if train_mask.sum() < 500 or test_mask.sum() < 24:
                continue

            yield Fold(
                fold_id=k,
                train_mask=train_mask,
                test_mask=test_mask,
                train_end=train_stop,
                test_start=test_start,
                test_end=test_stop,
            )


def calibration_split(
    train_mask: np.ndarray,
    origin_index: pd.DatetimeIndex,
    *,
    calib_days: int = 120,
    horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a held-out calibration block off the end of a training window.

    Conformal prediction requires a calibration set that the point/quantile model has
    never seen. Taking the most recent block rather than a random subset keeps the
    calibration distribution as close as possible to the test period, which matters when
    the data-generating process drifts -- and in European power markets it drifts a lot.

    A purge of ``horizon`` hours is applied between the fitting block and the calibration
    block for the same reason it is applied between train and test.
    """
    origins = pd.DatetimeIndex(origin_index)
    train_origins = origins[train_mask]
    if train_origins.empty:
        return train_mask, np.zeros_like(train_mask)

    calib_start = train_origins.max() - pd.Timedelta(days=calib_days)
    fit_stop = calib_start - pd.Timedelta(hours=horizon)

    fit_mask = train_mask & (origins <= fit_stop)
    calib_mask = train_mask & (origins > calib_start)

    if calib_mask.sum() < 200:  # fall back to no calibration split if too little data
        log.warning("calibration block too small (n=%s); using train set unsplit", calib_mask.sum())
        return train_mask, np.zeros_like(train_mask)
    return fit_mask, calib_mask


def summarise_folds(folds: list[Fold]) -> pd.DataFrame:
    """Tabulate fold geometry for the report and the apps."""
    return pd.DataFrame(
        [
            {
                "fold": f.fold_id,
                "n_train": int(f.train_mask.sum()),
                "n_test": int(f.test_mask.sum()),
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
            }
            for f in folds
        ]
    )
