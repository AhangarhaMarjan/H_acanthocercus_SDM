"""Verify published contrasts and variance shares using public tables only."""
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from partition_interactions import partition

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = RESULTS / "sensitivity"


def main():
    areas = pd.read_csv(DATA / "V4_resample_areas.csv")
    expected = pd.read_csv(DATA / "V4_threshold_contrasts.csv")
    assert len(areas) == 720 and len(expected) == 12
    for row in expected.itertuples():
        group = areas[(areas.scenario == row.scenario) & (areas.threshold == row.threshold)]
        a = group.pivot(index="rep", columns="baseline", values=row.policy)
        assert a.shape == (30, 4) and a.notna().all().all()
        contrast = ((a.WC_2000_2024 - a.WC_1970_1999) + (a.TC_2000_2024 - a.TC_1970_1999)) / 2
        np.testing.assert_allclose([contrast.mean(), *np.percentile(contrast, [2.5, 97.5])],
                                   [row.mean, row.lower, row.upper], rtol=1e-10, atol=1e-7)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in ["V4_factorial_cells.csv", "V4_factorial_shares.csv"]:
            copyfile(DATA / name, folder / name)
        partition(folder)
        pd.testing.assert_frame_equal(pd.read_csv(folder / "V4_1_factorial_terms.csv"),
                                      pd.read_csv(DATA / "V4_1_factorial_terms.csv"))
    print("Verified 12 threshold contrasts and all 126 factorial terms from public tables.")


if __name__ == "__main__":
    main()
