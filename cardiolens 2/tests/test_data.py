from src.prepare_data import CATEGORICAL, FEATURES, NUMERIC, TARGET, clean, load_raw


def test_raw_shape_and_missing_marker():
    raw = load_raw()
    assert raw.shape == (303, 14)
    assert raw[["ca", "thal"]].isna().sum().sum() == 6  # '?' parsed as NaN


def test_target_is_binary_and_matches_num():
    raw = load_raw(); df = clean(raw)
    assert set(df[TARGET].unique()) == {0, 1}
    assert ((raw["num"] > 0).astype(int).values == df[TARGET].values).all()
    assert df[TARGET].sum() == (raw["num"] > 0).sum()


def test_cleaning_keeps_all_rows_and_features_cover_columns():
    raw = load_raw(); df = clean(raw)
    assert len(df) == len(raw)
    assert sorted(FEATURES) == sorted(set(NUMERIC + CATEGORICAL))
    assert len(FEATURES) == 13
