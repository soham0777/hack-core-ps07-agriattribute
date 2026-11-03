"""Regression tests for yield attribution, dose handling and ROI consistency (audit P0 items)."""
import os
import sys

import pandas as pd
import pytest
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services import decision_simulator  # noqa: E402
from services.field_context import (  # noqa: E402
    ENCODED_COLUMNS, TRAINED_BIO_DOSE, TRAINING_RAINFALL_NORMAL_MM, build_field_context,
)

TRIALS = pd.read_csv(os.path.join(ROOT, "data", "field_trials.csv"))


@pytest.fixture(scope="module")
def model():
    # Same settings as the pipeline's XGBoost candidate; trained on 2021-2024 like production
    enc = pd.get_dummies(TRIALS, columns=["crop_type", "region"]).reindex(columns=ENCODED_COLUMNS, fill_value=0).astype(float)
    train = TRIALS["year"] < 2025
    m = xgb.XGBRegressor(n_estimators=160, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, random_state=42)
    m.fit(enc[train], TRIALS.loc[train, "yield"])
    return m


def make_ctx(crop="Soybean", dose=2.0, unit="L/acre", temp=30.0):
    return build_field_context(
        region="Maharashtra & Vidarbha (Deccan)", crop=crop, lat=21.15, lon=79.09, location_name="Nagpur",
        ow_live={"temp_c": temp}, shc_data={}, mandi_info={"realizable_price": 4900.0, "msp": 4892.0},
        bio_applied=True, bio_dosage=dose, product_cost_per_ha=1850.0, product_unit_cost_per_l=850.0,
        application_cost_acre=150.0, recommended_bio_dose=2.0, bio_dose_unit=unit,
    )


@pytest.mark.parametrize("dose", [0.5, 1.0, 3.0, 5.0, 150.0])
def test_model_input_dose_is_pinned_to_trained_rate(dose):
    x = make_ctx(dose=dose).to_feature_dataframe()
    assert x["bio_dosage_l_ha"].iloc[0] == TRAINED_BIO_DOSE["Soybean"]


def test_untrained_dose_no_longer_explodes_yield(model):
    # Before the fix, Soybean at 3.0 L was predicted near the Sugarcane level (~113 q/acre)
    y = [decision_simulator.simulate_5_scenarios(make_ctx(dose=d), model)["scenarios"][0]["expected_yield_q_acre"] for d in (2.0, 3.0, 5.0)]
    assert max(y) - min(y) < 1e-6
    assert max(y) < 20.0


def test_cost_scales_with_litre_dose_and_falls_back_for_grams():
    assert make_ctx(dose=2.0).product_cost_per_acre() == 2.0 * 850.0 + 150.0
    assert make_ctx(dose=3.0).product_cost_per_acre() == 3.0 * 850.0 + 150.0
    assert make_ctx(dose=150.0, unit="g/acre").product_cost_per_acre() == 1850.0


def test_roi_uses_per_acre_cost_of_chosen_dose(model):
    ctx = make_ctx(dose=3.0)
    cur = decision_simulator.simulate_5_scenarios(ctx, model, dosage_override=3.0)["scenarios"][0]
    assert cur["biological_cost_inr"] == ctx.product_cost_per_acre(3.0)
    # incremental yield is reported rounded to 0.01 q, so allow that rounding times the price
    assert cur["net_profit_inr"] == pytest.approx(cur["incremental_yield_q_acre"] * ctx.crop_price - cur["total_input_cost_inr"], abs=0.005 * ctx.crop_price + 0.05)


@pytest.mark.parametrize("crop", sorted(TRAINED_BIO_DOSE))
def test_decomposition_adds_up_and_matches_scenario_lift(model, crop):
    ctx = make_ctx(crop=crop)
    sim = decision_simulator.simulate_5_scenarios(ctx, model)
    parts = sim["attribution"]
    total = parts["base_q_acre"] + parts["soil_q_acre"] + parts["weather_q_acre"] + parts["bio_q_acre"]
    assert total == pytest.approx(parts["total_q_acre"], abs=0.05)
    assert parts["total_q_acre"] == pytest.approx(sim["scenarios"][0]["expected_yield_q_acre"], abs=0.05)
    assert parts["bio_q_acre"] == pytest.approx(sim["scenarios"][0]["incremental_yield_q_acre"], abs=0.05)


@pytest.mark.parametrize("crop", sorted(TRAINING_RAINFALL_NORMAL_MM))
def test_rainfall_input_is_inside_training_range(crop):
    seen = TRIALS.loc[TRIALS["crop"] == crop, "cumulative_rainfall_mm"]
    assert seen.min() <= make_ctx(crop=crop).cumulative_rainfall_mm <= seen.max()


def test_error_band_is_crop_specific(model):
    small = decision_simulator.crop_error_band(model, "Soybean")
    large = decision_simulator.crop_error_band(model, "Sugarcane")
    assert 0 < small < large


def test_why_this_result_uses_model_numbers(model):
    ctx = make_ctx()
    parts = decision_simulator.decompose_yield(ctx, model)
    factors = decision_simulator.explain_attribution(ctx, model)
    assert factors[0]["impact_q_acre"] == f"{parts['bio_q_acre']:+.2f} q/acre"
