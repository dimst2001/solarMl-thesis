"""Local prediction portal for PV system 4 (thesis models).

Run from this folder (or with its path):
    streamlit run app.py
It reads only bundle/ (models, scalers, detector settings, daylight hours), written by export_bundle.py, so the
app/ folder can be copied to another computer on its own. Nothing is trained here.
"""
import json
import pickle
from datetime import date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BUNDLE_DIR = Path(__file__).resolve().parent / "bundle"
TASKS = ("endogenous", "exogenous")
NOCT_C = 45.0  # nominal operating cell temperature, for the module-temperature estimate
DAYLIGHT_IRRADIANCE_W_M2 = 50.0
FULL_OUTAGE_SHARE = 0.1  # measured power below 10% of expected = full outage


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_bundle():
    """Bundle metadata, and the model and scaler of every task."""
    info = json.loads((BUNDLE_DIR / "bundle.json").read_text())
    objects = {}
    for task in TASKS:
        with (BUNDLE_DIR / f"{task}_model.pkl").open("rb") as file:
            model = pickle.load(file)
        with (BUNDLE_DIR / f"{task}_scaler.pkl").open("rb") as file:
            scaler = pickle.load(file)
        objects[task] = (model, scaler)
    info["daylight_slots"] = {tuple(slot) for slot in info["daylight_slots"]}
    return info, objects


def load_task(task):
    """Model chosen in notebook 4 for the task, its scaler and its test metrics."""
    info, objects = load_bundle()
    model, scaler = objects[task]
    metrics = info["models"][task]
    return metrics["name"], model, scaler, metrics


def is_daylight(timestamp):
    """True when the sun normally gives more than 50 W/m² at this month and 15-minute slot."""
    return (timestamp.month, timestamp.hour * 4 + timestamp.minute // 15) in load_bundle()[0]["daylight_slots"]


# ---------------------------------------------------------------- features and prediction
def calendar_features(timestamps):
    """Cyclical time-of-day and day-of-year features, as in notebook 2 (the year does not matter)."""
    timestamps = pd.DatetimeIndex(timestamps)
    hour = timestamps.hour + timestamps.minute / 60
    day_of_year = timestamps.dayofyear - 1 + hour / 24
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                         "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
                         "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25)}, index=timestamps)


def predict(task, timestamps, irradiance=None, module_temp=None, ambient_temp=None):
    """Predicted DC power (W, clipped at 0) for one or more timestamps; returns (power, raw inputs, model name)."""
    name, model, scaler, _ = load_task(task)
    calendar = calendar_features(timestamps)
    raw = pd.DataFrame(index=calendar.index)
    for column in scaler.feature_names_in_:
        if "irradiance" in column.lower():
            raw[column] = irradiance
        elif "module_temp" in column:
            raw[column] = module_temp
        elif "ambient_temp" in column:
            raw[column] = ambient_temp
        else:
            raw[column] = calendar[column]
    scaled = pd.DataFrame(scaler.transform(raw), index=raw.index, columns=raw.columns)
    return np.clip(model.predict(scaled), 0, None), raw, name


def range_warnings(task, raw):
    _, _, scaler, _ = load_task(task)
    notes = []
    for column, low, high in zip(scaler.feature_names_in_, scaler.data_min_, scaler.data_max_):
        value = raw[column].iloc[0]
        if not column.endswith(("_sin", "_cos")) and not low <= value <= high:
            notes.append(f"{column} = {value:g} is outside the range seen in training "
                         f"[{low:.1f}, {high:.1f}]; the prediction is an extrapolation.")
    return notes


def format_power(watts):
    return f"{watts / 1000:,.2f} kW" if abs(watts) >= 10_000 else f"{watts:,.0f} W"


def nominal_noct(ambient, irradiance):
    return ambient + (NOCT_C - 20) / 800 * irradiance


# ---------------------------------------------------------------- page
st.set_page_config(page_title="PV Power Prediction", page_icon="☀️", layout="wide")
st.title("PV power prediction and fault check: system 4")

if not (BUNDLE_DIR / "bundle.json").is_file():
    st.error("bundle/ is missing: run `python app/export_bundle.py` from the thesis folder (after notebooks 1-5).")
    st.stop()
detector = load_bundle()[0]["detector"]
st.caption("PV system 4 (NREL PVDAQ, about 1.3 kW). Each prediction uses the model chosen in notebook 4 for that "
           "input set (the fastest of the models tied with the most accurate).")

predict_tab, fault_tab, about_tab = st.tabs(["Predict power", "Fault check", "About the models"])

# ---------------------------------------------------------------- tab 1: prediction
with predict_tab:
    mode = st.radio("What do you know?",
                    ["Date, time and ambient temperature (e.g. a weather forecast)",
                     "Irradiance and module temperature (plant sensors)"])

    if mode.startswith("Date"):
        left, middle, right = st.columns(3)
        when_date = left.date_input("Date (any date, also in the future)", value=date.today() + timedelta(days=7))
        when_time = middle.time_input("Time", value=time(12, 0), step=900)
        ambient = right.number_input("Ambient temperature (°C)", -30.0, 50.0, 20.0, 0.5)
        timestamp = pd.Timestamp.combine(when_date, when_time)

        power, raw, name = predict("exogenous", [timestamp], ambient_temp=ambient)
        _, _, _, metrics = load_task("exogenous")
        if is_daylight(timestamp):
            st.metric(f"Predicted DC power at {timestamp:%H:%M, %d/%m/%Y}", format_power(power[0]))
        else:
            st.metric(f"Predicted DC power at {timestamp:%H:%M, %d/%m/%Y}", format_power(0))
            st.info("The sun is not up enough at this time of year and day (irradiance normally below 50 W/m²), "
                    "so the system produces practically nothing.")
        for note in range_warnings("exogenous", raw):
            st.warning(note)
        st.caption(f"Model: {name} on ambient temperature + date/time. Without an irradiance forecast it cannot "
                   f"know the clouds, so it predicts a typical day for this date and temperature "
                   f"(test error ± {format_power(metrics['test_RMSE'])} RMSE, R² {metrics['test_R2']:.2f}). "
                   f"The year does not enter the model, only the day of the year and the time.")

        # Whole-day curve at the same temperature.
        slots = pd.date_range(pd.Timestamp(when_date), periods=96, freq="15min")
        daylight = np.array([is_daylight(slot) for slot in slots])
        day_power, _, _ = predict("exogenous", slots, ambient_temp=ambient)
        curve = pd.Series(np.where(daylight, day_power, 0.0), index=slots, name="predicted DC power (W)")
        st.markdown(f"**Whole day {when_date:%d/%m/%Y} at {ambient:g} °C** "
                    f"(about {curve.sum() * 0.25 / 1000:,.2f} kWh)")
        st.line_chart(curve.loc[curve.index[daylight].min() - pd.Timedelta("30min"):
                                curve.index[daylight].max() + pd.Timedelta("30min")] if daylight.any() else curve,
                      y_label="DC power (W)")

    else:
        left, right = st.columns(2)
        irradiance = left.number_input("Plane-of-array irradiance (W/m²)", 0.0, 1600.0, 800.0, 10.0)
        known = right.radio("Module temperature", ["I know it", "Estimate it from the ambient temperature"],
                            horizontal=True)
        if known == "I know it":
            module_temp = right.number_input("Module temperature (°C)", -30.0, 90.0, 40.0, 0.5)
        else:
            ambient = right.number_input("Ambient temperature (°C)", -30.0, 50.0, 20.0, 0.5)
            module_temp = nominal_noct(ambient, irradiance)
            right.caption(f"Estimated module temperature: {module_temp:.1f} °C "
                          f"(T_ambient + (NOCT − 20) / 800 × irradiance, NOCT = {NOCT_C:g} °C)")

        power, raw, name = predict("endogenous", [pd.Timestamp.now()], irradiance=irradiance,
                                   module_temp=module_temp)
        _, _, _, metrics = load_task("endogenous")
        st.metric("Predicted DC power", format_power(power[0]))
        if irradiance <= DAYLIGHT_IRRADIANCE_W_M2:
            st.info("Irradiance ≤ 50 W/m²: below the daylight threshold the models were trained on.")
        for note in range_warnings("endogenous", raw):
            st.warning(note)
        st.caption(f"Model: {name} on irradiance + module temperature "
                   f"(test error ± {format_power(metrics['test_RMSE'])} RMSE, R² {metrics['test_R2']:.3f}).")

# ---------------------------------------------------------------- tab 2: fault check
SEVERITY_STYLE = {"green": ("#2e7d32", "white"), "yellow": ("#f9d71c", "black"),
                  "orange": ("#ef6c00", "white"), "red": ("#c62828", "white")}


def severity_colour(distance, levels):
    """Colour of a reading from its distance to the median in robust sigma (levels from notebook 5)."""
    for colour in ("green", "yellow", "orange"):
        if distance < levels[colour]:
            return colour
    return "red"


def coloured_box(colour, text):
    background, foreground = SEVERITY_STYLE[colour]
    st.markdown(f"<div style='background:{background};color:{foreground};padding:14px 18px;border-radius:8px;"
                f"font-size:1.15rem'>{text}</div>", unsafe_allow_html=True)


with fault_tab:
    st.markdown("Enter one measurement: irradiance, module temperature and the measured DC power. The app "
                "computes the power the system **should** produce in those conditions (endogenous model) and "
                "colours the deviation with the bands found in the anomaly-detection notebook.")
    left, middle, right = st.columns(3)
    check_irradiance = left.number_input("Irradiance (W/m²)", 0.0, 1600.0, 800.0, 10.0, key="check_irr")
    module_known = middle.radio("Module temperature", ["I know it", "Estimate it from the ambient temperature"],
                                key="check_module_mode")
    if module_known == "I know it":
        check_module = middle.number_input("Module temperature (°C)", -30.0, 90.0, 40.0, 0.5, key="check_mod")
    else:
        check_ambient = middle.number_input("Ambient temperature (°C)", -30.0, 50.0, 20.0, 0.5,
                                            key="check_ambient")
        check_module = nominal_noct(check_ambient, check_irradiance)
        middle.caption(f"Estimated module temperature: {check_module:.1f} °C")
    measured = right.number_input("Measured DC power (W)", min_value=0.0, value=500.0, step=10.0)

    band = detector["single_reading"]["endogenous"]
    levels = band["severity_levels"]
    expected, raw, name = predict("endogenous", [pd.Timestamp.now()], irradiance=check_irradiance,
                                  module_temp=check_module)
    expected = float(expected[0])

    if check_irradiance <= DAYLIGHT_IRRADIANCE_W_M2:
        st.info("Irradiance ≤ 50 W/m² (night or start-up): the system is not expected to produce, "
                "so a fault cannot be judged from the power.")
    else:
        scale = max(expected, detector["min_expected_share"] * detector["p_ref"])
        relative = (measured - expected) / scale
        distance = abs(relative - band["center"]) / band["sigma"]
        colour = severity_colour(distance, levels)
        deviation_watts = measured - expected
        deviation_share = deviation_watts / expected if expected > 0 else 0.0
        under = relative < band["center"]

        if colour == "green":
            text = f"<b>NORMAL</b>: {100 * deviation_share:+.0f}% from the expected power."
        elif colour == "yellow":
            text = (f"<b>DEVIATION, WATCH</b>: {format_power(abs(deviation_watts))} "
                    f"({100 * abs(deviation_share):.0f}%) {'below' if under else 'above'} the expected power.")
        else:
            severity = "FAULT" if colour == "orange" else "SEVERE FAULT"
            if under:
                kind = "full outage" if measured < FULL_OUTAGE_SHARE * expected else "partial loss"
                text = (f"<b>{severity}, {kind}</b>: the system produces {format_power(-deviation_watts)} "
                        f"({100 * -deviation_share:.0f}%) less than expected.")
            else:
                text = (f"<b>{severity}, over-production</b>: {format_power(deviation_watts)} "
                        f"({100 * deviation_share:.0f}%) more than expected; most likely the irradiance sensor "
                        f"reads too low (dirty, shaded or misaligned).")
        coloured_box(colour, text + f" &nbsp;({distance:.1f} σ)")

        a, b, c = st.columns(3)
        a.metric("Expected power", format_power(expected))
        b.metric("Measured power", format_power(measured), f"{100 * deviation_share:+.1f}%")
        rule_fault = measured < detector["fault_ratio"] * detector["k"] * check_irradiance
        c.metric("Cleaning rule (notebook 1)", "fault" if rule_fault else "no fault",
                 help="Power below 0.5 × k × irradiance: the rule only sees deep losses and outages.")
        for note in range_warnings("endogenous", raw):
            st.warning(note)

        # Power ranges of every colour for these conditions.
        def power_at(level):
            return (max(expected + (band["center"] - level * band["sigma"]) * scale, 0.0),
                    expected + (band["center"] + level * band["sigma"]) * scale)

        limits = {colour_name: power_at(level) for colour_name, level in levels.items()}
        ranges = pd.DataFrame({
            "distance": ["< 3σ", "3σ – 5σ", "5σ – 8σ", "> 8σ"],
            "relative deviation": [f"within ±{100 * levels['green'] * band['sigma']:.0f}%",
                                   f"±{100 * levels['green'] * band['sigma']:.0f}% to ±{100 * levels['yellow'] * band['sigma']:.0f}%",
                                   f"±{100 * levels['yellow'] * band['sigma']:.0f}% to ±{100 * levels['orange'] * band['sigma']:.0f}%",
                                   f"beyond ±{100 * levels['orange'] * band['sigma']:.0f}%"],
            "measured power (below expected)": [
                f"{format_power(limits['green'][0])} – {format_power(expected)}",
                f"{format_power(limits['yellow'][0])} – {format_power(limits['green'][0])}",
                f"{format_power(limits['orange'][0])} – {format_power(limits['yellow'][0])}",
                f"< {format_power(limits['orange'][0])}"],
            "measured power (above expected)": [
                f"{format_power(expected)} – {format_power(limits['green'][1])}",
                f"{format_power(limits['green'][1])} – {format_power(limits['yellow'][1])}",
                f"{format_power(limits['yellow'][1])} – {format_power(limits['orange'][1])}",
                f"> {format_power(limits['orange'][1])}"],
        }, index=["green: normal", "yellow: watch", "orange: fault", "red: severe fault"])
        st.markdown("**Colour bands for these conditions**")
        st.dataframe(ranges.style.apply(
            lambda row: [f"background-color: {SEVERITY_STYLE[row.name.split(':')[0]][0]};"
                         f"color: {SEVERITY_STYLE[row.name.split(':')[0]][1]}"] * len(row), axis=1))
        st.caption(f"Reference: {name} on irradiance + module temperature. σ = {100 * band['sigma']:.1f}% is the "
                   f"robust spread (1.4826 × MAD) of the relative residual on normal training readings "
                   f"(notebook 5, section 5.9); the levels 3σ / 5σ / 8σ are those of the sensitivity analysis "
                   f"(5.5.3), and 5σ is the threshold of the residual detector. The residual is relative to "
                   f"max(expected, {100 * detector['min_expected_share']:.0f}% of P_REF), so at low light the bands "
                   f"are fixed in watts. Notebook 5 applies the detector to the 15-minute median; one reading is "
                   f"noisier.")

# ---------------------------------------------------------------- tab 3: models
with about_tab:
    rows = []
    for task in TASKS:
        name, _, scaler, metrics = load_task(task)
        rows.append({"used for": "power prediction + fault check" if task == "endogenous" else "power prediction",
                     "inputs": ", ".join(scaler.feature_names_in_), "model": name,
                     "test RMSE": format_power(metrics["test_RMSE"]), "test R²": round(metrics["test_R2"], 3)})
    st.dataframe(pd.DataFrame(rows, index=["Plant sensors", "Weather + date"]))
    band = detector["single_reading"]["endogenous"]
    st.markdown(f"Fault-check colours (notebook 5, section 5.9): σ = {100 * band['sigma']:.1f}% of the expected "
                f"power; green < {band['severity_levels']['green']}σ, yellow < {band['severity_levels']['yellow']}σ, "
                f"orange < {band['severity_levels']['orange']}σ, red beyond. Models from notebooks 3A/3B, chosen in "
                f"notebook 4. All predictions are DC power of this system under normal operation.")
