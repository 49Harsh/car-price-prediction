import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "dataset.csv"
CURRENT_YEAR = datetime.datetime.now().year
NUMERIC_COLUMNS = [
    "Year",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
]
CATEGORICAL_COLUMNS = ["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"]


def _extract_number(values):
    return pd.to_numeric(
        values.astype("string").str.extract(r"^\s*([0-9]+(?:\.[0-9]+)?)", expand=False),
        errors="coerce",
    )


def train_model():
    dataset = pd.read_csv(DATASET_PATH)
    features = dataset[
        [
            "Year",
            "Kilometers_Driven",
            "Fuel_Type",
            "Transmission",
            "Owner_Type",
            "Mileage",
            "Engine",
            "Power",
            "Seats",
        ]
    ].copy()
    features["Manufacturer"] = dataset["Name"].astype("string").str.split().str[0]
    features["Year"] = CURRENT_YEAR - pd.to_numeric(features["Year"], errors="coerce")

    for column in ("Mileage", "Engine", "Power"):
        features[column] = _extract_number(features[column])

    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].fillna("Unknown").astype(str)

    category_values = {
        column: sorted(features[column].unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
        features[column] = features[column].fillna(features[column].mean())

    encoded_features = pd.get_dummies(
        features,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=float,
    )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(encoded_features)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(scaled_features, dataset["Price"])

    return model, scaler, encoded_features.columns.tolist(), category_values, int(dataset["Year"].min())


def predict_price(model_bundle, car_details):
    model, scaler, feature_columns, category_values, _ = model_bundle
    row = pd.DataFrame(
        [
            {
                "Year": CURRENT_YEAR - int(car_details["Year"]),
                "Kilometers_Driven": float(car_details["Kilometers_Driven"]),
                "Mileage": float(car_details["Mileage"]),
                "Engine": float(car_details["Engine"]),
                "Power": float(car_details["Power"]),
                "Seats": float(car_details["Seats"]),
                "Manufacturer": car_details["Manufacturer"],
                "Fuel_Type": car_details["Fuel_Type"],
                "Transmission": car_details["Transmission"],
                "Owner_Type": car_details["Owner_Type"],
            }
        ]
    )

    for column in CATEGORICAL_COLUMNS:
        row[column] = pd.Categorical(row[column], categories=category_values[column])

    encoded_row = pd.get_dummies(
        row,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=float,
    ).reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(scaler.transform(encoded_row))[0]
    return max(0.0, float(prediction))


class CarPriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Used Car Price Estimator")
        self.root.geometry("900x760")
        self.root.minsize(820, 700)
        self.root.configure(bg="#f3f4ee")
        self.model_bundle = None
        self.minimum_year = 1990
        self.fields = {}
        self.status = tk.StringVar(value="Preparing the prediction model...")

        self._configure_styles()
        self._build_layout()
        self.root.after(100, self._load_model)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f4ee")
        style.configure("TLabel", background="#f3f4ee", foreground="#26342d", font=("Segoe UI", 10))
        style.configure("Field.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", padding=(10, 9), fieldbackground="#ffffff")
        style.configure("TCombobox", padding=(8, 8), fieldbackground="#ffffff")
        style.configure(
            "Predict.TButton",
            background="#1e5945",
            foreground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            padding=(20, 11),
        )
        style.map("Predict.TButton", background=[("active", "#174634"), ("disabled", "#8eaaa0")])

    def _build_layout(self):
        header = tk.Frame(self.root, bg="#183d32", padx=34, pady=26)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Used Car Price Estimator",
            bg="#183d32",
            fg="#ffffff",
            font=("Georgia", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Enter the car details to estimate its resale value",
            bg="#183d32",
            fg="#d2e0d8",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(6, 0))

        content = ttk.Frame(self.root, padding=(34, 24, 34, 26))
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Vehicle details", font=("Segoe UI", 15, "bold")).pack(anchor="w")

        form = ttk.Frame(content, padding=(0, 15, 0, 8))
        form.pack(fill="x")
        form.columnconfigure(0, weight=1, uniform="form")
        form.columnconfigure(1, weight=1, uniform="form")

        definitions = [
            ("Year", "Year", "entry", "2015"),
            ("Manufacturer", "Manufacturer", "combo", None),
            ("Kilometers_Driven", "Kilometers driven", "entry", "45000"),
            ("Fuel_Type", "Fuel type", "combo", None),
            ("Transmission", "Transmission", "combo", None),
            ("Owner_Type", "Owner type", "combo", None),
            ("Mileage", "Mileage (km/l or km/kg)", "entry", "18.0"),
            ("Engine", "Engine (CC)", "entry", "1500"),
            ("Power", "Power (bhp)", "entry", "100"),
            ("Seats", "Seats", "entry", "5"),
        ]
        for index, (key, label, field_type, default) in enumerate(definitions):
            row, column = divmod(index, 2)
            field = ttk.Frame(form, padding=(0, 0, 18 if column == 0 else 0, 13))
            field.grid(row=row, column=column, sticky="ew")
            ttk.Label(field, text=label, style="Field.TLabel").pack(anchor="w", pady=(0, 6))

            if field_type == "combo":
                widget = ttk.Combobox(field, state="readonly", width=30)
            else:
                widget = ttk.Entry(field, width=32)
                widget.insert(0, default)
            widget.pack(fill="x")
            self.fields[key] = widget

        result = tk.Frame(content, bg="#e3ece4", padx=20, pady=16)
        result.pack(fill="x", pady=(8, 16))
        tk.Label(
            result,
            text="Estimated resale price",
            bg="#e3ece4",
            fg="#365b49",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        self.price_label = tk.Label(
            result,
            text="₹ -- lakh",
            bg="#e3ece4",
            fg="#183d32",
            font=("Georgia", 24, "bold"),
        )
        self.price_label.pack(anchor="w", pady=(4, 0))

        actions = ttk.Frame(content)
        actions.pack(fill="x")
        self.predict_button = ttk.Button(
            actions,
            text="Predict price",
            style="Predict.TButton",
            command=self._predict,
            state="disabled",
        )
        self.predict_button.pack(side="left")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=(16, 0))

    def _load_model(self):
        try:
            self.model_bundle = train_model()
            _, _, _, category_values, self.minimum_year = self.model_bundle
            defaults = {
                "Manufacturer": "Maruti",
                "Fuel_Type": "Diesel",
                "Transmission": "Manual",
                "Owner_Type": "First",
            }
            for key, widget in self.fields.items():
                if isinstance(widget, ttk.Combobox):
                    values = category_values[key]
                    widget.configure(values=values)
                    preferred = defaults.get(key)
                    widget.set(preferred if preferred in values else values[0])
            self.predict_button.configure(state="normal")
            self.status.set("Model ready")
        except Exception as error:
            self.status.set("Could not load the model")
            messagebox.showerror("Model error", str(error), parent=self.root)

    def _predict(self):
        try:
            details = {key: widget.get().strip() for key, widget in self.fields.items()}
            if any(not value for value in details.values()):
                raise ValueError("Please fill in every vehicle detail.")

            year = int(details["Year"])
            if year < self.minimum_year or year > CURRENT_YEAR:
                raise ValueError(f"Year must be between {self.minimum_year} and {CURRENT_YEAR}.")
            for key in ("Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"):
                if float(details[key]) <= 0:
                    raise ValueError(f"{key.replace('_', ' ')} must be greater than zero.")

            price = predict_price(self.model_bundle, details)
            self.price_label.configure(text=f"₹ {price:.2f} lakh")
            self.status.set("Prediction complete")
        except ValueError as error:
            messagebox.showwarning("Check the details", str(error), parent=self.root)


def main():
    root = tk.Tk()
    CarPriceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()