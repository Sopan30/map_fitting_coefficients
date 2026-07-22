from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline
import numpy as np
from GasCalculator import GasCalculator as gas_calc

class CurveFitter:

    def build_model(self, x, y, method):

        if method == 'Spline':
            idx = np.argsort(x)

            x = x[idx]
            y = y[idx]

            spline = CubicSpline(x, y)

            return {
                'type': 'spline',
                'model': spline,
                'xmin': x.min(),
                'xmax': x.max(),
                'r2': r2_score(y, spline(x))
            }

        degree = {
            'Linear': 1,
            'Quadratic': 2,
            'Cubic': 3,
            '4th Order': 4,
            '5th Order': 5
        }[method]

        poly = PolynomialFeatures(degree)

        X = poly.fit_transform(
            x.reshape(-1, 1)
        )

        lr = LinearRegression().fit(X, y)

        return {
            'type': 'poly',
            'poly': poly,
            'model': lr,
            'xmin': x.min(),
            'xmax': x.max(),
            'r2': r2_score(y, lr.predict(X))
        }

    def predict(self, model_obj, flow):

        if model_obj['type'] == 'spline':
            return model_obj['model'](flow)

        return model_obj['model'].predict(
            model_obj['poly'].transform(
                flow.reshape(-1, 1)
            )
        )

    def auto_best(self, x, y):

        best_model = None
        best_name = None
        best_r2 = -1e9

        methods = [
            'Linear',
            'Quadratic',
            'Cubic',
            '4th Order',
            '5th Order',
            'Spline'
        ]

        for method in methods:

            try:

                model = self.build_model(
                    x,
                    y,
                    method
                )

                if model['r2'] > best_r2:
                    best_model = model
                    best_name = method
                    best_r2 = model['r2']

            except Exception:
                pass

        return best_name, best_model

    def calculate_scaling_factors(
        self,
        final_df,
        gas_props,
        acoustic_vel,
        spec_vol
    ):

        if gas_props is None:
            return None

        required_cols = {
            "Speed",
            "Flow (m3/hr)"
        }

        if not required_cols.issubset(final_df.columns):
            return None

        diameter = gas_props["diameter_m"]

        q_factor = (
            1.0 /
            (acoustic_vel * diameter**2)
        )

        hp_factor = (
            1000.0 /
            (acoustic_vel**2)
        )

        p_factor = (
            1000.0 *
            spec_vol /
            (
                acoustic_vel**2 *
                diameter**3 *
                (2 * np.pi / 60.0)
            )
        )

        df = final_df.copy()

        flow_m3s = (
            df["Flow (m3/hr)"] / 3600.0
        )

        df["Qr"] = (
            df["Speed"] *
            flow_m3s *
            q_factor
        )

        if "Head (m)" in df.columns:

            df["Hpr"] = (
                df["Head (m)"] *
                gas_calc.G /
                (acoustic_vel**2)
            )

        if "Power (kW)" in df.columns:

            df["Pnr"] = (
                df["Power (kW)"] *
                p_factor
            )

        scaling = {}

        if "Hpr" in df.columns:

            qr_max = df["Qr"].max()
            hpr_max = df["Hpr"].max()

            qr_scale_head = 1.0
            hpr_scale = 1.0

            if hpr_max > qr_max:
                qr_scale_head = round(
                    hpr_max / qr_max,
                    4
                )

            elif qr_max > hpr_max:
                hpr_scale = round(
                    qr_max / hpr_max,
                    4
                )

            scaling.update({
                "Qr_Max_Head": qr_max,
                "Hpr_Max": hpr_max,
                "Qr_Scale_Head": qr_scale_head,
                "Hpr_Scale": hpr_scale
            })

        if "Pnr" in df.columns:

            qr_max = df["Qr"].max()
            pnr_max = df["Pnr"].max()

            qr_scale_power = 1.0
            pnr_scale = 1.0

            if pnr_max > qr_max:
                qr_scale_power = round(
                    pnr_max / qr_max,
                    4
                )

            elif qr_max > pnr_max:
                pnr_scale = round(
                    qr_max / pnr_max,
                    4
                )

            scaling.update({
                "Qr_Max_Power": qr_max,
                "Pnr_Max": pnr_max,
                "Qr_Scale_Power": qr_scale_power,
                "Pnr_Scale": pnr_scale
            })

        return df, scaling
