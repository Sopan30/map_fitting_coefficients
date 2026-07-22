import numpy as np
import plotly.graph_objects as go


class PlotManager:

    def __init__(self, curve_fitter):
        self.curve_fitter = curve_fitter

    def build_parameter_plot(
        self,
        df,
        param,
        method,
        points,
        stage_models,
        r2_rows,
        stage_name
    ):

        fig = go.Figure()

        if param not in stage_models:
            stage_models[param] = {}

        for speed in sorted(df['Speed'].unique()):

            sdf = df[df['Speed'] == speed]

            if len(sdf) < 4:
                continue

            x = sdf['Flow'].values.astype(float)
            y = sdf['Value'].values.astype(float)

            if method == 'Auto Best Fit':

                used, mdl = self.curve_fitter.auto_best(
                    x,
                    y
                )

            else:

                mdl = self.curve_fitter.build_model(
                    x,
                    y,
                    method
                )

                used = method

            if mdl is None:
                continue

            stage_models[param][speed] = mdl

            r2_rows.append([
                stage_name,
                speed,
                param,
                used,
                round(mdl['r2'], 6)
            ])

            flow_fit = np.linspace(
                x.min(),
                x.max(),
                points
            )

            y_fit = self.curve_fitter.predict(
                mdl,
                flow_fit
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name=f'{speed} Original'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=flow_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'{speed} Fit'
                )
            )

        return fig