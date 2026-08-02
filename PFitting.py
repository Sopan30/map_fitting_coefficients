import io, os, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

N2_FIXED, Pwr, Srgp, Swrp = 1.0, True, True, True
def round2(value):
    if value == 0: return 0.0
    return round(value, -int(math.floor(math.log10(abs(value)))) + 1)
def get_normal_val_and_slope(x, a1, b1, c1, n1):
    val = c1 + b1 * np.abs(x - a1) ** n1
    slope = b1 * n1 * np.abs(x - a1) ** (n1 - 1) * np.sign(x - a1)
    return val, slope
def evaluate_master_power_curve(X_vec, A2, N2, A1, B1, C1, N1, A3, q_join1, q_join2):
    H_at_J1, _ = get_normal_val_and_slope(q_join1, A1, B1, C1, N1)
    H_at_J2, dH_at_J2 = get_normal_val_and_slope(q_join2, A1, B1, C1, N1)
    C2 = 0.0
    B2 = (H_at_J1 - A2 * (q_join1 ** 2)) / q_join1 if q_join1 != 0 else 0.0
    B3 = dH_at_J2 - 2 * A3 * q_join2
    C3 = H_at_J2 - B3 * q_join2 - A3 * (q_join2 ** 2)
    Y_pred = np.zeros_like(X_vec)
    srg_mask = X_vec <= q_join1
    Y_pred[srg_mask] = C2 + B2 * X_vec[srg_mask] + A2 * (X_vec[srg_mask] ** 2)
    norm_mask = (X_vec > q_join1) & (X_vec < q_join2)
    Y_pred[norm_mask] = C1 + B1 * np.abs(X_vec[norm_mask] - A1)**N1
    sw_mask = X_vec >= q_join2
    Y_pred[sw_mask] = C3 + B3 * X_vec[sw_mask] + A3 * (X_vec[sw_mask] ** 2)
    return Y_pred, B2, B3, C2, C3
class PFitting:
    def __init__(self):
        self.Nn, self.Qn, self.Pn = None, None, None
        self.q_exp, self.p_exp, self.q_scale, self.p_scale = 1.0, 3.0, 1.0, 1.0
        self.QrJoin1, self.QrJoin2 = 0.18, 0.24
    def run_calibrations(self,df,QrHpScaleFtr):
        self.Nn, self.Qn, self.Pn = df['Nn'].to_numpy(), df['Qn'].to_numpy(), df['Pn'].to_numpy()
        q_grid, p_grid = np.arange(0.8, 1.2 + 1e-9, 0.01), np.arange(2.8, 3.2 + 1e-9, 0.01)
        best_q, best_p, min_adjacent_dist_sq = 1.0, 3.0, float('inf')
        for p_val in p_grid:
            for q_val in q_grid:
                X, Y = self.Qn / (self.Nn ** q_val), self.Pn / (self.Nn ** p_val)
                sort_idx = np.argsort(X)
                dist_sq = np.sum(np.diff(X[sort_idx])**2 + np.diff(Y[sort_idx])**2)
                if dist_sq < min_adjacent_dist_sq:
                    min_adjacent_dist_sq, best_q, best_p = dist_sq, q_val, p_val
        self.q_exp, self.p_exp = best_q, best_p
        Qr_coarse, Pr_coarse = self.Qn / (self.Nn ** self.q_exp), self.Pn / (self.Nn ** self.p_exp)
        Qnrmax, Pnrmax = np.max(Qr_coarse), np.max(Pr_coarse)
        if QrHpScaleFtr != 1.0:
            self.q_scale = QrHpScaleFtr
            self.p_scale = round2(QrHpScaleFtr * Qnrmax / Pnrmax)
        else:
            self.q_scale = round2(Pnrmax / Qnrmax) if Pnrmax > Qnrmax else 1.0
            self.p_scale = 1.0 if Pnrmax > Qnrmax else round2(Qnrmax / Pnrmax)
        X_coarse_scaled, Y_coarse_scaled = Qr_coarse * self.q_scale, Pr_coarse * self.p_scale
        sort_indices = np.argsort(X_coarse_scaled)
        node_x_init = np.linspace(np.min(X_coarse_scaled), np.max(X_coarse_scaled), num=11)
        node_y_init = np.interp(node_x_init, X_coarse_scaled[sort_indices], Y_coarse_scaled[sort_indices])
        initial_vector_plf = np.concatenate(([self.q_exp, self.p_exp], node_y_init))
        def plf_joint_objective(params):
            q_e, p_e, node_y = float(params[0]), float(params[1]), params[2:]
            X_curr, Y_curr = (self.Qn / (self.Nn ** q_e)) * self.q_scale, (self.Pn / (self.Nn ** p_e)) * self.p_scale
            sort_loop = np.argsort(X_curr)
            X_sorted, Y_sorted = X_curr[sort_loop], Y_curr[sort_loop]
            node_x_curr = np.linspace(np.min(X_sorted), np.max(X_sorted), num=11)
            return np.sum((Y_sorted - np.interp(X_sorted, node_x_curr, node_y)) ** 2)
        res_plf = minimize(plf_joint_objective, initial_vector_plf, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        self.q_exp, self.p_exp, node_y_refined = float(res_plf.x[0]), float(res_plf.x[1]), res_plf.x[2:]
        X_fine, Y_fine = (self.Qn / (self.Nn ** self.q_exp)) * self.q_scale, (self.Pn / (self.Nn ** self.p_exp)) * self.p_scale
        Qr_min_bound, Qr_max_bound = np.min(X_fine), np.max(X_fine)
        self.QrJoin1 = Qr_min_bound + 0.25 * (X_fine[np.argmax(Y_fine)] - Qr_min_bound) if Srgp else Qr_min_bound
        self.QrJoin2 = Qr_max_bound - 0.1 * (Qr_max_bound - X_fine[np.argmax(Y_fine)]) if Swrp else Qr_max_bound
        test_x_norm = np.linspace(self.QrJoin1, self.QrJoin2, num=4)
        PrPLF_Nodes = np.interp(test_x_norm, X_fine, Y_fine)
        N1_i, G10_val = 2.0, float(np.max(node_y_refined))
        if Swrp:
            ratio_val = (float(PrPLF_Nodes[0]) - G10_val) / (float(PrPLF_Nodes[-1]) - G10_val)
            denom = (1.0 + (abs(ratio_val) ** (1.0 / N1_i)))
            A1_init = float((self.QrJoin1 + self.QrJoin2 * (abs(ratio_val) ** (1.0 / N1_i))) / (denom if abs(denom) >= 1e-6 else 1e-6))
            B1_init = float((float(PrPLF_Nodes[0]) - G10_val) / max(1e-6, abs(self.QrJoin1 - A1_init) ** N1_i))
        else:
            A1_init, B1_init = float(Qr_max_bound), -1.0
        def solver1_obj(p):
            y_fit, _ = get_normal_val_and_slope(test_x_norm, p[0], p[1], p[2], p[3])
            return np.sum((PrPLF_Nodes - y_fit) ** 2)
        res_s1 = minimize(solver1_obj, [float(A1_init), float(B1_init), G10_val, N1_i], method='SLSQP', bounds=[(self.QrJoin1, self.QrJoin2 if Swrp else None), (None, -1e-6), (1e-6, G10_val if Swrp else None), (1.000001, None)])
        A1_s1, B1_s1, C1_s1, N1_s1 = res_s1.x
        h_j2, dh_j2 = get_normal_val_and_slope(self.QrJoin2, A1_s1, B1_s1, C1_s1, N1_s1)
        A3_init = float((Y_fine[np.argmax(X_fine)] - h_j2 - (dh_j2 * (Qr_max_bound - self.QrJoin2))) / (Qr_max_bound**2 - self.QrJoin2**2 - 2 * self.QrJoin2 * (Qr_max_bound - self.QrJoin2))) if Swrp else -0.5
        def solver2_obj(p):
            y_fn, _ = get_normal_val_and_slope(test_x_norm, p[1], p[2], p[3], p[4])
            hj1, _ = get_normal_val_and_slope(self.QrJoin1, p[1], p[2], p[3], p[4])
            b2 = (hj1 - p[0] * (self.QrJoin1 ** 2)) / self.QrJoin1 if self.QrJoin1 != 0 else 0.0
            err_srg = (Y_fine[np.argmin(X_fine)] - (b2 * Qr_min_bound + p[0] * (Qr_min_bound ** 2))) ** 2 if Srgp else 0.0
            err_sw = 0.0
            if Swrp:
                hj2_c, dh2_c = get_normal_val_and_slope(self.QrJoin2, p[1], p[2], p[3], p[4])
                p11 = dh2_c - 2 * p[5] * self.QrJoin2
                err_sw = (Y_fine[np.argmax(X_fine)] - (hj2_c - p11 * self.QrJoin2 - p[5] * (self.QrJoin2 ** 2) + p11 * Qr_max_bound + p[5] * (Qr_max_bound ** 2))) ** 2
            return np.sum((PrPLF_Nodes - y_fn) ** 2) + err_srg + err_sw
        res_s2 = minimize(solver2_obj, [-0.5, A1_s1, B1_s1, C1_s1, N1_s1, A3_init], method='SLSQP', bounds=[(None, -1e-6), (self.QrJoin1 if Srgp else None, self.QrJoin2 if Swrp else None), (None, -1e-6), (1e-6, G10_val if Swrp else None), (1.000001, None), (None, -1e-6)])
        A2_s2, A1_s2, B1_s2, C1_s2, N1_s2, A3_s2 = res_s2.x
        def global_objective_function(p):
            preds, _, _, _, _ = evaluate_master_power_curve(X_fine, p[0], p[1], p[2], p[3], p[4], p[5], p[6], self.QrJoin1, self.QrJoin2)
            return np.sum((Y_fine - preds) ** 2)
        def surge_vertex_constraint(p):
            hj1, _ = get_normal_val_and_slope(self.QrJoin1, p[2], p[3], p[4], p[5])
            b2 = (hj1 - p[0] * (self.QrJoin1 ** 2)) / self.QrJoin1
            return ( -b2 / (2.0 * p[0]) ) - self.QrJoin1 - 1e-6
        def stonewall_vertex_constraint(p):
            _, dh2 = get_normal_val_and_slope(self.QrJoin2, p[2], p[3], p[4], p[5])
            return self.QrJoin2 - ( -(dh2 - 2 * p[6] * self.QrJoin2) / (2.0 * p[6]) ) - 1e-6
        opt_res = minimize(global_objective_function, [A2_s2, N2_FIXED, A1_s2, B1_s2, C1_s2, N1_s2, A3_s2], method='SLSQP', bounds=[(None, -1e-6), (1e-6, 3.0), (self.QrJoin1 + 1e-6, self.QrJoin2 - 1e-6), (None, -1e-6), (1e-6, G10_val * 2.0), (1.000001, None), (None, -1e-6)], constraints=[{'type':'ineq','fun':surge_vertex_constraint}, {'type':'ineq','fun':stonewall_vertex_constraint}])
        A2, N2, A1, B1, C1, N1, A3 = opt_res.x
        final_preds, B2_v, B3_v, C2_v, C3_v = evaluate_master_power_curve(X_fine, A2, N2, A1, B1, C1, N1, A3, self.QrJoin1, self.QrJoin2)
        residuals = Y_fine - final_preds
        list_of_results = {
            "Variables": ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3", "N1", "N2", "QrJoin1", "QrJoin2", "Qexp", "Hexp", "QrScale", "Yscale", "QrMinMaxHead", "QrMinMaxPower", "FittingAccuracy"],
            "PowerCurve": [A1, B1, C1, A2, B2_v, C2_v, A3, B3_v, C3_v, N1, N2, self.QrJoin1, self.QrJoin2, self.q_exp, self.p_exp, self.q_scale, self.p_scale, Qr_max_bound, Qr_max_bound, 100.0 - np.mean(np.abs(residuals) / Y_fine) * 100.0]
        }
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_fine, y=Y_fine, mode='markers', marker=dict(size=6, color='green'), name='Excel Points'))
        X_smooth = np.linspace(0.0, np.max(X_fine), num=500)
        Y_smooth, _, _, _, _ = evaluate_master_power_curve(X_smooth, A2, N2, A1, B1, C1, N1, A3, self.QrJoin1, self.QrJoin2)
        fig.add_trace(go.Scatter(x=X_smooth, y=Y_smooth, mode='lines', line=dict(color='red', width=2.5), name='Fit Curve'))
        fig.add_vline(x=self.QrJoin1, line_width=1.2, line_dash="dash", line_color="orange", annotation_text="Join1 (Surge Limit)")
        fig.add_vline(x=self.QrJoin2, line_width=1.2, line_dash="dash", line_color="purple", annotation_text="Join2 (Stonewall Limit)")
        fig.update_layout(title='Reduced Power for Fitted Cases',xaxis_title='Reduced Flow (Qr)',yaxis_title='Reduced Power (Hpr)',template='plotly_white',legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig, use_container_width=True)
        return list_of_results
