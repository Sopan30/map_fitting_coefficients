import io
import os
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

Pwr = True  
Swr = True 

def round2(value):
    if value == 0:
        return 0.0
    decimals = -int(math.floor(math.log10(abs(value)))) + 1
    return round(value, decimals)

def get_normal_val_and_slope(x, a1, b1, c1, n1, pwr=True):
    if pwr:
        val = c1 + b1 * np.abs(x - a1) ** n1
        slope = b1 * n1 * np.abs(x - a1) ** (n1 - 1) * np.sign(x - a1)
    else:
        val = c1 - np.exp(a1 * x ** n1 + b1)
        slope = -a1 * n1 * x ** (n1 - 1) * np.exp(a1 * x ** n1 + b1)
    return val, slope

class HpFitting:
    def __init__(self):
        # Raw Data Arrays
        self.Nn = None
        self.Qn = None
        self.Hp = None

        self.q_exp = 1.0
        self.hp_exp = 2.0
        self.q_scale = 1.0
        self.hp_scale = 1.0

        self.QrJoin1 = 0.18
        self.QrJoin2 = 0.24

    def execute_pipeline(self,df):
        self.Nn = df['Nn'].to_numpy()
        self.Qn = df['Qn'].to_numpy()
        self.Hp = df['Hp'].to_numpy()       
        q_grid = np.arange(0.8, 1.2 + 1e-9, 0.01)
        h_grid = np.arange(1.8, 2.2 + 1e-9, 0.01)
        best_q, best_h = 1.0, 2.0
        min_adjacent_dist_sq = float('inf')
        
        for h_val in h_grid:
            for q_val in q_grid:
                X = self.Qn / (self.Nn ** q_val)
                Y = self.Hp / (self.Nn ** h_val)
                sort_idx = np.argsort(X)
                dist_sq = np.diff(X[sort_idx])**2 + np.diff(Y[sort_idx])**2
                total_dist = np.sum(dist_sq)
                if total_dist < min_adjacent_dist_sq:
                    min_adjacent_dist_sq = total_dist
                    best_q, best_h = q_val, h_val
                    
        self.q_exp, self.hp_exp = best_q, best_h

        Qr_coarse = self.Qn / (self.Nn ** self.q_exp)
        Hpr_coarse = self.Hp / (self.Nn ** self.hp_exp)
        Qnrmax, Hpnrmax = np.max(Qr_coarse), np.max(Hpr_coarse)
        
        if Hpnrmax > Qnrmax:
            self.q_scale = round2(Hpnrmax / Qnrmax)
            self.hp_scale = 1.0
        elif Hpnrmax < Qnrmax:
            self.hp_scale = round2(Qnrmax / Hpnrmax)
            self.q_scale = 1.0
        else:
            self.q_scale, self.hp_scale = 1.0, 1.0

        X_coarse_scaled = Qr_coarse * self.q_scale
        Y_coarse_scaled = Hpr_coarse * self.hp_scale
        sort_indices = np.argsort(X_coarse_scaled)
        node_x_init = np.linspace(np.min(X_coarse_scaled), np.max(X_coarse_scaled), num=11)
        node_y_init = np.interp(node_x_init, X_coarse_scaled[sort_indices], Y_coarse_scaled[sort_indices])
        initial_vector_plf = np.concatenate(([self.q_exp, self.hp_exp], node_y_init))
        def plf_joint_objective(params):
            q_e, h_e, node_y = float(params[0]), float(params[1]), params[2:]
            X_curr = (self.Qn / (self.Nn ** q_e)) * self.q_scale
            Y_curr = (self.Hp / (self.Nn ** h_e)) * self.hp_scale
            sort_loop = np.argsort(X_curr)
            X_sorted = X_curr[sort_loop]
            Y_sorted = Y_curr[sort_loop]
            node_x_curr = np.linspace(np.min(X_sorted), np.max(X_sorted), num=11)
            Y_fit_curr = np.interp(X_sorted, node_x_curr, node_y)
            return np.sum((Y_sorted - Y_fit_curr) ** 2)
        res_plf = minimize(plf_joint_objective, initial_vector_plf, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        self.q_exp, self.hp_exp = float(res_plf.x[0]), float(res_plf.x[1])
        X_fine = (self.Qn / (self.Nn ** self.q_exp)) * self.q_scale
        Y_fine = (self.Hp / (self.Nn ** self.hp_exp)) * self.hp_scale

        Qr_min_bound = np.min(X_fine)
        Qr_max_bound = np.max(X_fine)
        Qr_total_span = Qr_max_bound - Qr_min_bound
        
        self.QrJoin1 = Qr_min_bound + 0.25 * Qr_total_span
        self.QrJoin2 = Qr_max_bound - 0.25 * Qr_total_span

        test_x_norm = np.linspace(self.QrJoin1, self.QrJoin2, num=4)
        HprPLF_QrNodes = np.interp(test_x_norm, X_fine, Y_fine)
        HprPLF_Qrmin = Y_fine[np.argmin(X_fine)]
        HprPLF_Qrmax = Y_fine[np.argmax(X_fine)]
        
        N1_init = 2.0
        C1_init = HprPLF_Qrmin
        denom_guess = (1.0 - ((HprPLF_QrNodes[0] - C1_init) / (HprPLF_QrNodes[-1] - C1_init)) ** (1.0 / N1_init))
        if abs(denom_guess) < 1e-6: denom_guess = 1e-6
        A1_init = (self.QrJoin1 - self.QrJoin2 * ((HprPLF_QrNodes[0] - C1_init) / (HprPLF_QrNodes[-1] - C1_init)) ** (1.0 / N1_init)) / denom_guess
        B1_init = (HprPLF_QrNodes[0] - C1_init) / max(1e-6, abs(self.QrJoin1 - A1_init) ** N1_init)
        
        def step1_normal_obj(p):
            a1, b1, c1, n1 = p
            y_fit, _ = get_normal_val_and_slope(test_x_norm, a1, b1, c1, n1, pwr=Pwr)
            return np.sum((HprPLF_QrNodes - y_fit) ** 2)

        res_s1 = minimize(step1_normal_obj, [A1_init, B1_init, C1_init, N1_init], method='L-BFGS-B')
        A1_s1, B1_s1, C1_s1, N1_s1 = res_s1.x
        
        h_j2, dH_j2 = get_normal_val_and_slope(self.QrJoin2, A1_s1, B1_s1, C1_s1, N1_s1, pwr=Pwr)
        A3_init = (HprPLF_Qrmax - h_j2 - dH_j2 * (Qr_max_bound - self.QrJoin2)) / max(1e-6, (Qr_max_bound**2 - self.QrJoin2**2 - 2*self.QrJoin2*(Qr_max_bound - self.QrJoin2)))
        A2_init, N2_init = 1.0, 1.0

        def global_objective_function(params):
            A2, N2, A1, B1, C1, N1, A3 = params
            H_at_J1, dH_at_J1 = get_normal_val_and_slope(self.QrJoin1, A1, B1, C1, N1, pwr=Pwr)
            H_at_J2, dH_at_J2 = get_normal_val_and_slope(self.QrJoin2, A1, B1, C1, N1, pwr=Pwr)
            denom_srg = A2 * N2 * (self.QrJoin1 ** (N2 - 1)) if self.QrJoin1 != 0 else 1.0
            C2 = H_at_J1 - (dH_at_J1 / denom_srg)
            B2 = np.log(np.maximum(1e-15, C2 - H_at_J1)) - A2 * (self.QrJoin1 ** N2)
            B3 = dH_at_J2 - 2 * A3 * self.QrJoin2
            C3 = H_at_J2 - B3 * self.QrJoin2 - A3 * (self.QrJoin2 ** 2)
            Y_pred = np.zeros_like(X_fine)
            srg_mask = X_fine <= self.QrJoin1
            Y_pred[srg_mask] = C2 - np.exp(A2 * (X_fine[srg_mask] ** N2) + B2)
            norm_mask = (X_fine > self.QrJoin1) & (X_fine < self.QrJoin2)
            Y_pred[norm_mask] = C1 + B1 * np.abs(X_fine[norm_mask] - A1)**N1
            sw_mask = X_fine >= self.QrJoin2
            Y_pred[sw_mask] = C3 + B3 * X_fine[sw_mask] + A3 * (X_fine[sw_mask] ** 2)
            return np.sum((Y_fine - Y_pred) ** 2)

        bounds_solver = [(0.000001, None),                 # A2 must be > 0.000001 (Relation 3)
                         (0.000001, None),                 # N2 must be > 0.000001 (Relation 3)
                         (None, self.QrJoin1 - 0.000001),  # A1 must be < QrJoin.Surge - Delta (Relation 1)
                         (None, -0.000001),                # B1 must be < 0 - Delta (Relation 1)
                         (0.000001, None),                 # C1 must be > 0 + Delta (Relation 3)
                         (1.000001, None),                 # N1 must be > 1 + Delta (Relation 3)
                         (None, -0.000001)]                # A3 must be < 0 - Delta (Relation 1)
        def surge_c2_constraint(params):
            A2, N2, A1, B1, C1, N1, A3 = params
            H_at_J1, dH_at_J1 = get_normal_val_and_slope(self.QrJoin1, A1, B1, C1, N1, pwr=Pwr)
            denom_srg = A2 * N2 * (self.QrJoin1 ** (N2 - 1)) if self.QrJoin1 != 0 else 1.0
            C2 = H_at_J1 - (dH_at_J1 / denom_srg)
            return C2 - 0.000001  # Must be > 0.000001
        def surge_peak_constraint(params):
            A2,N2,_,_,_,_,_ = params
            return N2 * (1.0 + A2 * (Qr_min_bound ** N2)) - 1.0 - 0.000001    
        def normal_n21_constraint(params):
            _,_,A1,_,_,N1,_ = params
            return N1 * (1.0 + A1 *(self.QrJoin1 ** N1)) - 1.000001  # Must be > 1.000001
        constraints_solver = [{'type': 'ineq', 'fun': surge_c2_constraint},
                              {'type': 'ineq', 'fun': surge_peak_constraint},
                              {'type': 'ineq', 'fun': normal_n21_constraint}]
        initial_vector_step6 = [A2_init, N2_init, A1_s1, B1_s1, C1_s1, N1_s1, A3_init]
        opt_res = minimize(global_objective_function, initial_vector_step6, method='SLSQP',bounds=bounds_solver, constraints=constraints_solver,options={'ftol': 1e-8, 'maxiter': 1000})
        A2, N2, A1, B1, C1, N1, A3 = opt_res.x
        H_at_J1, dH_at_J1 = get_normal_val_and_slope(self.QrJoin1, A1, B1, C1, N1, pwr=Pwr)
        H_at_J2, dH_at_J2 = get_normal_val_and_slope(self.QrJoin2, A1, B1, C1, N1, pwr=Pwr)
        C2_f = H_at_J1 - (dH_at_J1 / (A2 * N2 * (self.QrJoin1 ** (N2 - 1))))
        B2_f = np.log(np.maximum(1e-15, C2_f - H_at_J1)) - A2 * (self.QrJoin1 ** N2)
        B3_f = dH_at_J2 - 2 * A3 * self.QrJoin2
        C3_f = H_at_J2 - B3_f * self.QrJoin2 - A3 * (self.QrJoin2 ** 2)
        global_preds = np.zeros_like(X_fine)
        srg_m = X_fine <= self.QrJoin1;global_preds[srg_m] = C2_f - np.exp(A2 * (X_fine[srg_m] ** N2) + B2_f)
        norm_m = (X_fine > self.QrJoin1) & (X_fine < self.QrJoin2);global_preds[norm_m] = C1 + B1 * np.abs(X_fine[norm_m] - A1)**N1
        sw_m = X_fine >= self.QrJoin2; global_preds[sw_m] = C3_f + B3_f * X_fine[sw_m] + A3 * (X_fine[sw_m] ** 2)
        residuals = Y_fine - global_preds
        # total_sse = opt_res.fun
        # total_rmse = np.sqrt(np.mean(residuals ** 2))
        # max_absolute_error = np.max(np.abs(residuals))
        list_of_results = {"Variables": ["A1", "B1", "C1", "A2", "B2", "C2","A3","B3", "C3","N1","N2","QrJoin1","QrJoin2","Qexp","Hexp","QrScale","Yscale","QrMinMaxHead","QrMinMaxPower","FittingAccuracy"],
                                   "HeadCurve": [A1, B1, C1, A2, B2_f, C2_f, A3, B3_f, C3_f, N1, N2, self.QrJoin1, self.QrJoin2, self.q_exp, self.hp_exp, self.q_scale, self.hp_scale, Qr_min_bound,Qr_min_bound, 100-np.mean(abs(residuals)/Y_fine)*100]}
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_fine, y=Y_fine, mode='markers', marker=dict(size=6, color='blue', symbol='diamond'), name='Original'))
        X_smooth = np.linspace(np.min(X_fine), np.max(X_fine), num=400)
        Y_smooth = np.zeros_like(X_smooth)
        s_m = X_smooth <= self.QrJoin1; Y_smooth[s_m] = C2_f - np.exp(A2 * (X_smooth[s_m] ** N2) + B2_f)
        n_m = (X_smooth > self.QrJoin1) & (X_smooth < self.QrJoin2); Y_smooth[n_m] = C1 + B1 * np.abs(X_smooth[n_m] - A1)**N1
        w_m = X_smooth >= self.QrJoin2; Y_smooth[w_m] = C3_f + B3_f * X_smooth[w_m] + A3 * (X_smooth[w_m] ** 2)
        fig.add_trace(go.Scatter(x=X_smooth, y=Y_smooth, mode='lines', line=dict(color='red', width=2.5), name='Fit'))
        fig.add_vline(x=self.QrJoin1, line_width=1.2, line_dash="dash", line_color="orange", annotation_text="Join1 (Surge Limit)")
        fig.add_vline(x=self.QrJoin2, line_width=1.2, line_dash="dash", line_color="purple", annotation_text="Join2 (Stonewall Limit)")
        fig.update_layout(title='Reduced Polytropic Head for Fitted Cases',xaxis_title='Reduced Flow (Qr)',yaxis_title='Reduced Head (Hpr)',template='plotly_white',legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig, use_container_width=True)
        return list_of_results
