import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import os
import re
from ExcelParser import ExcelParser
from CurveFitter import CurveFitter
from GasCalculator import GasCalculator
from UnitConverter import UnitConverter
from XMLExporter import XMLExporter
from WorkbookExporter import WorkbookExporter
from PlotManager import PlotManager
from HpFitting import HpFitting
from PFitting import PFitting

gas_calc = GasCalculator()
curve_fitter = CurveFitter()
plot_manager = PlotManager(curve_fitter)

st.set_page_config(page_title='Compressor Curve Regression', layout='wide')
st.title('Compressor Curve Regression Tool')
compressor_type = st.radio("Select Compressor Type",["Centrifugal Compressor", "Multi-Side Stream Compressor"],horizontal=True)
if compressor_type != "Centrifugal Compressor":
    st.warning(
        """
        ⚠ **Multi-Side Stream Compressor**
        - Flow values will be converted to **Mass Flow (kg/hr)** using the operating conditions provided in the Excel file.
        - Please ensure all Operating Condition values are correct, especially:
            - Pressure
            - Temperature
            - Molecular Weight (MW)
            - Compressibility (Z)
            - Isentropic Exponent (k)
        Incorrect operating conditions will result in incorrect Mass Flow and performance calculations.
        """)
else:
    st.info(
        """
        ℹ **Centrifugal Compressor Selected**
        - Please ensure all Operating Condition values are correct, especially:
            - Pressure
            - Temperature
            - Molecular Weight (MW)
            - Compressibility (Z)
            - Isentropic Exponent (k)
        Incorrect operating conditions will result in incorrect Mass Flow and performance calculations.
        """)

method = st.sidebar.selectbox('Regression Method',
    ['Auto Best Fit','Linear','Quadratic','Cubic','4th Order','5th Order','Spline'])
points = st.sidebar.slider('Number of Points',10,50,15)
file = st.file_uploader('Upload Workbook', type=['xlsx'])

if file:
    xls = pd.ExcelFile(file)
    output = BytesIO()
    r2_rows = []
    overview = []
    property_rows = []
    stage_xml_exports = []
    final_df = pd.DataFrame()
    side_stream_df = pd.DataFrame()
    NumberOfStages = 1
    fatal_error = None

    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            scaling_rows = []
            for stage in xls.sheet_names:
                acoustic_vel = None
                spec_vol = None
                temp2 = None
                st.header(stage)
                try:
                    raw = pd.read_excel(file, sheet_name=stage, header=None)
                    parser = ExcelParser(raw)
                    blocks = parser.detect_triplet_blocks()

                    prop_block = parser.detect_property_block()
                    prop_df = parser.extract_property_block(prop_block)
                    gas_props = None
                    
                    if not prop_df.empty:
                        st.subheader('Operating Conditions')
                        st.dataframe(prop_df, use_container_width=True)
                        for _, row in prop_df.iterrows():
                            property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                        gas_props = gas_calc.gas_properties_from_df(prop_df)
                        

                        if gas_props is not None:
                            t_k = UnitConverter.c_to_k(gas_props['temperature_c'])
                            p_pa = UnitConverter.kg_cm2a_to_pa(gas_props['pressure_kg_cm2a'])
                            
                            acoustic_vel = np.sqrt((gas_props['k'] * gas_props['z'] * gas_calc.R_UNIVERSAL * t_k) / gas_props['mw'])
                            spec_vol = (gas_props['z'] * gas_calc.R_UNIVERSAL * t_k) / (p_pa * gas_props['mw'])
                            
                            if compressor_type == "Centrifugal Compressor":    
                                speed_factor = (2 * np.pi * gas_props['diameter_m']) / (60.0 * acoustic_vel)
                                flow_factor = 1.0 / (acoustic_vel * gas_props['diameter_m']**2)
                                head_factor = 1000.0 / (acoustic_vel**2)
                                power_factor = (1000.0 * 30 * spec_vol) / (np.pi * acoustic_vel**2 * gas_props['diameter_m']**3)
                                
                                derived_df = pd.DataFrame([
                                    {'Parameter': 'Acoustic Velocity', 'Value': round(acoustic_vel, 2), 'Units': 'm/s'},
                                    {'Parameter': 'Specific Volume', 'Value': round(spec_vol, 5), 'Units': 'm3/kg'},
                                    {'Parameter': 'Rotational Speed', 'Value': f"{speed_factor:.5e}", 'Units': 'rpm'},
                                    {'Parameter': 'Volumetric Flow', 'Value': f"{flow_factor:.5e}", 'Units': 'm3/s'},
                                    {'Parameter': 'Polytropic Head', 'Value': f"{head_factor:.5e}", 'Units': 'kJ/kg'},
                                    {'Parameter': 'Power', 'Value': f"{power_factor:.5e}", 'Units': 'kW'}
                                ])
                                st.dataframe(derived_df, use_container_width=True)
                                
                                for _, row in derived_df.iterrows():
                                    property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                        else:
                            st.info('Could not read Pressure/Temperature/MW/Compressibility as numbers — '
                                    'skipping derived calculations and missing-parameter steps.')
                    else:
                        st.warning(f'No operating-conditions block found in {stage}')

                    if not blocks:
                        st.warning(f'No blocks found in {stage}')
                        overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                          'Calculated Parameter': '', 'Status': 'no blocks found'})
                        continue

                    clean_blocks = []
                    for b in blocks:
                        looks_numeric = False
                        for u in (b['speed_unit'], b['flow_unit'], b['value_unit']):
                            try:
                                float(u)
                                looks_numeric = True
                            except (ValueError, TypeError):
                                pass
                        if looks_numeric:
                            st.warning(f"Skipping a detected block for '{b['parameter']}' at column "
                                       f"{b['start_col']} — its units row looks like data, not units "
                                       f"('{b['speed_unit']}', '{b['flow_unit']}', '{b['value_unit']}'). "
                                       f"This usually means a duplicated/mislabeled header row in the source sheet.")
                        else:
                            clean_blocks.append(b)
                    blocks = clean_blocks

                    if not blocks:
                        st.warning(f'No valid blocks remained after validation in {stage}')
                        overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                          'Calculated Parameter': '', 'Status': 'blocks failed validation'})
                        continue

                    block_summary = pd.DataFrame([
                        {'parameter': b['parameter'], 'speed_unit': b['speed_unit'],
                         'flow_unit': b['flow_unit'], 'value_unit': b['value_unit']}
                        for b in blocks
                    ])
                    st.dataframe(block_summary)

                    stage_models = {}
                    stage_parameters = []

                    tabs = st.tabs([b['parameter'] for b in blocks])

                    for tab, block in zip(tabs, blocks):
                        with tab:
                            param = block['parameter']
                            df, flow_conv, val_conv = parser.extract_block_data(block)
                            stage_parameters.append(param)

                            unit_note = []
                            if not flow_conv:
                                unit_note.append(f"flow unit '{block['flow_unit']}' not recognized, left as-is")
                            if not val_conv:
                                unit_note.append(f"{param} unit '{block['value_unit']}' not recognized, left as-is")
                            if unit_note:
                                st.caption('⚠ ' + '; '.join(unit_note))
                            else:
                                target_unit = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(param, '')
                                st.caption(f"Converted to Flow: m³/hr, {param}: {target_unit}")

                            fig = plot_manager.build_parameter_plot(df=df,
                                                                    param=param,
                                                                    method=method,
                                                                    points=points,
                                                                    stage_models=stage_models,
                                                                    r2_rows=r2_rows,
                                                                    stage_name=stage
                                                                    )
                            st.plotly_chart(fig, use_container_width=True)

                    speeds = set()
                    for p in stage_models:
                        speeds.update(stage_models[p].keys())

                    export_rows = []
                    computed_param_name = None
                    stage_status = 'ok'

                    for speed in sorted(speeds):
                        available = []
                        available_params = []
                        for p in stage_models:
                            if speed in stage_models[p]:
                                available.append(stage_models[p][speed])
                                available_params.append(p)

                        if len(available) < 1:
                            continue

                        common_min = max(m['xmin'] for m in available)
                        common_max = min(m['xmax'] for m in available)

                        if common_max <= common_min:
                            err_msg = f"Flow values do not overlap for Stage: **{stage}** at Speed: **{speed}** across parameters ({', '.join(available_params)})."
                            st.error(err_msg)
                            stage_status = f"error: flow overlap failed at speed {speed}"
                            continue

                        common_flow = np.linspace(common_min, common_max, points)

                        temp = {'Speed': [speed] * points, 'Flow (m3/hr)': common_flow}

                        predicted = {}
                        for p in stage_models:
                            if speed in stage_models[p]:
                                vals = curve_fitter.predict(stage_models[p][speed], common_flow)
                                predicted[p] = vals
                                unit_label = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(p, '')
                                temp[f'{p} ({unit_label})'] = vals
                        
                        if gas_props is not None and acoustic_vel is not None:
                            try:
                                rho = gas_calc.gas_density_kg_m3(gas_props['pressure_kg_cm2a'], gas_props['temperature_c'],
                                                         gas_props['mw'], gas_props['z'])
                                mass_flow_kg_s = common_flow * rho / 3600.0
                                name, values = gas_calc.compute_missing_parameter(predicted, mass_flow_kg_s)
                                if name is not None:
                                    computed_param_name = name
                                    unit_label = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(name, '')
                                    temp[f'{name} ({unit_label}, calculated)'] = values
                                    predicted[name] = values
                                
                                head_meters = predicted['Head']
                                head_kj_kg = (head_meters * gas_calc.G) / 1000.0
                                eff_pct = predicted['Efficiency']
                                
                                k_val = gas_props['k']
                                L5 = (k_val * (eff_pct / 100.0)) / (k_val - 1.0)
                                M5 = (acoustic_vel ** 2) / k_val
                                
                                pressure_ratio = (1.0 + (1000.0 * head_kj_kg) / (M5 * L5)) ** L5
                                temp['Pressure Ratio'] = pressure_ratio
                                if compressor_type != "Centrifugal Compressor":
                                    temp2 = pd.DataFrame(temp).copy()
                                    temp2.drop(columns=['Flow (m3/hr)'],inplace=True)
                                    mass_flow_kg_hr = mass_flow_kg_s * 3600
                                    temp2.insert(1,f'Stage{NumberOfStages}_MassFlow',mass_flow_kg_hr)
                                    temp2.rename(columns={'Head (m)':f'Stage{NumberOfStages}_OperatingPolyHead',
                                                          'Head (m, calculated)':f'Stage{NumberOfStages}_OperatingPolyHead',
                                                          'Power (kW, calculated)':f'Stage{NumberOfStages}_OperatingShaftPower',
                                                          'Power (kW)':f'Stage{NumberOfStages}_OperatingShaftPower',
                                                          'Efficiency (%)':f'Stage{NumberOfStages}_OperatingPolyEfficiency',
                                                          'Efficiency (%, calculated)':f'Stage{NumberOfStages}_OperatingPolyEfficiency',
                                                          'Pressure Ratio' : 'PressureRatio'
                                                          },inplace=True)
                                
                            except (ZeroDivisionError, ValueError, KeyError) as e:
                                st.warning(f"Could not compute missing parameter/pressure ratio for {stage} @ speed {speed}: {e}")

                        if compressor_type == "Centrifugal Compressor":
                            export_rows.append(pd.DataFrame(temp))
                        else:
                            if temp2 is not None:
                                export_rows.append(temp2)

                    if computed_param_name and stage_status == 'ok':
                        st.success(f"Calculated missing parameter **{computed_param_name}** and **Pressure Ratio** for {stage} "
                                   f"using gas density from Operating Conditions.")

                    if export_rows:
                        final_df = pd.concat(export_rows, ignore_index=True)
                        WorkbookExporter.write_stage_sheet(writer,stage,final_df)
                        # if gas_props is not None and compressor_type == "Centrifugal Compressor":
                        #     scaling_result = curve_fitter.calculate_scaling_factors(final_df, gas_props, acoustic_vel, spec_vol)
                        #     if scaling_result is not None:
                        #         final_df, scaling_info = scaling_result
                        #         scaling_info["Stage"] = stage
                        #         scaling_rows.append(scaling_info)
                        # if scaling_rows:
                        #     pd.DataFrame(scaling_rows).to_excel(writer, sheet_name="Scaling_Factors", index=False)
                        if compressor_type == "Centrifugal Compressor":
                            non_df = final_df.copy()
                            column_mapping = {
                                                "Speed": "Speed",
                                                "Flow (m3/hr)": "flow",
                                                "Head (m)": "head",
                                                "Efficiency (%)": "eff",
                                                "Efficiency (%, calculated)": "eff",
                                                "Power (kW)": "P",
                                                "Power (kW, calculated)": "P",
                                                "Pressure Ratio": "PR"
                            }
                            non_df.rename(columns = column_mapping , inplace=True)
                            non_df["Q"] = non_df["flow"]/3600
                            non_df["H"] = (non_df["head"]* gas_calc.G) / 1000.0
                            non_df["Nn"] = non_df["Speed"] * speed_factor
                            non_df["Qn"] = non_df["Q"] * flow_factor
                            non_df["Hp"] = non_df["H"] * head_factor
                            non_df["Pn"] = non_df["P"] * power_factor
                            hp_coefficients = HpFitting().execute_pipeline(non_df)
                            p_coefficients = PFitting().run_calibrations(df = non_df, QrHpScaleFtr = hp_coefficients["HeadCurve"][15])
                            combine_coefficients = {
                                "Variables": hp_coefficients["Variables"],
                                "HeadCurve": hp_coefficients["HeadCurve"],
                                "PowerCurve": p_coefficients["PowerCurve"]
                            }
                            coefficients_df = pd.DataFrame(combine_coefficients)
                            st.dataframe(coefficients_df, use_container_width=True)
                            WorkbookExporter.write_stage_sheet(writer,f'{stage}_Coefficients',coefficients_df)
                            xml_content = XMLExporter.dataframe_to_tabular_xml(final_df,compressor_type,'poly')
                            coefficients_xml = XMLExporter.dataframe_to_tabular_xml(final_df,compressor_type,'coeff')
                        else:
                            xml_content = XMLExporter.dataframe_to_tabular_xml(final_df,compressor_type,'custom')
                        # stage_xml_exports.append({'Stage': stage, 'XML': xml_content})
                        if coefficients_xml:
                            stage_xml_exports.append({"Attribute":"PolyPerformanceCoeff","Stage": stage, "XML": coefficients_xml})
                        stage_xml_exports.append({"Attribute": ("PolyPerformanceData" if compressor_type == "Centrifugal Compressor"
                                                                else f"Stage{NumberOfStages}_CustomPerformanceDataInput"),"Stage": stage,"XML": xml_content})
                            
                    if compressor_type != "Centrifugal Compressor":
                        final_df.insert(0,'NumberOfStages',[NumberOfStages] * len(final_df))
                        remap={f'Stage{NumberOfStages}_MassFlow':'Stage1_MassFlow',
                               f'Stage{NumberOfStages}_OperatingPolyHead':'OperatingPolyHead',
                               f'Stage{NumberOfStages}_OperatingShaftPower':'OperatingShaftPower',
                               f'Stage{NumberOfStages}_OperatingPolyEfficiency':'OperatingPolyEfficiency'}
                        final_df.rename(columns=remap,inplace=True)
                        final_df.drop(columns=['OperatingShaftPower'],inplace=True,errors='ignore')
                        side_stream_df = pd.concat([side_stream_df,final_df],ignore_index=True)
                        NumberOfStages = NumberOfStages + 1

                    overview.append({
                        'Stage': stage,
                        'Parameters': ','.join(stage_parameters),
                        'Blocks Found': len(blocks),
                        'Calculated Parameter': computed_param_name or '',
                        'Status': stage_status
                    })
                    

                except Exception as e:
                    st.error(f"Error processing '{stage}': {e}")
                    overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                      'Calculated Parameter': '', 'Status': f'error: {e}'})

            if compressor_type != "Centrifugal Compressor":
                if not side_stream_df.empty:
                    WorkbookExporter.write_side_stream_sheet(writer,side_stream_df)
                    xml=XMLExporter.dataframe_to_tabular_xml(side_stream_df,compressor_type,'side')
                    stage_xml_exports.append({'Stage': 'SideStreamPolyPerformanceData', 'XML': xml})

            WorkbookExporter.write_all_summary_sheets(writer,stage_xml_exports,
                                                      property_rows,
                                                      overview,
                                                      r2_rows,
                                                      scaling_rows
                                                      )
    except Exception as e:
        fatal_error = e

    if fatal_error is not None:
        st.error(f"Could not build the output workbook: {fatal_error}")
        st.info("Nothing to download — see the error above.")
    else:
        output.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.splitext(file.name)[0]
        output_filename = f"{input_filename}_Map_Fiting_{timestamp}.xlsx"

        st.download_button(
            label="Download Regression Workbook",
            data=output.getvalue(),
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if stage_xml_exports:
            st.subheader('Stage XML Downloads')
            for entry in stage_xml_exports:
                safe_stage_name = re.sub(r'[^A-Za-z0-9._-]+', '_', entry['Stage'])
                st.download_button(
                    label=f"Download XML for {entry['Stage']}",
                    data=entry['XML'],
                    file_name=f"{input_filename}_{safe_stage_name}_{timestamp}.txt",
                    mime="application/octet-stream"
                )

            with st.expander('XML Export Sheet Preview', expanded=False):
                xml_sheet_df = pd.DataFrame(stage_xml_exports)
                st.dataframe(xml_sheet_df, use_container_width=True)
        else:
            st.info('No tabular data was generated for XML export yet.')
