import pandas as pd
import re
from UnitConverter import UnitConverter

class ExcelParser:

    def __init__(self, raw_df):
        self.raw_df = raw_df

    @staticmethod
    def clean_parameter_name(name):
        n = str(name).lower()

        if 'head' in n:
            return 'Head'

        if 'eff' in n:
            return 'Efficiency'

        if 'power' in n or 'bhp' in n or 'kw' in n:
            return 'Power'

        return str(name)

    def detect_triplet_blocks(self):

        blocks = []

        rows, cols = self.raw_df.shape

        for r in range(rows):

            for c in range(cols - 2):

                v1 = str(self.raw_df.iloc[r, c]).strip().lower()
                v2 = str(self.raw_df.iloc[r, c + 1]).strip().lower()
                v3 = str(self.raw_df.iloc[r, c + 2]).strip()

                if v1 == "speed" and "flow" in v2:

                    p = self.clean_parameter_name(v3)

                    if p.lower() != "nan":

                        speed_unit = ""
                        flow_unit = ""
                        value_unit = ""

                        if r + 1 < rows:

                            speed_unit = str(
                                self.raw_df.iloc[r + 1, c]
                            ).strip()

                            flow_unit = str(
                                self.raw_df.iloc[r + 1, c + 1]
                            ).strip()

                            value_unit = str(
                                self.raw_df.iloc[r + 1, c + 2]
                            ).strip()

                        blocks.append(
                            {
                                "parameter": p,
                                "header_row": r,
                                "start_col": c,
                                "speed_unit": speed_unit,
                                "flow_unit": flow_unit,
                                "value_unit": value_unit,
                            }
                        )

        uniq = []
        seen = set()

        for b in blocks:

            if b["start_col"] not in seen:
                seen.add(b["start_col"])
                uniq.append(b)

        return uniq

    def detect_property_block(self):

        rows, cols = self.raw_df.shape

        for r in range(rows):

            for c in range(cols - 2):

                v1 = str(self.raw_df.iloc[r, c]).strip().lower()
                v2 = str(self.raw_df.iloc[r, c + 1]).strip().lower()
                v3 = str(self.raw_df.iloc[r, c + 2]).strip().lower()

                if (
                    v1 == "parameter"
                    and v2 == "value"
                    and v3 == "units"
                ):
                    return {
                        "header_row": r,
                        "start_col": c
                    }

        return None
        
    def extract_property_block(self, block):

        if block is None:
            return pd.DataFrame(
                columns=['Parameter', 'Value', 'Units']
            )

        r = block['header_row']
        c = block['start_col']

        rows_out = []

        for row in range(r + 1, len(self.raw_df)):

            param = self.raw_df.iloc[row, c]
            value = self.raw_df.iloc[row, c + 1]
            units = self.raw_df.iloc[row, c + 2]

            if pd.isna(param) or str(param).strip() == '':
                break

            p_name = str(param).strip()

            v_val = (
                value.item()
                if hasattr(value, 'item')
                else value
            )

            u_str = (
                ''
                if pd.isna(units)
                else str(units).strip()
            )

            if 'diameter' in p_name.lower():

                try:

                    if isinstance(v_val, str):
                        v_val = re.split(
                            r'[,/]',
                            v_val
                        )[0].strip()

                    converted_val, success = (
                        UnitConverter.convert_unit(
                            float(v_val),
                            u_str,
                            UnitConverter.DIAMETER_TO_M
                        )
                    )

                    if success:
                        v_val = converted_val
                        u_str = 'm'

                except (ValueError, TypeError):
                    pass

            elif 'pressure' in p_name.lower():

                try:

                    if isinstance(v_val, str):
                        v_val = re.split(
                            r'[,/]',
                            v_val
                        )[0].strip()

                    converted_val, success = (
                        UnitConverter.convert_pressure_to_kg_cm2a(
                            float(v_val),
                            u_str
                        )
                    )

                    if success:
                        v_val = converted_val
                        u_str = 'kg/cm2a'

                except (ValueError, TypeError):
                    pass

            elif 'temperature' in p_name.lower():

                try:

                    if isinstance(v_val, str):
                        v_val = re.split(
                            r'[,/]',
                            v_val
                        )[0].strip()

                    converted_val, success = (
                        UnitConverter.convert_temperature_to_c(
                            float(v_val),
                            u_str
                        )
                    )

                    if success:
                        v_val = converted_val
                        u_str = 'deg C'

                except (ValueError, TypeError):
                    pass

            rows_out.append(
                {
                    'Parameter': p_name,
                    'Value': v_val,
                    'Units': u_str
                }
            )

        return pd.DataFrame(
            rows_out,
            columns=['Parameter', 'Value', 'Units']
        )
    def extract_block_data(self, block):

        r = block['header_row']
        c = block['start_col']

        data = []

        for row in range(r + 1, len(self.raw_df)):

            try:

                sp = float(self.raw_df.iloc[row, c])
                fl = float(self.raw_df.iloc[row, c + 1])
                val = float(self.raw_df.iloc[row, c + 2])

                data.append([sp, fl, val])

            except Exception:
                pass

        df = pd.DataFrame(
            data,
            columns=['Speed', 'Flow', 'Value']
        )

        if df.empty:
            return df, False, False

        flow_converted = False
        value_converted = False

        df['Flow'], flow_converted = (
            UnitConverter.convert_unit(
                df['Flow'],
                block['flow_unit'],
                UnitConverter.FLOW_TO_M3HR
            )
        )

        param = block['parameter']

        if param == 'Head':

            df['Value'], value_converted = (
                UnitConverter.convert_unit(
                    df['Value'],
                    block['value_unit'],
                    UnitConverter.HEAD_TO_M
                )
            )

        elif param == 'Power':

            df['Value'], value_converted = (
                UnitConverter.convert_unit(
                    df['Value'],
                    block['value_unit'],
                    UnitConverter.POWER_TO_KW
                )
            )

        elif param == 'Efficiency':

            df['Value'], value_converted = (
                UnitConverter.convert_unit(
                    df['Value'],
                    block['value_unit'],
                    UnitConverter.EFF_TO_PCT
                )
            )

        return df, flow_converted, value_converted