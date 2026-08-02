import pandas as pd
import numpy as np
from datetime import datetime
from xml.sax.saxutils import escape


class XMLExporter:

    @staticmethod
    def infer_tabular_data_type(series):

        if pd.api.types.is_bool_dtype(series):
            return 'Boolean'

        if pd.api.types.is_integer_dtype(series):
            return 'Int32'

        if pd.api.types.is_float_dtype(series):
            return 'Double'

        if pd.api.types.is_datetime64_any_dtype(series):
            return 'DateTime'

        return 'String'

    @staticmethod
    def format_xml_scalar(value):

        if pd.isna(value):
            return ''

        if isinstance(value, (bool, np.bool_)):
            return 'true' if value else 'false'

        if isinstance(value, (np.integer, int)):
            return str(int(value))

        if isinstance(value, (np.floating, float)):
            return format(float(value), '.15g')

        if isinstance(value, datetime):
            return value.isoformat()

        return str(value)

    @classmethod
    def dataframe_to_tabular_xml(
        cls,
        df1,
        compressor_type,
        dtype
    ):

        df = df1.copy()

        if compressor_type == "Centrifugal Compressor" & str(dtype).lower() == "poly":
            column_mapping = {
                "Speed": "Speed",
                "Flow (m3/hr)": "Inlet1_ActualVolumetricFlow",
                "Head (m)": "OperatingPolyHead",
                "Efficiency (%)": "OperatingPolyEff",
                "Efficiency (%, calculated)": "OperatingPolyEff",
                "Power (kW)": "Power",
                "Power (kW, calculated)": "Power",
                "Pressure Ratio": "PresRatio"
            }

            required_cols = [
                "Speed",
                "Inlet1_ActualVolumetricFlow",
                "OperatingPolyHead",
                "OperatingPolyEff",
                "PresRatio"
            ]
            
            df.rename(columns=column_mapping, inplace=True)
            df.drop(columns=["Power"], inplace=True, errors="ignore")
            
            missing_cols = [c for c in required_cols if c not in df.columns ]
            for col in missing_cols:
                df[col] = np.nan
            df = df[required_cols]

        elif str(dtype).lower() == "custom":
            df.drop(columns=["PressureRatio"] , inplace=True , errors="ignore")
        else:
            pass
            
        parts = []

        parts.append(
            '<?xml version="1.0" encoding="utf-16"?>'
        )

        parts.append(
            '<TabularData '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
        )

        parts.append('<Columns>')

        for col_name in df.columns:

            parts.append('<TabularDataColumn>')

            parts.append(
                f'<Name>{escape(str(col_name))}</Name>'
            )

            parts.append(
                f'<DataType>'
                f'{cls.infer_tabular_data_type(df[col_name])}'
                f'</DataType>'
            )

            parts.append('<Values>')

            for value in df[col_name].tolist():

                parts.append(
                    f'<string>'
                    f'{escape(cls.format_xml_scalar(value))}'
                    f'</string>'
                )

            parts.append('</Values>')
            parts.append('</TabularDataColumn>')

        parts.append('</Columns>')
        parts.append('</TabularData>')

        return ''.join(parts)
