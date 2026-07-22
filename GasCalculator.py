import numpy as np
from UnitConverter import UnitConverter


class GasCalculator:

    R_UNIVERSAL = 8314.462618   # J/(kmol.K)
    G = 9.80665                 # m/s²

    @classmethod
    def gas_density_kg_m3(
        cls,
        pressure_kg_cm2a,
        temperature_c,
        mw,
        z
    ):
        p_pa = UnitConverter.kg_cm2a_to_pa(pressure_kg_cm2a)
        t_k = UnitConverter.c_to_k(temperature_c)

        return (p_pa * mw) / (z * cls.R_UNIVERSAL * t_k)

    @staticmethod
    def gas_properties_from_df(prop_df):

        lookup = {}

        for _, row in prop_df.iterrows():
            name = str(row['Parameter']).lower()

            if 'pressure' in name:
                lookup['pressure'] = row['Value']

            elif 'temperature' in name:
                lookup['temperature'] = row['Value']

            elif ('molecular weight' in name or 'mw' in name):
                lookup['mw'] = row['Value']

            elif ('compressibility' in name or name.strip() == 'z'):
                lookup['z'] = row['Value']

            elif ('isentropic' in name or name.strip() == 'k'):
                lookup['k'] = float(row['Value'])

                if np.isclose(lookup['k'], 1.0):
                    lookup['k'] = 1.001

            elif 'diameter' in name:
                lookup['diameter'] = row['Value']

        try:
            return {
                'pressure_kg_cm2a': float(lookup['pressure']),
                'temperature_c': float(lookup['temperature']),
                'mw': float(lookup['mw']),
                'z': float(lookup['z']),
                'k': float(lookup.get('k', 1.4)),
                'diameter_m': float(lookup.get('diameter', 1.0))
            }

        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def compute_missing_parameter(
        cls,
        available,
        mass_flow_kg_s
    ):

        have = set(available.keys())
        needed = {'Head', 'Efficiency', 'Power'} - have

        if len(needed) != 1:
            return None, None

        missing = needed.pop()

        if missing == 'Power':
            eff = available['Efficiency'] / 100.0
            power_kw = (
                mass_flow_kg_s *
                cls.G *
                available['Head'] /
                eff /
                1000.0
            )
            return 'Power', power_kw

        if missing == 'Head':
            eff = available['Efficiency'] / 100.0
            head_m = (
                available['Power'] *
                1000.0 *
                eff /
                (mass_flow_kg_s * cls.G)
            )
            return 'Head', head_m

        if missing == 'Efficiency':
            eff_pct = (
                (mass_flow_kg_s * cls.G * available['Head']) /
                (available['Power'] * 1000.0)
            ) * 100.0
            return 'Efficiency', eff_pct

        return None, None
