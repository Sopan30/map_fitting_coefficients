

class UnitConverter:

    # All target units: Flow -> m3/hr, Head -> m, Power -> kW, Efficiency -> %
    FLOW_TO_M3HR = {
        'cfm': 1.699010796, 'acfm': 1.699010796, 'icfm': 1.699010796,
        'ft3/min': 1.699010796, 'ft3min': 1.699010796, 'ft/min': 1.699010796, 'cf/min': 1.699010796,
        'cfh': 0.028316847, 'ft3/hr': 0.028316847, 'ft3/h': 0.028316847, 'ft3hr': 0.028316847,
        'ft/hr': 0.028316847, 'cf/hr': 0.028316847,
        'cfs': 101.9406, 'ft3/s': 101.9406, 'ft3s': 101.9406, 'ft/s': 101.9406, 'cf/s': 101.9406,
        'm3/hr': 1.0, 'm3/h': 1.0, 'm3hr': 1.0, 'm3h': 1.0,
        'm3/min': 60.0, 'm3min': 60.0,
        'm3/s': 3600.0, 'm3s': 3600.0,
        'l/min': 0.06, 'lpm': 0.06,
        'l/s': 3.6, 'lps': 3.6, 'l/hr': 0.001, 'lph': 0.001,
        'gpm': 0.227124707, 'usgpm': 0.227124707, 'galmin': 0.227124707,
        'igpm': 0.272765, 'ukgpm': 0.272765, 'impgpm': 0.272765,
        'gph': 0.003785412, 'usgph': 0.003785412,
        'bbl/day': 0.006624459, 'bpd': 0.006624459, 'bbl/d': 0.006624459,
        'mmscfd': 1179.874,
    }

    HEAD_TO_M = {
        'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048,
        'ftlbf/lbm': 0.3048, 'lbfft/lbm': 0.3048, 'ftlb/lb': 0.3048,'lbft/lb': 0.3048,
        'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
        'm': 1.0, 'meter': 1.0, 'metre': 1.0, 'meters': 1.0, 'metres': 1.0,
        'mm': 0.001,
        'kj/kg': 101.9716, 'j/kg': 0.1019716,
        'btu/lb': 237.2075,
    }

    POWER_TO_KW = {
        'hp': 0.745699872, 'bhp': 0.745699872, 'mechhp': 0.745699872, 'hp(i)': 0.745699872,
        'ps': 0.735499, 'cv': 0.735499, 'metrichp': 0.735499, 'hp(m)': 0.735499,
        'kw': 1.0, 'w': 0.001, 'mw': 1000.0,
        'btu/hr': 0.000293071, 'btu/h': 0.000293071, 'btuh': 0.000293071,
        'btu/s': 1.055056, 'btus': 1.055056,
        'ftlb/s': 0.001355818, 'ftlbf/s': 0.001355818,
        'kcal/hr': 0.001163, 'kcal/h': 0.001163,
    }

    EFF_TO_PCT = {
        '%': 1.0, 'pct': 1.0, 'percent': 1.0, 'percentage': 1.0,
        'fraction': 100.0, 'decimal': 100.0, 'ratio': 100.0, 'frac': 100.0,
    }

    DIAMETER_TO_M = {
        'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
        'mm': 0.001, 'milimeter': 0.001, 'milimeters': 0.001,
        'cm': 0.01, 'centimeter': 0.01,
        'm': 1.0, 'meter': 1.0, 'meters': 1.0
    }

    P_ATM_KG_CM2 = 1.033227

    @staticmethod
    def normalize_unit(u):
        s = str(u).strip().lower()
        s = s.replace('³', '3').replace('²', '2')
        s = s.replace('^3', '3').replace('^2', '2')
        s = s.replace(' ', '').replace('.', '')
        s = s.replace('-', '').replace('_', '')
        s = s.replace('cu', '')
        return s

    @classmethod
    def convert_unit(cls, value, unit_str, table):

        key = cls.normalize_unit(unit_str)

        if key in table:
            return value * table[key], True

        return value, False

    @staticmethod
    def convert_temperature_to_c(val, unit_str):

        u = str(unit_str).strip().lower()

        u = u.replace('°', '')
        u = u.replace('degree', '')
        u = u.replace('deg', '')
        u = u.replace(' ', '')
        u = u.replace('.', '')
        u = u.replace('-', '')
        u = u.replace('_', '')

        if u in ['f', 'fahrenheit']:
            return (val - 32) * 5.0 / 9.0, True

        if u in ['k', 'kelvin']:
            return val - 273.15, True

        if u in ['r', 'rankine']:
            return (val - 491.67) * 5.0 / 9.0, True

        if u in ['c', 'celsius', 'centigrade']:
            return val, True

        return val, False

    @classmethod
    def convert_pressure_to_kg_cm2a(
        cls,
        val,
        unit_str
    ):

        u = cls.normalize_unit(unit_str)

        multipliers = {
            'psi': 0.070306958,
            'psia': 0.070306958,
            'psig': 0.070306958,
            'bar': 1.01971621,
            'bara': 1.01971621,
            'barg': 1.01971621,
            'kpa': 0.010197162,
            'kpaa': 0.010197162,
            'kpag': 0.010197162,
            'mpa': 10.1971621,
            'mpaa': 10.1971621,
            'mpag': 10.1971621,
            'kg/cm2': 1.0,
            'kg/cm2a': 1.0,
            'kg/cm2g': 1.0,
            'kgf/cm2': 1.0,
            'kgf/cm2a': 1.0,
            'kgf/cm2g': 1.0,
            'atm': 1.033227,
            'atmg': 1.033227,
            'pa': 0.00001019716,
            'pag': 0.00001019716
        }

        if u not in multipliers:
            return val, False

        kg_cm2_val = val * multipliers[u]

        if u.endswith('g'):
            return kg_cm2_val + cls.P_ATM_KG_CM2, True

        return kg_cm2_val, True
    
    @staticmethod
    def kg_cm2a_to_pa(kg_cm2a):
        return kg_cm2a * 98066.5

    @staticmethod
    def c_to_k(deg_c):
        return deg_c + 273.15