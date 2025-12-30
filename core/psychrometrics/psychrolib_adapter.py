import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)

def moist_air_enthalpy(T: float, RH: float, p: float) -> float:
    """
    Specific enthalpy of moist air [J/kg dry air].
    """
    h = psychrolib.GetMoistAirEnthalpy(T - 273.15, RH, p)
    return h * 1000.0  # kJ/kg -> J/kg
