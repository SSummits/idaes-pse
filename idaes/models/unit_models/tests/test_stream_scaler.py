#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Tests for Stream Scaler unit model.

Author: Tanner Polley
"""

import pytest
import pandas
from numpy import number

from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    value,
    units as pyunits,
)

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.stream_scaler import StreamScaler, StreamScalerInitializer

from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)

from idaes.models.properties import iapws95
from idaes.models.properties.examples.saponification_thermo import (
    SaponificationParameterBlock,
)

from idaes.core.util.model_statistics import (
    number_variables,
    number_total_constraints,
    number_unused_variables,
    variables_set,
)
from idaes.core.util.testing import PhysicalParameterTestBlock, initialization_tester
from idaes.core.solvers import get_solver
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)
from idaes.core.util import DiagnosticsToolbox

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver("ipopt_v2")


# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_config():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = PhysicalParameterTestBlock()

    m.fs.unit = StreamScaler(property_package=m.fs.properties)

    # Check unit config arguments
    assert len(m.fs.unit.config) == 4

    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert m.fs.unit.config.property_package is m.fs.properties

    assert m.fs.unit.default_initializer is StreamScalerInitializer


class TestSaponification(object):
    @pytest.fixture(scope="class")
    def sapon(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = SaponificationParameterBlock()
        m.fs.unit = StreamScaler(property_package=m.fs.properties)
        m.fs.unit.multiplier.fix(1)

        m.fs.unit.inlet.flow_vol.fix(1.0e-03)
        m.fs.unit.inlet.conc_mol_comp[0, "H2O"].fix(55388.0)
        m.fs.unit.inlet.conc_mol_comp[0, "NaOH"].fix(100.0)
        m.fs.unit.inlet.conc_mol_comp[0, "EthylAcetate"].fix(100.0)
        m.fs.unit.inlet.conc_mol_comp[0, "SodiumAcetate"].fix(1e-8)
        m.fs.unit.inlet.conc_mol_comp[0, "Ethanol"].fix(1e-8)

        m.fs.unit.inlet.temperature.fix(303.15)
        m.fs.unit.inlet.pressure.fix(101325.0)
        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, sapon):

        assert hasattr(sapon.fs.unit, "inlet")
        assert len(sapon.fs.unit.inlet.vars) == 4
        assert hasattr(sapon.fs.unit.inlet, "flow_vol")
        assert hasattr(sapon.fs.unit.inlet, "conc_mol_comp")
        assert hasattr(sapon.fs.unit.inlet, "temperature")
        assert hasattr(sapon.fs.unit.inlet, "pressure")

        assert number_variables(sapon) == 9
        assert number_total_constraints(sapon) == 0
        assert number_unused_variables(sapon) == 9

    @pytest.mark.component
    def test_structural_issues(self, sapon):
        dt = DiagnosticsToolbox(sapon)
        dt.assert_no_structural_warnings()

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_performance_contents(self, sapon):
        perf_dict = sapon.fs.unit._get_performance_contents()

        assert perf_dict is None

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_stream_table_contents(self, sapon):
        stable = sapon.fs.unit._get_stream_table_contents()

        expected = pandas.DataFrame.from_dict(
            {
                "Units": {
                    "Volumetric Flowrate": getattr(
                        pyunits.pint_registry, "m**3/second"
                    ),
                    "Molar Concentration H2O": getattr(
                        pyunits.pint_registry, "mole/m**3"
                    ),
                    "Molar Concentration NaOH": getattr(
                        pyunits.pint_registry, "mole/m**3"
                    ),
                    "Molar Concentration EthylAcetate": getattr(
                        pyunits.pint_registry, "mole/m**3"
                    ),
                    "Molar Concentration SodiumAcetate": getattr(
                        pyunits.pint_registry, "mole/m**3"
                    ),
                    "Molar Concentration Ethanol": getattr(
                        pyunits.pint_registry, "mole/m**3"
                    ),
                    "Temperature": getattr(pyunits.pint_registry, "K"),
                    "Pressure": getattr(pyunits.pint_registry, "Pa"),
                },
                "Inlet": {
                    "Volumetric Flowrate": 1e-3,
                    "Molar Concentration H2O": 55388,
                    "Molar Concentration NaOH": 100.00,
                    "Molar Concentration EthylAcetate": 100.00,
                    "Molar Concentration SodiumAcetate": 0,
                    "Molar Concentration Ethanol": 0,
                    "Temperature": 303.15,
                    "Pressure": 1.0132e05,
                },
            }
        )

        pandas.testing.assert_frame_equal(stable, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, sapon):
        initialization_tester(sapon)

    # No solve or numerical tests, as StreamScaler block has nothing to solve


class TestBTX(object):
    @pytest.fixture(scope="class")
    def btx(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = BTXParameterBlock(valid_phase="Liq")
        m.fs.unit = StreamScaler(property_package=m.fs.properties)
        m.fs.unit.multiplier.fix(1)
        m.fs.unit.inlet.flow_mol[0].fix(5)  # mol/s
        m.fs.unit.inlet.temperature[0].fix(365)  # K
        m.fs.unit.inlet.pressure[0].fix(101325)  # Pa
        m.fs.unit.inlet.mole_frac_comp[0, "benzene"].fix(0.5)
        m.fs.unit.inlet.mole_frac_comp[0, "toluene"].fix(0.5)
        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, btx):

        assert hasattr(btx.fs.unit, "inlet")
        assert len(btx.fs.unit.inlet.vars) == 4
        assert hasattr(btx.fs.unit.inlet, "flow_mol")
        assert hasattr(btx.fs.unit.inlet, "mole_frac_comp")
        assert hasattr(btx.fs.unit.inlet, "temperature")
        assert hasattr(btx.fs.unit.inlet, "pressure")

        assert number_variables(btx) == 9
        assert number_total_constraints(btx) == 3
        assert number_unused_variables(btx) == 3

    @pytest.mark.component
    def test_structural_issues(self, btx):
        dt = DiagnosticsToolbox(btx)
        dt.assert_no_structural_warnings()

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_performance_contents(self, btx):
        perf_dict = btx.fs.unit._get_performance_contents()

        assert perf_dict is None

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_stream_table_contents(self, btx):
        stable = btx.fs.unit._get_stream_table_contents()

        expected = pandas.DataFrame.from_dict(
            {
                "Units": {
                    "flow_mol": getattr(pyunits.pint_registry, "mole/second"),
                    "mole_frac_comp benzene": getattr(
                        pyunits.pint_registry, "dimensionless"
                    ),
                    "mole_frac_comp toluene": getattr(
                        pyunits.pint_registry, "dimensionless"
                    ),
                    "temperature": getattr(pyunits.pint_registry, "kelvin"),
                    "pressure": getattr(pyunits.pint_registry, "Pa"),
                },
                "Inlet": {
                    "flow_mol": 5.0,
                    "mole_frac_comp benzene": 0.5,
                    "mole_frac_comp toluene": 0.5,
                    "temperature": 365,
                    "pressure": 101325.0,
                },
            }
        )

        pandas.testing.assert_frame_equal(stable, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, btx):
        initialization_tester(btx)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, btx):
        results = solver.solve(btx)

        # Check for optimal solution
        assert check_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, btx):
        assert pytest.approx(5, abs=1e-3) == value(btx.fs.unit.inlet.flow_mol[0])
        assert pytest.approx(0.5, abs=1e-3) == value(
            btx.fs.unit.inlet.mole_frac_comp[0, "benzene"]
        )
        assert pytest.approx(0.5, abs=1e-3) == value(
            btx.fs.unit.inlet.mole_frac_comp[0, "toluene"]
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_numerical_issues(self, btx):
        dt = DiagnosticsToolbox(btx)
        dt.assert_no_numerical_warnings()


# -----------------------------------------------------------------------------
@pytest.mark.iapws
@pytest.mark.skipif(not iapws95.iapws95_available(), reason="IAPWS not available")
class TestIAPWS(object):
    @pytest.fixture(scope="class")
    def iapws(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.properties = iapws95.Iapws95ParameterBlock()

        m.fs.unit = StreamScaler(property_package=m.fs.properties)

        m.fs.unit.multiplier.fix(1)
        m.fs.unit.inlet.flow_mol[0].fix(100)
        m.fs.unit.inlet.enth_mol[0].fix(5000)
        m.fs.unit.inlet.pressure[0].fix(101325)

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, iapws):
        assert len(iapws.fs.unit.inlet.vars) == 3
        assert hasattr(iapws.fs.unit.inlet, "flow_mol")
        assert hasattr(iapws.fs.unit.inlet, "enth_mol")
        assert hasattr(iapws.fs.unit.inlet, "pressure")

        assert number_variables(iapws) == 4
        assert number_total_constraints(iapws) == 0
        assert number_unused_variables(iapws) == 4

    @pytest.mark.component
    def test_structural_issues(self, iapws):
        dt = DiagnosticsToolbox(iapws)
        dt.assert_no_structural_warnings()

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_performance_contents(self, iapws):
        perf_dict = iapws.fs.unit._get_performance_contents()

        assert perf_dict is None

    @pytest.mark.ui
    @pytest.mark.unit
    def test_get_stream_table_contents(self, iapws):
        stable = iapws.fs.unit._get_stream_table_contents()

        expected = pandas.DataFrame.from_dict(
            {
                "Units": {
                    "Molar Flow": getattr(pyunits.pint_registry, "mole/second"),
                    "Mass Flow": getattr(pyunits.pint_registry, "kg/second"),
                    "T": getattr(pyunits.pint_registry, "K"),
                    "P": getattr(pyunits.pint_registry, "Pa"),
                    "Vapor Fraction": getattr(pyunits.pint_registry, "dimensionless"),
                    "Molar Enthalpy": getattr(pyunits.pint_registry, "J/mole"),
                },
                "Inlet": {
                    "Molar Flow": 100,
                    "Mass Flow": 1.8015,
                    "T": 339.43,
                    "P": 101325,
                    "Vapor Fraction": 0,
                    "Molar Enthalpy": 5000,
                },
            }
        )

        pandas.testing.assert_frame_equal(stable, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, iapws):
        initialization_tester(iapws)
