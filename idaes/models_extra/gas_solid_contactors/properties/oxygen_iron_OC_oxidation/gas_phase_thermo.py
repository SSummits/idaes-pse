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
This package provides the necessary constraints for gas phase properties for
the oxidation of a chemical looping oxygen carrier.
Components - Oxygen (O2), Nitrogen (N2), Carbon Dioxide (CO2), Water (H2O)

Equations written in this model were derived from:
(1) B.E. Poling, J.M. Prausnitz, J.P. O'connell, The Properties of Gases and
Liquids, Mcgraw-Hill, New York, 2001.
(2) National Institute of Standards and Technology, NIST Chemistry WebBook,
https://webbook.nist.gov/chemistry/ (accessed March 10, 2018).

"""
# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

# Import Pyomo libraries
from pyomo.environ import (
    Constraint,
    Expression,
    Param,
    Reals,
    value,
    log,
    Var,
    units as pyunits,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    Component,
    VaporPhase,
)
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.util.misc import add_object_reference
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_unfixed_variables_in_activated_equalities,
)
from idaes.core.util.constants import Constants
import idaes.logger as idaeslog
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver

# Some more information about this module
__author__ = "Chinedu Okoli"


# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("GasPhaseParameterBlock")
class PhysicalParameterData(PhysicalParameterBlock):
    """
    Property Parameter Block Class
    Contains parameters and indexing sets associated with properties for
    oxidation of oxygen carrier with oxygen.
    """

    def build(self):
        """
        Callable method for Block construction.
        """
        super(PhysicalParameterData, self).build()

        self._state_block_class = GasPhaseStateBlock

        # Create Phase object
        self.Vap = VaporPhase()

        # Create Component objects
        self.N2 = Component()
        self.O2 = Component()
        self.CO2 = Component()
        self.H2O = Component()

        # Mol. weights of gas - units = kg/mol. ref: NIST webbook
        mw_comp_dict = {"O2": 0.032, "N2": 0.028, "CO2": 0.044, "H2O": 0.018}
        self.mw_comp = Param(
            self.component_list,
            mutable=False,
            initialize=mw_comp_dict,
            doc="Molecular weights of gas components [kg/mol]",
            units=pyunits.kg / pyunits.mol,
        )

        # Std. heat of formation of comp. - units = kJ/(mol comp) - ref: NIST
        enth_mol_form_comp_dict = {
            "O2": 0,
            "N2": 0,
            "CO2": -393.5224e3,
            "H2O": -241.8264e3,
        }
        self.enth_mol_form_comp = Param(
            self.component_list,
            mutable=False,
            initialize=enth_mol_form_comp_dict,
            doc="Component molar heats of formation [J/mol]",
            units=pyunits.J / pyunits.mol,
        )

        # Ideal gas spec. heat capacity parameters (Shomate) of
        # components - ref: NIST webbook. Shomate equations from NIST.
        # Parameters A-E are used for cp calcs while A-H are used for enthalpy
        # calc.
        #
        # Note that the temperature ranges over which these parameters are
        # valid, per NIST webbook, are different depending on species. In
        # addition they may not contain the entire operating region of a
        # unit model. Really enthalpy should be a piecewise continuous
        # function of temperature that uses parameters only in the range
        # they are valid, but we have not implemented this.
        #
        # 1e3*cp_comp = A + B*T + C*T^2 + D*T^3 + E/(T^2)
        # where T = Temperature (K)/1000, and cp_comp = (kJ/mol.K)
        # H_comp = H - H(298.15) = A*T + B*T^2/2 + C*T^3/3 +
        # D*T^4/4 - E/T + F - H where T = Temp (K)/1000 and H_comp = (kJ/mol)
        cp_param_dict = {
            ("O2", 1): 30.03235,
            ("O2", 2): 8.772972,
            ("O2", 3): -3.988133,
            ("O2", 4): 0.788313,
            ("O2", 5): -0.741599,
            ("O2", 6): -11.32468,
            ("O2", 7): 236.1663,
            ("O2", 8): 0.0000,
            ("N2", 1): 19.50583,
            ("N2", 2): 19.88705,
            ("N2", 3): -8.598535,
            ("N2", 4): 1.369784,
            ("N2", 5): 0.527601,
            ("N2", 6): -4.935202,
            ("N2", 7): 212.3900,
            ("N2", 8): 0.0000,
            ("CO2", 1): 24.9973500,
            ("CO2", 2): 55.1869600,
            ("CO2", 3): -33.6913700,
            ("CO2", 4): 7.9483870,
            ("CO2", 5): -0.1366380,
            ("CO2", 6): -403.6075000,
            ("CO2", 7): 228.2431000,
            ("CO2", 8): -393.5224000,
            ("H2O", 1): 30.0920000,
            ("H2O", 2): 6.8325140,
            ("H2O", 3): 6.7934350,
            ("H2O", 4): -2.5344800,
            ("H2O", 5): 0.0821390,
            ("H2O", 6): -250.8810000,
            ("H2O", 7): 223.3967000,
            ("H2O", 8): -241.8264000,
        }
        self.cp_param_1 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 1},
            doc="Shomate equation heat capacity coeff 1",
            units=pyunits.J / pyunits.mol / pyunits.K,
        )
        self.cp_param_2 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 2},
            doc="Shomate equation heat capacity coeff 2",
            units=pyunits.J / pyunits.mol / pyunits.K / pyunits.kK,
        )
        self.cp_param_3 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 3},
            doc="Shomate equation heat capacity coeff 3",
            units=pyunits.J / pyunits.mol / pyunits.K / pyunits.kK**2,
        )
        self.cp_param_4 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 4},
            doc="Shomate equation heat capacity coeff 4",
            units=pyunits.J / pyunits.mol / pyunits.K / pyunits.kK**3,
        )
        self.cp_param_5 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 5},
            doc="Shomate equation heat capacity coeff 5",
            units=pyunits.J / pyunits.mol / pyunits.K * pyunits.kK**2,
        )
        self.cp_param_6 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 6},
            doc="Shomate equation heat capacity coeff 6",
            units=pyunits.kJ / pyunits.mol,
        )
        self.cp_param_7 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 7},
            doc="Shomate equation heat capacity coeff 7",
            units=pyunits.J / pyunits.mol / pyunits.K,
        )
        self.cp_param_8 = Param(
            self.component_list,
            mutable=False,
            initialize={k: v for (k, j), v in cp_param_dict.items() if j == 8},
            doc="Shomate equation heat capacity coeff 8",
            units=pyunits.kJ / pyunits.mol,
        )

        # Viscosity constants:
        # Reference: Perry and Green Handbook; McGraw Hill, 2008
        visc_d_param_dict = {
            ("O2", 1): 1.101e-6,
            ("O2", 2): 0.5634,
            ("O2", 3): 96.3,
            ("O2", 4): 0,
            ("N2", 1): 6.5592e-7,
            ("N2", 2): 0.6081,
            ("N2", 3): 54.714,
            ("N2", 4): 0,
            ("CO2", 1): 2.148e-6,
            ("CO2", 2): 0.46,
            ("CO2", 3): 290,
            ("CO2", 4): 0,
            ("H2O", 1): 1.7096e-8,
            ("H2O", 2): 1.1146,
            ("H2O", 3): 0,
            ("H2O", 4): 0,
        }
        self.visc_d_param_1 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in visc_d_param_dict.items() if j == 1},
            doc="Dynamic viscosity constants",
            units=pyunits.kg / pyunits.m / pyunits.s,
        )
        # The units of parameter 1 are dependent upon the value of parameter 2:
        # [visc_d_param_1] = kg/m-s * K^(-(value([visc_d_param_2)))
        # this is accounted for in the equation for visc_d_comp
        self.visc_d_param_2 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in visc_d_param_dict.items() if j == 2},
            doc="Dynamic viscosity constants",
            units=pyunits.dimensionless,
        )
        self.visc_d_param_3 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in visc_d_param_dict.items() if j == 3},
            doc="Dynamic viscosity constants",
            units=pyunits.K,
        )
        self.visc_d_param_4 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in visc_d_param_dict.items() if j == 4},
            doc="Dynamic viscosity constants",
            units=pyunits.K**2,
        )

        # Thermal conductivity constants:
        # Reference: Perry and Green Handbook; McGraw Hill, 2008
        therm_cond_param_dict = {
            ("N2", 1): 3.3143e-4,
            ("N2", 2): 0.7722,
            ("N2", 3): 16.323,
            ("N2", 4): 0,
            ("O2", 1): 4.4994e-4,
            ("O2", 2): 0.7456,
            ("O2", 3): 56.699,
            ("O2", 4): 0,
            ("CO2", 1): 3.69,
            ("CO2", 2): -0.3838,
            ("CO2", 3): 964,
            ("CO2", 4): 1.86e6,
            ("H2O", 1): 6.204e-6,
            ("H2O", 2): 1.3973,
            ("H2O", 3): 0,
            ("H2O", 4): 0,
        }
        self.therm_cond_param_1 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in therm_cond_param_dict.items() if j == 1},
            doc="Dynamic viscosity constants",
            units=pyunits.J / pyunits.m / pyunits.s,
        )
        # The units of parameter 1 are dependent upon the value of parameter 2:
        # [therm_cond_param_1] = J/m-s * K^(-(1 + value([therm_cond_param_2)))
        # this is accounted for in the equation for therm_cond_comp
        self.therm_cond_param_2 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in therm_cond_param_dict.items() if j == 2},
            doc="Dynamic viscosity constants",
            units=pyunits.dimensionless,
        )
        self.therm_cond_param_3 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in therm_cond_param_dict.items() if j == 3},
            doc="Dynamic viscosity constants",
            units=pyunits.K,
        )
        self.therm_cond_param_4 = Param(
            self.component_list,
            mutable=True,
            initialize={k: v for (k, j), v in therm_cond_param_dict.items() if j == 4},
            doc="Dynamic viscosity constants",
            units=pyunits.K**2,
        )

        # Component diffusion volumes:
        # Ref: (1) Prop gas & liquids (2) Fuller et al. IECR, 58(5), 19, 1966
        # NOTE: These parameters are dimensionless.
        diff_vol_param_dict = {"O2": 16.6, "N2": 17.9, "CO2": 26.9, "H2O": 13.1}
        self.diff_vol_param = Param(
            self.component_list,
            mutable=True,
            initialize=diff_vol_param_dict,
            doc="Component diffusion volumes",
        )

        # Set default scaling for state variables
        self.set_default_scaling("flow_mol", 1e-3)
        self.set_default_scaling("pressure", 1e-5)
        self.set_default_scaling("temperature", 1e-2)
        for comp in self.component_list:
            self.set_default_scaling("mole_frac_comp", 1e1, index=comp)

        # Set default scaling for thermophysical and transport properties
        self.set_default_scaling("enth_mol", 1e-6)
        self.set_default_scaling("enth_mol_comp", 1e-6)
        self.set_default_scaling("cp_mol", 1e-6)
        self.set_default_scaling("cp_mol_comp", 1e-6)
        self.set_default_scaling("cp_mass", 1e-6)
        self.set_default_scaling("entr_mol", 1e-2)
        self.set_default_scaling("entr_mol_phase", 1e-2)
        self.set_default_scaling("dens_mol_comp", 1)
        self.set_default_scaling("dens_mass", 1e2)
        self.set_default_scaling("visc_d_comp", 1e4)
        self.set_default_scaling("diffus_comp", 1e5)
        self.set_default_scaling("therm_cond_comp", 1e2)
        self.set_default_scaling("visc_d", 1e5)
        self.set_default_scaling("therm_cond", 1e0)
        self.set_default_scaling("mw", 1e2)

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "flow_mol": {"method": None},
                "pressure": {"method": None},
                "temperature": {"method": None},
                "mole_frac_comp": {"method": None},
                "mw": {"method": "_mw"},
                "cp_mol": {"method": "_cp_mol"},
                "cp_mol_comp": {"method": "_cp_mol_comp"},
                "cp_mass": {"method": "_cp_mass"},
                "dens_mol": {"method": "_dens_mol"},
                "dens_mol_comp": {"method": "_dens_mol_comp"},
                "dens_mass": {"method": "_dens_mass"},
                "enth_mol": {"method": "_enth_mol"},
                "enth_mol_comp": {"method": "_enth_mol_comp"},
                "entr_mol": {"method": "_entr_mol"},
                "visc_d": {"method": "_visc_d"},
                "therm_cond": {"method": "_therm_cond"},
                "diffus_comp": {"method": "_diffus_comp"},
            }
        )

        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )
        # def add_default_units(self, u): u (dict): Key=property, Value=units
        # def add_properties(self, p): p (dict): Key=property, Value=PropertyMetadata or equiv. dict
        # def get_derived_units(self, units):
        # obj.get_derived_units("power") = pyunits.kJ * pyunits.s ** -1


class _GasPhaseStateBlock(StateBlock):
    """
    This Class contains methods which should be applied to State Blocks as a
    whole, rather than individual elements of indexed State Blocks.
    """

    def initialize(
        blk,
        state_args=None,
        hold_state=False,
        state_vars_fixed=False,
        outlvl=idaeslog.NOTSET,
        solver="ipopt",
        optarg=None,
    ):
        """
        Initialization routine for property package.
        Keyword Arguments:
            state_args : Dictionary with initial guesses for the state vars
                         chosen. Note that if this method is triggered
                         through the control volume, and if initial guesses
                         were not provided at the unit model level, the
                         control volume passes the inlet values as initial
                         guess.
                         Keys for the state_args dictionary are:
                         flow_mol, temperature, pressure and mole_frac_comp
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = "ipopt")
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states varaibles are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 relase_state method
        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="properties")

        init_log.info_high("Starting initialization")

        # Deactivate the constraints specific for non-inlet blocks i.e.
        # when defined state is False
        for k in blk.values():
            if k.config.defined_state is False:
                k.sum_component_eqn.deactivate()

        # Fix state variables if not already fixed
        if state_vars_fixed is False:
            flags = fix_state_vars(blk, state_args)
        else:
            # Check when the state vars are fixed already result in dof 0
            for k in blk.values():
                if degrees_of_freedom(k) != 0:
                    raise Exception(
                        "State vars fixed but degrees of freedom "
                        "for state block is not zero during "
                        "initialization."
                    )

        # ---------------------------------------------------------------------
        # Initialise values
        for k in blk.values():

            if hasattr(k, "mw_eqn"):
                calculate_variable_from_constraint(k.mw, k.mw_eqn)

            if hasattr(k, "ideal_gas"):
                calculate_variable_from_constraint(k.dens_mol, k.ideal_gas)

            if hasattr(k, "dens_mass_basis"):
                calculate_variable_from_constraint(k.dens_mass, k.dens_mass_basis)

            if hasattr(k, "mixture_heat_capacity_eqn"):
                calculate_variable_from_constraint(
                    k.cp_mol, k.mixture_heat_capacity_eqn
                )

            if hasattr(k, "cp_mass_basis"):
                calculate_variable_from_constraint(k.cp_mass, k.cp_mass_basis)

            if hasattr(k, "visc_d_constraint"):
                calculate_variable_from_constraint(k.visc_d, k.visc_d_constraint)

            if hasattr(k, "therm_cond_constraint"):
                calculate_variable_from_constraint(
                    k.therm_cond, k.therm_cond_constraint
                )

            if hasattr(k, "mixture_enthalpy_eqn"):
                calculate_variable_from_constraint(k.enth_mol, k.mixture_enthalpy_eqn)

            for j in k.params.component_list:

                if hasattr(k, "comp_conc_eqn"):
                    calculate_variable_from_constraint(
                        k.dens_mol_comp[j], k.comp_conc_eqn[j]
                    )

                if hasattr(k, "diffus_comp_constraint"):
                    calculate_variable_from_constraint(
                        k.diffus_comp[j], k.diffus_comp_constraint[j]
                    )

                if hasattr(k, "cp_shomate_eqn"):
                    calculate_variable_from_constraint(
                        k.cp_mol_comp[j], k.cp_shomate_eqn[j]
                    )

                if hasattr(k, "enthalpy_shomate_eqn"):
                    calculate_variable_from_constraint(
                        k.enth_mol_comp[j], k.enthalpy_shomate_eqn[j]
                    )

        # Solve property block if non-empty
        free_vars = 0
        for k in blk.values():
            free_vars += number_unfixed_variables_in_activated_equalities(k)

        if free_vars > 0:
            # Create solver
            opt = get_solver(solver, optarg)
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solve_indexed_blocks(opt, [blk], tee=slc.tee)
        else:
            res = ""
        init_log.info_high(
            "Initialization complete {}.".format(idaeslog.condition(res))
        )

        # ---------------------------------------------------------------------
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                blk.release_state(flags)

    def release_state(blk, flags, outlvl=0):
        """
        Method to relase state variables fixed during initialization.
        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of logging
        """
        if flags is None:
            return

        # Unfix state variables
        revert_state_vars(blk, flags)

        # Activate state variable related constraints
        for k in blk.values():
            if k.config.defined_state is False:
                k.sum_component_eqn.activate()

        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        init_log.info_high("States released.")


@declare_process_block_class("GasPhaseStateBlock", block_class=_GasPhaseStateBlock)
class GasPhaseStateBlockData(StateBlockData):
    """
    Property package for gas phase properties of methane combustion in CLC FR
    """

    def build(self):
        """
        Callable method for Block construction
        """
        super(GasPhaseStateBlockData, self).build()

        units_meta = self.params.get_metadata().derived_units

        # Object reference for molecular weight if needed by CV1D
        # Molecular weights
        add_object_reference(self, "mw_comp", self.config.parameters.mw_comp)

        # List the necessary state variable objects.
        self.flow_mol = Var(
            initialize=1.0,
            domain=Reals,
            doc="Component molar flowrate",
            units=units_meta.FLOW_MOLE,
        )
        self.mole_frac_comp = Var(
            self.params.component_list,
            domain=Reals,
            initialize=1 / len(self.params.component_list),
            doc="State component mole fractions",
            units=pyunits.dimensionless,
        )
        self.pressure = Var(
            initialize=1.01325e5,
            domain=Reals,
            doc="State pressure",
            units=units_meta.PRESSURE,
        )
        self.temperature = Var(
            initialize=298.15,
            domain=Reals,
            doc="State temperature",
            units=units_meta.TEMPERATURE,
        )

        # Create standard constraints
        # Sum mole fractions if not inlet block
        if self.config.defined_state is False:

            def sum_component_eqn(b):
                return 1 == sum(b.mole_frac_comp[j] for j in b.params.component_list)

            self.sum_component_eqn = Constraint(rule=sum_component_eqn)

    def _mw(self):
        # Molecular weight of gas mixture
        units_meta = self.params.get_metadata().derived_units
        self.mw = Var(
            domain=Reals,
            initialize=1.0,
            doc="Molecular weight of gas mixture",
            units=units_meta.MOLECULAR_WEIGHT,
        )

        def mw_eqn(b):
            return b.mw == sum(
                b.mole_frac_comp[j] * b.params.mw_comp[j]
                for j in b.params.component_list
            )

        try:
            # Try to build constraint
            self.mw_eqn = Constraint(rule=mw_eqn)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.mw)
            self.del_component(self.mw_eqn)
            raise

    def _dens_mol(self):
        # Molar density
        units_meta = self.params.get_metadata().derived_units
        self.dens_mol = Var(
            domain=Reals,
            initialize=1.0,
            doc="Molar density or concentration",
            units=units_meta.DENSITY_MOLE,
        )

        def ideal_gas(b):
            pressure = pyunits.convert(b.pressure, to_units=pyunits.Pa)
            temperature = pyunits.convert(b.temperature, to_units=pyunits.K)
            dens_mol = pyunits.convert(b.dens_mol, to_units=pyunits.mol / pyunits.m**3)
            gas_constant = pyunits.convert(
                Constants.gas_constant, to_units=pyunits.J / pyunits.mol / pyunits.K
            )
            return dens_mol * temperature * gas_constant == pressure

        try:
            # Try to build constraint
            self.ideal_gas = Constraint(rule=ideal_gas)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.dens_mol)
            self.del_component(self.ideal_gas)
            raise

    def _dens_mol_comp(self):
        # Mixture heat capacities
        units_meta = self.params.get_metadata().derived_units
        self.dens_mol_comp = Var(
            self.params.component_list,
            domain=Reals,
            initialize=1.0,
            doc="Component molar concentration",
            units=units_meta.DENSITY_MOLE,
        )

        def comp_conc_eqn(b, j):
            return b.dens_mol_comp[j] == b.dens_mol * b.mole_frac_comp[j]

        try:
            # Try to build constraint
            self.comp_conc_eqn = Constraint(
                self.params.component_list, rule=comp_conc_eqn
            )
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.dens_mol_comp)
            self.del_component(self.comp_conc_eqn)
            raise

    def _dens_mass(self):
        # Mass density
        units_meta = self.params.get_metadata().derived_units
        self.dens_mass = Var(
            domain=Reals,
            initialize=1.0,
            doc="Mass density",
            units=units_meta.DENSITY_MASS,
        )

        def dens_mass_basis(b):
            return b.dens_mass == b.mw * b.dens_mol

        try:
            # Try to build constraint
            self.dens_mass_basis = Constraint(rule=dens_mass_basis)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.dens_mass)
            self.del_component(self.dens_mass_basis)
            raise

    def _visc_d(self):
        # Mixture dynamic viscosity
        units_meta = self.params.get_metadata().derived_units
        self.visc_d = Var(
            domain=Reals,
            initialize=1e-5,
            doc="Mixture dynamic viscosity",
            units=units_meta.DYNAMIC_VISCOSITY,
        )

        def visc_d_comp(i):
            visc_d_param_1 = self.params.visc_d_param_1[i] * pyunits.K ** (
                -self.params.visc_d_param_2[i]
            )
            return (
                visc_d_param_1
                * (self.temperature ** self.params.visc_d_param_2[i])
                / (
                    (1 + (self.params.visc_d_param_3[i] / self.temperature))
                    + (self.params.visc_d_param_4[i] / (self.temperature**2))
                )
            )

        def visc_d_constraint(b):
            return b.visc_d == sum(
                b.mole_frac_comp[i]
                * visc_d_comp(i)
                / (
                    sum(
                        b.mole_frac_comp[j]
                        * (b.params.mw_comp[j] / b.params.mw_comp[i]) ** 0.5
                        for j in b.params.component_list
                    )
                )
                for i in b.params.component_list
            )

        try:
            # Try to build constraint
            self.visc_d_constraint = Constraint(rule=visc_d_constraint)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.visc_d)
            self.del_component(self.visc_d_constraint)
            raise

    def _diffus_comp(self):
        # Component diffusion in a gas mixture - units of cm2/s to help scaling
        self.diffus_comp = Var(
            self.params.component_list,
            domain=Reals,
            initialize=1.0,
            doc="Component diffusion in a gas mixture" "[cm2/s]",
            units=pyunits.cm**2 / pyunits.s,
        )

        def D_bin(i, j):

            empirical_coef_units = (
                pyunits.cm**2
                / pyunits.s**3
                / pyunits.K**1.75
                * pyunits.kg**1.5
                / pyunits.kmol**0.5
                / pyunits.m
            )

            return (
                1.43e-3
                * empirical_coef_units
                * (self.temperature**1.75)
                * (
                    (
                        pyunits.convert(
                            self.params.mw_comp[i], to_units=pyunits.kg / pyunits.kmol
                        )
                        + pyunits.convert(
                            self.params.mw_comp[j], to_units=pyunits.kg / pyunits.kmol
                        )
                    )
                    / (
                        2
                        * (
                            pyunits.convert(
                                self.params.mw_comp[i],
                                to_units=pyunits.kg / pyunits.kmol,
                            )
                        )
                        * (
                            pyunits.convert(
                                self.params.mw_comp[j],
                                to_units=pyunits.kg / pyunits.kmol,
                            )
                        )
                    )
                )
                ** 0.5
            ) / (
                (self.pressure)
                * (
                    (self.params.diff_vol_param[i] ** (1 / 3))
                    + (self.params.diff_vol_param[j] ** (1 / 3))
                )
                ** 2
            )

        def diffus_comp_constraint(b, i):
            return b.diffus_comp[i] * sum(
                b.mole_frac_comp[j] / D_bin(i, j)
                for j in b.params.component_list
                if i != j
            ) == (1 - b.mole_frac_comp[i])

        try:
            # Try to build constraint
            self.diffus_comp_constraint = Constraint(
                self.params.component_list, rule=diffus_comp_constraint
            )
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.diffus_comp)
            self.del_component(self.diffus_comp_constraint)
            raise

    def _therm_cond(self):
        # Thermal conductivity of gas
        units_meta = self.params.get_metadata().derived_units
        units_therm_cond = units_meta.THERMAL_CONDUCTIVITY
        self.therm_cond = Var(
            domain=Reals,
            initialize=1e-5,
            doc="Thermal conductivity of gas",
            units=units_therm_cond,
        )

        def therm_cond_comp(i):
            therm_cond_param_1 = self.params.therm_cond_param_1[i] * pyunits.K ** (
                -(1 + self.params.therm_cond_param_2[i])
            )
            return (
                therm_cond_param_1
                * (self.temperature ** self.params.therm_cond_param_2[i])
                / (
                    (1 + (self.params.therm_cond_param_3[i] / self.temperature))
                    + (self.params.therm_cond_param_4[i] / (self.temperature**2))
                )
            )

        def A_bin(i, j):
            return (
                1
                + ((therm_cond_comp(j) / therm_cond_comp(i)) ** 0.5)
                * ((self.params.mw_comp[j] / self.params.mw_comp[i]) ** 0.25)
            ) ** 2 / (
                8 * (1 + (self.params.mw_comp[j] / self.params.mw_comp[i]))
            ) ** 0.5

        def therm_cond_constraint(b):
            return b.therm_cond == pyunits.convert(
                sum(
                    b.mole_frac_comp[i]
                    * therm_cond_comp(i)
                    / (
                        sum(
                            b.mole_frac_comp[j] * A_bin(i, j) ** 0.5
                            for j in b.params.component_list
                        )
                    )
                    for i in b.params.component_list
                ),
                to_units=units_therm_cond,
            )

        try:
            # Try to build constraint
            self.therm_cond_constraint = Constraint(rule=therm_cond_constraint)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.therm_cond)
            self.del_component(self.therm_cond_constraint)
            raise

    def _cp_mol_comp(self):
        # Pure component vapour heat capacities
        units_meta = self.params.get_metadata().derived_units
        units_cp_mol = units_meta.HEAT_CAPACITY_MOLE
        self.cp_mol_comp = Var(
            self.params.component_list,
            domain=Reals,
            initialize=1.0,
            doc="Pure component vapour heat capacities",
            units=units_cp_mol,
        )

        def pure_component_cp_mol(b, j):
            t = pyunits.convert(b.temperature, to_units=pyunits.kK)
            return b.cp_mol_comp[j] == pyunits.convert(
                (
                    b.params.cp_param_1[j]
                    + b.params.cp_param_2[j] * t
                    + b.params.cp_param_3[j] * t**2
                    + b.params.cp_param_4[j] * t**3
                    + b.params.cp_param_5[j] / (t**2)
                ),
                to_units=units_cp_mol,
            )

        try:
            # Try to build constraint
            self.cp_shomate_eqn = Constraint(
                self.params.component_list, rule=pure_component_cp_mol
            )
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.cp_mol_comp)
            self.del_component(self.cp_shomate_eqn)
            raise

    def _cp_mol(self):
        # Mixture heat capacities
        units_meta = self.params.get_metadata().derived_units
        units_cp_mol = units_meta.HEAT_CAPACITY_MOLE
        self.cp_mol = Var(
            domain=Reals,
            initialize=1.0,
            doc="Mixture heat capacity",
            units=units_cp_mol,
        )

        def cp_mol(b):
            return b.cp_mol == sum(
                b.cp_mol_comp[j] * b.mole_frac_comp[j] for j in b.params.component_list
            )

        try:
            # Try to build constraint
            self.mixture_heat_capacity_eqn = Constraint(rule=cp_mol)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.cp_mol)
            self.del_component(self.mixture_heat_capacity_eqn)
            raise

    def _cp_mass(self):
        # Mixture heat capacities
        units_meta = self.params.get_metadata().derived_units
        units_cp_mass = units_meta.HEAT_CAPACITY_MASS
        self.cp_mass = Var(
            domain=Reals,
            initialize=1.0,
            doc="Mixture heat capacity, mass-basis",
            units=units_cp_mass,
        )

        def cp_mass(b):
            return b.cp_mass * b.mw == b.cp_mol

        try:
            # Try to build constraint
            self.cp_mass_basis = Constraint(rule=cp_mass)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.cp_mass)
            self.del_component(self.cp_mass_basis)
            raise

    def _enth_mol_comp(self):
        # Pure component vapour enthalpies
        units_meta = self.params.get_metadata().derived_units
        units_enth_mol = units_meta.ENERGY_MOLE
        self.enth_mol_comp = Var(
            self.params.component_list,
            domain=Reals,
            initialize=1.0,
            doc="Pure component enthalpies",
            units=units_enth_mol,
        )

        def pure_comp_enthalpy(b, j):
            t = pyunits.convert(b.temperature, to_units=pyunits.kK)
            return b.enth_mol_comp[j] == pyunits.convert(
                # parameters 1-5 are defined in J
                b.params.cp_param_1[j] * t
                + b.params.cp_param_2[j] * (t**2) / 2
                + b.params.cp_param_3[j] * (t**3) / 3
                + b.params.cp_param_4[j] * (t**4) / 4
                - b.params.cp_param_5[j] / (t),
                to_units=units_enth_mol,
            ) + pyunits.convert(
                # parameters 6 and 8 are defined in kJ, and must be added
                # after converting to the enthalpy units set
                b.params.cp_param_6[j] - b.params.cp_param_8[j],
                to_units=units_enth_mol,
            )

        try:
            # Try to build constraint
            self.enthalpy_shomate_eqn = Constraint(
                self.params.component_list, rule=pure_comp_enthalpy
            )
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.enth_mol_comp)
            self.del_component(self.enthalpy_shomate_eqn)
            raise

    def _enth_mol(self):
        # Mixture molar enthalpy
        units_meta = self.params.get_metadata().derived_units
        units_enth_mol = units_meta.ENERGY_MOLE
        self.enth_mol = Var(
            domain=Reals,
            initialize=1.0,
            doc="Mixture specific enthalpy",
            units=units_enth_mol,
        )
        try:
            # Try to build constraint
            self.mixture_enthalpy_eqn = Constraint(
                expr=(
                    self.enth_mol
                    == sum(
                        self.mole_frac_comp[j] * self.enth_mol_comp[j]
                        for j in self.params.component_list
                    )
                )
            )
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.enth_mol)
            self.del_component(self.mixture_enthalpy_eqn)
            raise

    def _entr_mol(self):
        units_meta = self.params.get_metadata().derived_units
        units_entr_mol = units_meta.ENTROPY_MOLE
        self.entr_mol = Var(
            doc="Specific Entropy",
            initialize=1.0,
            units=units_entr_mol,
        )
        # Specific Entropy

        def rule_entr_phase(b, p):
            # This property module only has one phase
            return self.entr_mol

        self.entr_mol_phase = Expression(self.params.phase_list, rule=rule_entr_phase)

        def entropy_correlation(b):
            t = pyunits.convert(self.temperature, to_units=pyunits.kK)
            x = b.mole_frac_comp
            p = pyunits.convert(b.pressure, to_units=pyunits.Pa)
            r_gas = pyunits.convert(
                Constants.gas_constant, to_units=pyunits.J / pyunits.mol / pyunits.K
            )
            return (
                self.entr_mol + r_gas * log(p / 1e5 / pyunits.Pa)
            ) * b.flow_mol == sum(
                b.flow_mol
                * b.mole_frac_comp[j]
                * (
                    self.params.cp_param_1[j] * log(t / pyunits.kK)
                    + self.params.cp_param_2[j] * t
                    + self.params.cp_param_3[j] * t**2 / 2
                    + self.params.cp_param_4[j] * t**3 / 3
                    - self.params.cp_param_5[j] / t**2 / 2
                    + self.params.cp_param_7[j]
                    + r_gas * log(x[j])
                )
                for j in self.params.component_list
            )

        try:
            self.entropy_correlation = Constraint(rule=entropy_correlation)
        except AttributeError:
            self.del_component(self.entr_mol_phase)
            self.del_component(self.entropy_correlation)

    def get_material_flow_terms(self, p, j):
        if not self.is_property_constructed("material_flow_terms"):
            try:

                def rule_material_flow_terms(b, j):
                    return b.flow_mol * b.mole_frac_comp[j]

                self.material_flow_terms = Expression(
                    self.params.component_list, rule=rule_material_flow_terms
                )
            except AttributeError:
                self.del_component(self.material_flow_terms)
        return self.material_flow_terms[j]

    def get_enthalpy_flow_terms(self, p):
        if not self.is_property_constructed("enthalpy_flow_terms"):
            try:

                def rule_enthalpy_flow_terms(b):
                    return self.enth_mol * self.flow_mol

                self.enthalpy_flow_terms = Expression(rule=rule_enthalpy_flow_terms)
            except AttributeError:
                self.del_component(self.enthalpy_flow_terms)
        return self.enthalpy_flow_terms

    def get_material_density_terms(self, p, j):
        # return b.dens_mol_comp[j]
        if not self.is_property_constructed("material_density_terms"):
            try:

                def rule_material_density_terms(b, j):
                    return b.dens_mol_comp[j]

                self.material_density_terms = Expression(
                    self.params.component_list, rule=rule_material_density_terms
                )
            except AttributeError:
                self.del_component(self.material_density_terms)
        return self.material_density_terms[j]

    def get_energy_density_terms(self, p):
        if not self.is_property_constructed("energy_density_terms"):
            try:

                def rule_energy_density_terms(b):
                    return self.enth_mol * self.dens_mol

                self.energy_density_terms = Expression(rule=rule_energy_density_terms)
            except AttributeError:
                self.del_component(self.energy_density_terms)
        return self.energy_density_terms

    def define_state_vars(b):
        return {
            "flow_mol": b.flow_mol,
            "temperature": b.temperature,
            "pressure": b.pressure,
            "mole_frac_comp": b.mole_frac_comp,
        }

    def get_material_flow_basis(b):
        return MaterialFlowBasis.molar

    def model_check(blk):
        """
        Model checks for property block
        """
        # Check temperature bounds
        if value(blk.temperature) < blk.temperature.lb:
            _log.error("{} Temperature set below lower bound.".format(blk.name))
        if value(blk.temperature) > blk.temperature.ub:
            _log.error("{} Temperature set above upper bound.".format(blk.name))

        # Check pressure bounds
        if value(blk.pressure) < blk.pressure.lb:
            _log.error("{} Pressure set below lower bound.".format(blk.name))
        if value(blk.pressure) > blk.pressure.ub:
            _log.error("{} Pressure set above upper bound.".format(blk.name))

    def default_material_balance_type(blk):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(blk):
        return EnergyBalanceType.enthalpyTotal

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # scale some variables
        if hasattr(self, "dens_mol"):
            for v in self.dens_mol.values():
                if iscale.get_scaling_factor(v) is None:
                    sf1 = iscale.get_scaling_factor(
                        self.pressure, default=1, warning=True
                    )
                    sf2 = iscale.get_scaling_factor(
                        self.temperature, default=1, warning=True
                    )
                    iscale.set_scaling_factor(v, sf1 / sf2)

        # scale some constraints
        if self.is_property_constructed("material_flow_terms"):
            for i, c in self.material_flow_terms.items():
                sf1 = iscale.get_scaling_factor(self.mole_frac_comp[i])
                sf2 = iscale.get_scaling_factor(self.flow_mol)
                iscale.set_scaling_factor(c, sf1 * sf2)

        if self.is_property_constructed("material_density_terms"):
            for i, c in self.material_density_terms.items():
                sf1 = iscale.get_scaling_factor(self.mole_frac_comp[i])
                sf2 = iscale.get_scaling_factor(self.dens_mol)
                iscale.set_scaling_factor(c, sf1 * sf2)

        if self.is_property_constructed("energy_density_terms"):
            for i, c in self.energy_density_terms.items():
                sf1 = iscale.get_scaling_factor(self.enth_mol)
                sf2 = iscale.get_scaling_factor(self.dens_mol)
                iscale.set_scaling_factor(c, sf1 * sf2)

        if self.is_property_constructed("enthalpy_flow_terms"):
            for i, c in self.enthalpy_flow_terms.items():
                sf1 = iscale.get_scaling_factor(self.enth_mol)
                sf2 = iscale.get_scaling_factor(self.flow_mol)
                iscale.set_scaling_factor(c, sf1 * sf2)

        # Scale some constraints
        if self.is_property_constructed("sum_component_eqn"):
            iscale.constraint_scaling_transform(
                self.sum_component_eqn,
                iscale.get_scaling_factor(self.mole_frac_comp["H2O"]),
                overwrite=False,
            )

        if self.is_property_constructed("mw_eqn"):
            sf = iscale.get_scaling_factor(self.mw)
            iscale.constraint_scaling_transform(self.mw_eqn, sf, overwrite=False)

        if self.is_property_constructed("ideal_gas"):
            sf = iscale.get_scaling_factor(self.pressure)
            iscale.constraint_scaling_transform(self.ideal_gas, sf, overwrite=False)

        if self.is_property_constructed("comp_conc_eqn"):
            for i, c in self.comp_conc_eqn.items():
                sf1 = iscale.get_scaling_factor(self.mole_frac_comp[i])
                sf2 = iscale.get_scaling_factor(self.dens_mol)
                iscale.constraint_scaling_transform(
                    self.comp_conc_eqn[i], sf1 * sf2, overwrite=False
                )

        if self.is_property_constructed("visc_d_constraint"):
            for i, c in self.visc_d_constraint.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.visc_d[i]), overwrite=False
                )
        if self.is_property_constructed("diffus_comp_constraint"):
            for i, c in self.diffus_comp_constraint.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.diffus_comp[i]),
                    overwrite=False,
                )
        if self.is_property_constructed("therm_cond_constraint"):
            for c in self.therm_cond_constraint.values():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.therm_cond), overwrite=False
                )
        if self.is_property_constructed("cp_shomate_eqn"):
            for i, c in self.cp_shomate_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.cp_mol_comp[i]), overwrite=False
                )
        if self.is_property_constructed("mixture_heat_capacity_eqn"):
            iscale.constraint_scaling_transform(
                self.mixture_heat_capacity_eqn,
                iscale.get_scaling_factor(self.cp_mol),
                overwrite=False,
            )
        if self.is_property_constructed("cp_mass_basis"):
            iscale.constraint_scaling_transform(
                self.cp_mass_basis,
                iscale.get_scaling_factor(self.cp_mass),
                overwrite=False,
            )
        if self.is_property_constructed("enthalpy_shomate_eqn"):
            for i, c in self.enthalpy_shomate_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.enth_mol_comp[i]), overwrite=False
                )
        if self.is_property_constructed("mixture_enthalpy_eqn"):
            iscale.constraint_scaling_transform(
                self.mixture_enthalpy_eqn,
                iscale.get_scaling_factor(self.enth_mol),
                overwrite=False,
            )
        if self.is_property_constructed("entropy_correlation"):
            iscale.constraint_scaling_transform(
                self.entropy_correlation,
                iscale.get_scaling_factor(self.entr_mol)
                * iscale.get_scaling_factor(self.flow_mol),
                overwrite=False,
            )
