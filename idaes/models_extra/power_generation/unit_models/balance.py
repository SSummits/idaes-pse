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
Basic unit model for mass and energy balances and if the has_pressure_change
option is true, Pout = Pin + deltaP.
"""

__author__ = "John Eslick"

# Import Pyomo libraries
from pyomo.environ import Reference
from pyomo.common.config import ConfigBlock, ConfigValue, In, Bool

# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    EnergyBalanceType,
    MomentumBalanceType,
    MaterialBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)


def make_balance_control_volume(o, name, config, dynamic=None, has_holdup=None):
    """
    This is seperated from the main heater class so it can be reused to create
    control volumes for different types of heat exchange models.
    """
    if dynamic is None:
        dynamic = config.dynamic
    if has_holdup is None:
        has_holdup = config.has_holdup
    if "has_heat_transfer" in config:
        has_heat_transfer = config.has_heat_transfer
    else:
        has_heat_transfer = False
    if "has_work_transfer" in config:
        has_work_transfer = config.has_work_transfer
    else:
        has_work_transfer = False
    # we have to attach this control volume to the model for the rest of
    # the steps to work
    o.add_component(
        name,
        ControlVolume0DBlock(
            dynamic=dynamic,
            has_holdup=has_holdup,
            property_package=config.property_package,
            property_package_args=config.property_package_args,
        ),
    )
    control_volume = getattr(o, name)
    # Add inlet and outlet state blocks to control volume
    if has_holdup:
        control_volume.add_geometry()
    control_volume.add_state_blocks(has_phase_equilibrium=config.has_phase_equilibrium)
    # Add material balance
    control_volume.add_material_balances(
        balance_type=config.material_balance_type,
        has_phase_equilibrium=config.has_phase_equilibrium,
    )
    # add energy balance
    control_volume.add_energy_balances(
        balance_type=config.energy_balance_type,
        has_heat_transfer=has_heat_transfer,
        has_work_transfer=has_work_transfer,
    )
    # add momentum balance
    control_volume.add_momentum_balances(
        balance_type=config.momentum_balance_type,
        has_pressure_change=config.has_pressure_change,
    )
    return control_volume


def make_balance_config_block(config):
    """
    Declare configuration options for HeaterData block.
    """
    config.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.useDefault.
**Valid values:** {
**MaterialBalanceType.useDefault - refer to property package for default
balance type
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    config.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.useDefault,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
**default** - EnergyBalanceType.useDefault.
**Valid values:** {
**EnergyBalanceType.useDefault - refer to property package for default
balance type
**EnergyBalanceType.none** - exclude energy balances,
**EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
**EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
**EnergyBalanceType.energyTotal** - single energy balance for material,
**EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )
    config.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
**default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )
    config.declare(
        "has_phase_equilibrium",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Phase equilibrium construction flag",
            doc="""Indicates whether terms for phase equilibrium should be
constructed, **default** = False.
**Valid values:** {
**True** - include phase equilibrium terms
**False** - exclude phase equilibrium terms.}""",
        ),
    )
    config.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
constructed,
**default** - False.
**Valid values:** {
**True** - include pressure change terms,
**False** - exclude pressure change terms.}""",
        ),
    )
    config.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    config.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
        ),
    )
    config.declare(
        "has_work_transfer",
        ConfigValue(
            default=True,
            domain=Bool,
            description="True if model a has work transfer term.",
        ),
    )
    config.declare(
        "has_heat_transfer",
        ConfigValue(
            default=True,
            domain=Bool,
            description="True if model has a heat transfer term.",
        ),
    )


@declare_process_block_class("BalanceBlock", doc="Mass/energy balance block.")
class BalanceBlockData(UnitModelBlockData):
    """
    Simple mass and energy balance unit.
    """

    CONFIG = UnitModelBlockData.CONFIG()
    make_balance_config_block(CONFIG)

    def build(self):
        """Building model

        Args:
            None
        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super().build()
        # Add Control Volume
        make_balance_control_volume(self, "control_volume", self.config)
        # Add Ports
        self.add_inlet_port()
        self.add_outlet_port()
        # Add convienient references to control volume quantities deltaP.
        if (
            self.config.has_pressure_change is True
            and self.config.momentum_balance_type != MomentumBalanceType.none
        ):
            self.deltaP = Reference(self.control_volume.deltaP)
        if self.config.has_heat_transfer is True:
            self.heat_duty = Reference(self.control_volume.heat)
        if self.config.has_work_transfer is True:
            self.work = Reference(self.control_volume.work)
