
from re import U
from pyomo.environ import (
    Binary,
    Integers,
    Param,
    Var,
    Constraint,
    Expression,
    NonNegativeReals,
    log,
    exp,
    units as pyunits,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.constants import Constants as CONST
from idaes.core.util import scaling as iscale


def build_intensified_packing_model(blk, mode):

    _reformat_column_model(blk)
    _add_packing_equations(blk, mode)
    _activate_packing_model(blk, mode)
    
def _delete_old_components(blk):
    old_components = (
        "eps_ref",
        "hydraulic_diameter",
        "holdup_vap_eqn",
        "log_holdup_vap_eqn",
        "vapor_phase_area",
        "liquid_phase_area",
        "wetted_perimeter",
        "gas_velocity_fld_eqn",
        "mass_transfer_coeff_vap_eqn",
        "mass_transfer_coeff_liq_eqn",
        "area_interfacial_eqn",
    )
    for comp in old_components:
        blk.del_component(comp)

def _reformat_column_model(blk):
    """Changes components of the column model to be compatible with intensified packing.
        Reconstructs various components of the column model to be indexed by the length domain.

        Components changed:
            - `eps_ref`: Packing void space, m3/m3
                 - Results in 'hydraulic_diameter', 'wetted_perimeter' being indexed by the length domain
                 causing a cascade of changes to other components.
            - `hydraulic_diameter`: Hydraulic diameter, m
            - `holdup_vap`: Volumetric vapor holdup, [-]
            - `vapor_phase_area`: Vapor phase cross-sectional area constraint
            - `liquid_phase_area`: Liquid phase cross-sectional area constraint
            - `wetted_perimeter`: Wetted perimeter, m
            - `gas_velocity_fld_eqn`: Gas velocity field equation
            - `mass_transfer_coeff_vap_eqn`: Vapor phase mass transfer coefficient
            - `mass_transfer_coeff_liq_eqn`: Liquid phase mass transfer coefficient
            - `area_interfacial_eqn`: Specific interfacial area constraint

        Args:
            blk: Block to modify

        Returns:
            None
        """
    lunits = (
        blk.config.liquid_phase.property_package.get_metadata().get_derived_units
    )

    _delete_old_components(blk)
    
    # === Component changes stemming from eps_ref ===
    blk.eps_ref = Var(
        blk.liquid_phase.length_domain,
        initialize=0.97,
        units=pyunits.dimensionless,
        bounds = (0.2, 0.97),
        doc="Packing void space m3/m3",
        )
    blk.eps_ref.fix()

    @blk.Expression(
        blk.liquid_phase.length_domain)
    def hydraulic_diameter(blk, x):
        return 4 * blk.eps_ref[x] / blk.packing_specific_area
    
    # @blk.Expression(
    #     blk.flowsheet().time,
    #     blk.vapor_phase.length_domain,
    #     doc="Volumetric vapor holdup [-]",
    # )
    # def holdup_vap(blk, t, x):
    #     if x == blk.vapor_phase.length_domain.first():
    #         return Expression.Skip
    #     else:
    #         zb = blk.vapor_phase.length_domain.prev(x)
    #         return blk.eps_ref[x] - blk.holdup_liq[t, zb]
    # blk.holdup_vap = Var(
    #         blk.flowsheet().time,
    #         blk.vapor_phase.length_domain,
    #         initialize=1-0.97-0.001,
    #         units=pyunits.dimensionless,
    #         doc="Volumetric vapor holdup [-]",
    #     )
    @blk.Constraint(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
    )
    def holdup_vap_eqn(b, t, x):
        if x == blk.vapor_phase.length_domain.first():
            return Constraint.Skip
        else:
            zb = blk.vapor_phase.length_domain.prev(x)
            return b.holdup_vap[t, x] == b.eps_ref[x] - b.holdup_liq[t, zb]
        
    @blk.Constraint(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
        doc="""Defines log variable for vapor holdup""",
    )
    def log_holdup_vap_eqn(b, t, x):
        if x == b.vapor_phase.length_domain.first():
            return Constraint.Skip
        else:
            return exp(b.log_holdup_vap[t, x]) == b.holdup_vap[t, x]

    @blk.Constraint(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
        doc="Vapor phase cross-sectional area constraint",
    )
    def vapor_phase_area(blk, t, x):
        if x == blk.vapor_phase.length_domain.first():
            return blk.vapor_phase.area[t, x] == (blk.eps_ref[x] * blk.area_column)
        else:
            return blk.vapor_phase.area[t, x] == (
                blk.area_column * blk.holdup_vap[t, x]
            )
        
    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Liquid phase cross-sectional area constraint",
    )
    def liquid_phase_area(blk, t, x):
        if x == blk.liquid_phase.length_domain.last():
            return blk.liquid_phase.area[t, x] == (blk.eps_ref[x] * blk.area_column)
        else:
            return blk.liquid_phase.area[t, x] == (
                blk.area_column * blk.holdup_liq[t, x]
            )
    
    @blk.Expression(blk.liquid_phase.length_domain)
    def wetted_perimeter(blk, x):
        return blk.area_column * blk.packing_specific_area / blk.eps_ref[x]
    
    @blk.Constraint(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
    )
    def gas_velocity_fld_eqn(blk, t, x):
        if x == blk.vapor_phase.length_domain.first():
            return Expression.Skip
        else:
            x_liq = blk.liquid_phase.length_domain.prev(x)
            log_packing_specific_area = log(
                blk.packing_specific_area
                / (lunits("length") ** 2 / lunits("length") ** 3)
            )
            log_g = log(
                pyunits.convert(
                    CONST.acceleration_gravity, to_units=lunits("acceleration")
                )
                / lunits("acceleration")
            )
            # Value of 0.001 Pa*s for viscosity of water was here when I found it: Doug
            log_visc_d_liq_H2O = log(
                pyunits.convert(
                    0.001 * pyunits.Pa * pyunits.s,
                    to_units=lunits("dynamic_viscosity"),
                )
                / lunits("dynamic_viscosity")
            )
            return blk.log_gas_velocity_fld[t, x] == 0.5 * (
                log_g
                + 3 * log(blk.eps_ref[x])
                - log_packing_specific_area
                + blk.log_dens_mass_liq[t, x_liq]
                - blk.log_dens_mass_vap[t, x]
                - 0.2 * (blk.log_visc_d_liq[t, x_liq] - log_visc_d_liq_H2O)
                - 4 * blk.fourth_root_flood_H[t, x]
            )
        
    # === Component changes stemming from hydraulic_diameter ===
    @blk.Constraint(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
        blk.equilibrium_comp,
        doc="Vapor phase mass transfer coefficient",
    )
    def mass_transfer_coeff_vap_eqn(blk, t, x, j):
        if x == blk.vapor_phase.length_domain.first():
            return Constraint.Skip
        else:
            log_R_gas = log(
                pyunits.convert(CONST.gas_constant, lunits("gas_constant"))
                / lunits("gas_constant")
            )
            log_vapor_temp = log(
                pyunits.convert(
                    blk.vapor_phase.properties[t, x].temperature,
                    to_units=lunits("temperature"),
                )
                / lunits("temperature")
            )
            log_packing_specific_area = log(
                blk.packing_specific_area
                / (lunits("length") ** 2 / lunits("length") ** 3)
            )
            log_hydraulic_diameter = log(blk.hydraulic_diameter[x] / lunits("length"))
            return blk.log_mass_transfer_coeff_vap[t, x, j] == (
                blk.log_Cv_ref
                - log_R_gas
                - log_vapor_temp
                - 0.5 * blk.log_holdup_vap[t, x]
                + 0.5 * (log_packing_specific_area - log_hydraulic_diameter)
                + (2 / 3) * blk.log_diffus_vap_comp[t, x, j]
                + (1 / 3) * (blk.log_visc_d_vap[t, x] - blk.log_dens_mass_vap[t, x])
                + (3 / 4)
                * (
                    blk.log_velocity_vap[t, x]
                    + blk.log_dens_mass_vap[t, x]
                    - log_packing_specific_area
                    - blk.log_visc_d_vap[t, x]
                )
            )
    
    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        blk.solute_comp_list,
        doc="Liquid phase mass transfer coefficient",
    )
    def mass_transfer_coeff_liq_eqn(blk, t, x, j):
        if x == blk.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            log_hydraulic_diameter = log(blk.hydraulic_diameter[x] / lunits("length"))
            return blk.log_mass_transfer_coeff_liq[t, x, j] == (
                blk.log_Cl_ref
                + (1 / 6) * log(12)
                + 0.5
                * (
                    blk.log_velocity_liq[t, x]
                    + blk.log_diffus_liq_comp[t, x, j]
                    - blk.log_holdup_liq[t, x]
                    - log_hydraulic_diameter
                )
            )

    # === Component changes stemming from wetted_perimeter ===
    @blk.Constraint(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
        doc="Defines specific interfacial area",
    )
    def area_interfacial_eqn(blk, t, x):
        if x == blk.vapor_phase.length_domain.first():
            return Constraint.Skip
        else:
            x_liq = blk.liquid_phase.length_domain.prev(x)
            log_packing_specific_area = log(
                blk.packing_specific_area
                / (lunits("length") ** 2 / lunits("length") ** 3)
            )
            log_g = log(
                pyunits.convert(
                    CONST.acceleration_gravity, to_units=lunits("acceleration")
                )
                / lunits("acceleration")
            )
            log_cross_sectional_area = log(blk.area_column / lunits("area"))
            log_wetted_perimeter = log(blk.wetted_perimeter[x] / lunits("length"))
            return blk.log_area_interfacial[t, x] == (
                log_packing_specific_area
                + blk.log_area_interfacial_parA
                + blk.area_interfacial_parB
                * (
                    blk.log_dens_mass_liq[t, x_liq]
                    - blk.log_surf_tens_liq[t, x_liq]
                    + (1 / 3) * log_g
                    + (4 / 3)
                    * (
                        blk.log_velocity_liq[t, x_liq]
                        + log_cross_sectional_area
                        - log_wetted_perimeter
                    )
                )
            )

def _add_packing_equations(blk, mode):
    if mode != "absorber" and mode != "stripper":
        raise ConfigurationError('Invalid column mode for internal heat exchanger model')
    
    print('Here we go again')
    
    lunits = (
        blk.config.liquid_phase.property_package.get_metadata().get_derived_units
    )
    
    blk.eps_ref_base = Param(initialize=0.97,
                         units=pyunits.dimensionless,
                         doc="Packing void space m3/m3")
        
    # Embedded Heat Exchanger placement parameters
    blk.N_start = Var(
        blk.vapor_phase.length_domain,
        initialize=0,
        within=Binary,
        doc="Heat exchanger precesence in element i")
    blk.N_start.fix()
    
    blk.N = Var(
        blk.vapor_phase.length_domain,
        initialize=0,
        within=Binary,
        doc="Heat exchanger precesence in element i")
    blk.N.fix()
    
    blk.N_min = Param(
        initialize = 5,
        mutable = True,
        doc = "Min CVs for Internal HE")
    
    blk.HE_Penalty = Param(
        initialize=0.4,
        mutable=True,
        doc="Penalty to void space and surface area due to HE prescence")
    
    blk.d_HE = Var(
        # blk.vapor_phase.length_domain,
        within=Integers,
        bounds=(-1,1),
        initialize=0,
        doc="Specifies the direction of the coolant flow: 1 for down, -1 for up, 0 for no flow")
    blk.d_HE.fix()
        
    
    # ======================================================================
    # Embedded heat exchanger performance variables and contraints
    blk.T_util = Var(blk.flowsheet().time,
                   blk.liquid_phase.length_domain,
                   units=pyunits.K,
                   domain=NonNegativeReals,
                   # bounds=(303.15, 350),
                   initialize=303.15,
                   doc='Cooling water temperature')
    blk.T_util.fix()
    if mode == "absorber":
        blk.T_util.setlb(303.15)
        blk.T_util.setub(313.15)
    if mode == "stripper":
        blk.T_util.fix(423)
        blk.T_util.setlb(273.15+80)
        blk.T_util.setub(273.15+308)
    
    blk.U = Var(initialize=35, # XXX: 32.5 - 34.9 (Miramontes and Tsouris 2020)
                 domain=NonNegativeReals,
                 units=pyunits.J/ (pyunits.s * pyunits.m**2 * pyunits.K),
                 doc='Heat Transfer Coefficient')
    blk.U.fix()
    if mode == 'stripper':
        blk.U.fix(750)
    
    blk.mcw = Var(blk.liquid_phase.length_domain,
                   initialize=50,
                   domain=NonNegativeReals,
                   units=pyunits.mol / pyunits.s, 
                   doc='Cooling Water Flowrate')
    blk.mcw.fix()

    blk.cpcw = Var(initialize=75.40,
                    domain=NonNegativeReals,
                    units=pyunits.J / pyunits.mol * pyunits.K,
                    doc='Specific heat of Cooling Water' )
    blk.cpcw.fix()
    
    blk.diameter_util_tube = Var(initialize=0.02,
                                domain=NonNegativeReals,
                                units=pyunits.m,
                                doc='Diameter of cooling water cooling tubes')
    blk.diameter_util_tube.fix()
    
    blk.specific_area_util = Var(blk.liquid_phase.length_domain,
                                initialize=200,
                                domain=NonNegativeReals,
                                bounds=(0,300),
                                units=pyunits.m**2 / pyunits.m**3,
                                doc='Specific surface area of cooling water tube')
        
    blk.eps_util = Var(blk.liquid_phase.length_domain,
                      initialize=0.0,
                      domain=NonNegativeReals,
                      bounds=(0.0,0.13),
                      units=pyunits.m**3 / pyunits.m**3,
                      doc='Void fraction of cooling water channels',)
    blk.eps_util.fix()
    
    @blk.Constraint(blk.liquid_phase.length_domain,
        doc='Constraint linking the diameter of cooling water tube with specific surfacea area')
    def specific_area_util_con(blk, x):
        # return blk.specific_area_util[x] == (blk.eps_ref[x]/blk.eps_ref_base) * blk.packing_specific_area
        return blk.specific_area_util[x] == (blk.eps_util[x]/blk.eps_util[x].ub) * (1-blk.eps_util[x]) * blk.packing_specific_area
        # return blk.specific_area_util[x] * blk.diameter_util_tube * 0.25 == blk.eps_util[x]
    
    # TODO: Figure out a better way to implement penalty
    @blk.Constraint(
        blk.liquid_phase.length_domain,
        doc="Void space penalty for HE")
    def eps_ref_penalty(blk, x):
        return blk.eps_ref[x] == blk.eps_ref_base - blk.eps_util[x]
    
    # @blk.Constraint(
    #     blk.vapor_phase.length_domain,
    #     doc="Constraint to include penalty for HE use")
    # def HE_Area_Penalty(blk, i):
    #     return blk.a_ref[i] == blk.a_ref_base #* (1 - blk.HE_Penalty * blk.N[i])
    
    @blk.Expression(blk.liquid_phase.length_domain)
    def CV_Volume(blk, x):
        diameter = blk.diameter_column
        CV_length = blk.liquid_phase.length_domain.next(0) * blk.length_column
        return 0.25 * 3.14159 * diameter**2 * CV_length
    
    blk.heat_util = Var(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        initialize=0,
        units=pyunits.W / pyunits.m,
        bounds=(-1e9, 1e9),
        doc="Heat exchanged to liquid phase (W/m)",
    )
    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Heat exchanged to liquid phase (W/m)")
    def heat_util_eqn(blk, t, x):
        N = blk.N[x]
        U = blk.U
        a_p = blk.specific_area_util[x]
        area = blk.area_column
        T_util = blk.T_util[t,x]
        T_L = blk.liquid_phase.properties[t,x].temperature
        return blk.heat_util[t,x] == N * U * a_p * area * (T_util - T_L)
        # return blk.N[x] * blk.U * blk.specific_area_util[x] * blk.area_column * (blk.T_util[t,x] - blk.liquid_phase.properties[t,x].temperature)
    
    # @blk.Expression(
    #     blk.flowsheet().time,
    #     blk.liquid_phase.length_domain,
    #     doc="Heat exchanged to liquid phase (W/m)")
    # def heat_util(blk, t, x):
    #     return blk.N[x] * blk.U * blk.specific_area_util[x] * blk.area_column * (blk.T_util[t,x] - blk.liquid_phase.properties[t,x].temperature)
    
    @blk.Expression(
        blk.flowsheet().time,
    )
    def heat_util_total(blk, t):
        heat_util_per_m_total = 0
        delta_x = blk.liquid_phase.length_domain.next(0)
        for x in blk.liquid_phase.length_domain:
            if x != blk.liquid_phase.length_domain.first():
                heat_util_per_m_total += blk.heat_util[t,x] * delta_x
        return heat_util_per_m_total * blk.length_column
    
    # ======================================================================
    # Logic Constraints for HE packing placement
        
    blk.L_min = Var(
        initialize=0.0,
        doc="Minimum length of internal heat exchanger")
    blk.L_min.fix()
        
    @blk.Constraint(
        blk.vapor_phase.length_domain,
        blk.vapor_phase.length_domain,)
    def HE_Con1(blk, x, y):
        if x > y:
            return Constraint.Skip
        else:
            return blk.N[y] >= blk.N_start[x]*(x+blk.L_min - y)
        
    @blk.Constraint(
        blk.vapor_phase.length_domain)
    def HE_Con2(blk, x):
        if x == blk.vapor_phase.length_domain.first():
            return blk.N_start[x] >= blk.N[x]
        else:
            xb = blk.vapor_phase.length_domain.prev(x)
            return blk.N_start[x] >= blk.N[x] - blk.N[xb]
    
    # ======================================================================
    # Cooling water temperature
    
    if mode == "absorber":
        Cp_util = 4184
        @blk.Constraint(
            blk.flowsheet().time,
            blk.vapor_phase.length_domain,
            doc="inequality for temperature at element i with previous element")
        def CW_temp_con1(blk, t, x):
            if x == blk.vapor_phase.length_domain.first():
                return Constraint.Skip
            else:
                zb = blk.vapor_phase.length_domain.prev(x)
                return blk.T_util[t,x] >= (blk.T_util[t,zb] -(-blk.heat_util[t,zb])/(blk.mcw[zb]*blk.cpcw*250*0.6*0.333*0.04*15))\
                        * blk.d_HE\
                        * blk.N[x] * blk.N[zb] # XXX: Change later
                    #blk.d_HE[x]*blk.d_HE[zb]
                # return blk.T_util[t,x] >= (blk.T_util[t,zb] -(blk.Q_to_vapor[t,zb] + blk.Q_to_liquid[t,zb])/(blk.mcw[zb]*Cp_util))*blk.d_HE[x]*blk.d_HE[zb]
        # blk.CW_temp_con1.deactivate()
        
        @blk.Constraint(
            blk.flowsheet().time,
            blk.vapor_phase.length_domain,
            doc="inequality for temperature at element i with next element")
        def CW_temp_con2(blk, t, x):
            if x == blk.vapor_phase.length_domain.last():
                return Constraint.Skip
            else:
                zn = blk.vapor_phase.length_domain.next(x)
                return blk.T_util[t,x] >= (blk.T_util[t,zn] -(-blk.heat_util[t,zn])/(blk.mcw[zn]*blk.cpcw*250*0.6*0.333*0.04*15))\
                        * -blk.d_HE\
                        * blk.N[x] * blk.N[zn]
                    #*blk.d_HE[x]*blk.d_HE[zn]
                    
    # Heat transfer
    lunits = (
        blk.config.liquid_phase.property_package.get_metadata().get_derived_units
    )
    
    @blk.Constraint(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
        doc="Heat transfer calculation",
    )
    def heat_transfer_with_utility_eqn1(blk, t, x):
        if x == blk.vapor_phase.length_domain.first():
            return Constraint.Skip
        else:
            zb = blk.liquid_phase.length_domain.prev(x)
            return pyunits.convert(
                blk.vapor_phase.heat[t, x],
                to_units=lunits("power") / lunits("length"),
            ) == (
                blk.heat_transfer_coeff[t, x]
                * (
                    blk.liquid_phase.properties[t, zb].temperature
                    - pyunits.convert(
                        blk.vapor_phase.properties[t, x].temperature,
                        to_units=lunits("temperature"),
                    )
                )
            )
    blk.heat_transfer_eqn1.deactivate()

    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Heat transfer balance",
    )
    def heat_transfer_with_utility_eqn2(blk, t, x):
        if x == blk.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            zf = blk.vapor_phase.length_domain.next(x)
            return blk.liquid_phase.heat[t, x] == -pyunits.convert(
                blk.vapor_phase.heat[t, zf],
                to_units=lunits("power") / lunits("length")) + blk.heat_util[t,zf]
    blk.heat_transfer_eqn2.deactivate()
    
    # Scale heat transfer constraints
    for (t, x), v in blk.heat_transfer_with_utility_eqn1.items():
        iscale.constraint_scaling_transform(
            v,
            iscale.get_scaling_factor(
                blk.vapor_phase.heat[t, x], default=1, warning=True
            ),
        )

    for (t, x), v in blk.heat_transfer_with_utility_eqn2.items():
        iscale.constraint_scaling_transform(
            v,
            iscale.get_scaling_factor(
                blk.liquid_phase.heat[t, x], default=1, warning=True
            ),
        )

def _activate_packing_model(blk, mode):
    if mode != "absorber" and mode != "stripper":
        raise ConfigurationError('Invalid column mode for internal heat exchanger model')
    
    elif mode == "absorber":
        blk.eps_ref.unfix()
        blk.eps_util.unfix()
        blk.CW_temp_con1.deactivate()
        blk.CW_temp_con2.deactivate()
    blk.eps_ref.unfix()
    blk.eps_util.unfix()
    blk.N.unfix()
    
    blk.N_start.unfix()

    blk.HE_Con1.deactivate()
    blk.HE_Con2.deactivate()
