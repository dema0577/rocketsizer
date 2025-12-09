"""
RocketSizer - Educational Software for Liquid Propellant Rocket Engine Sizing
=============================================================================

This software calculates the main dimensions of a liquid propellant rocket engine
based on user input parameters, and provides a production cost estimate.

Main formulas based on:
- Huzel & Huang, "Design of Liquid Propellant Rocket Engines", NASA SP-125
- Sutton & Biblarz, "Rocket Propulsion Elements"
- Anderson, "Fundamentals of Aerodynamics"

Author: [Matteo De Martini]
Date: 09/12/2025
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle



class RocketEngine:
    """
    Class for the physical sizing of a rocket engine.
    
    Implements the fundamental rocket propulsion equations to calculate:
    - Mass flow rate
    - Throat and nozzle dimensions
    - Combustion chamber dimensions
    """
    
    # Lookup dictionary for propellant properties
    # Format: (Flame Temperature [K], Molecular Weight [kg/kmol], 
    #          Specific Heat Ratio gamma, Theoretical Isp [s])
    PROPELLANT_PROPERTIES = {
        "LOX/Kerosene (RP-1)": {
            "Tc": 3670,      # Adiabatic combustion temperature [K]
            "MW": 23.2,      # Average molecular weight of combustion gases [kg/kmol]
            "gamma": 1.22,   # Specific heat ratio Cp/Cv
            "Isp_theoretical": 330,  # Theoretical Isp at sea level [s]
            "density_oxidizer": 1141,  # LOX density [kg/m³]
            "density_fuel": 820       # RP-1 density [kg/m³]
        },
        "LOX/Liquid Hydrogen": {
            "Tc": 3580,
            "MW": 16.0,
            "gamma": 1.25,
            "Isp_theoretical": 360,
            "density_oxidizer": 1141,
            "density_fuel": 71
        },
        "LOX/Methane": {
            "Tc": 3550,
            "MW": 20.5,
            "gamma": 1.23,
            "Isp_theoretical": 340,
            "density_oxidizer": 1141,
            "density_fuel": 423
        },
        "N2O4/MMH": {
            "Tc": 3200,
            "MW": 25.0,
            "gamma": 1.24,
            "Isp_theoretical": 310,
            "density_oxidizer": 1442,
            "density_fuel": 880
        }
    }
    
    def __init__(self, thrust_kN, Pc_bar, propellant_name, OF_ratio):
        """
        Initialize the engine with input parameters.
        
        Parameters:
        ----------
        thrust_kN : float
            Desired thrust in kilonewtons
        Pc_bar : float
            Combustion chamber pressure in bar
        propellant_name : str
            Propellant name (must be in PROPELLANT_PROPERTIES dictionary)
        OF_ratio : float
            Oxidizer-to-fuel mixture ratio
        """
        self.thrust_N = thrust_kN * 1000  # Conversion kN -> N
        self.Pc = Pc_bar * 100000  # Conversion bar -> Pa
        self.propellant_name = propellant_name
        self.OF_ratio = OF_ratio
        
        # Load propellant properties
        if propellant_name not in self.PROPELLANT_PROPERTIES:
            raise ValueError(f"Propellant '{propellant_name}' not recognized")
        
        props = self.PROPELLANT_PROPERTIES[propellant_name]
        self.Tc = props["Tc"]
        self.MW = props["MW"]
        self.gamma = props["gamma"]
        self.Isp_theoretical = props["Isp_theoretical"]
        self.density_oxidizer = props["density_oxidizer"]
        self.density_fuel = props["density_fuel"]
        
        # Physical constants
        self.g0 = 9.80665  # Standard gravitational acceleration [m/s²]
        self.R_universal = 8314.462618  # Universal gas constant [J/(kmol·K)]
        
        # Calculate derived properties
        self._calculate_engine_parameters()
    
    def _calculate_engine_parameters(self):
        """
        Performs all engine sizing calculations.
        
        This function applies the fundamental propulsion equations:
        1. Thrust equation to calculate mass flow rate
        2. Isentropic flow equations to size throat and nozzle
        3. L* parameter to size the combustion chamber
        """
        
        # ====================================================================
        # CALCULATION 1: Total Mass Flow Rate (m_dot)
        # ====================================================================
        # Thrust equation: F = Isp * g0 * m_dot
        # Therefore: m_dot = F / (Isp * g0)
        # 
        # Note: In reality, for an ideal nozzle: F = m_dot * Ve + (Pe - Pa) * Ae
        # For simplicity, we use the simplified form assuming optimal expansion.
        # 
        # Source: Huzel & Huang, Section 3-3
        
        self.Isp_effective = self.Isp_theoretical * 0.92  # Efficiency factor (0.92 = 92%)
        self.m_dot = self.thrust_N / (self.Isp_effective * self.g0)  # [kg/s]
        
        # Separate mass flows for oxidizer and fuel
        self.m_dot_oxidizer = self.m_dot * (self.OF_ratio / (1 + self.OF_ratio))
        self.m_dot_fuel = self.m_dot * (1 / (1 + self.OF_ratio))
        
        # ====================================================================
        # CALCULATION 2: Throat Area (At)
        # ====================================================================
        # From the isentropic flow equation for a converging-diverging nozzle:
        # m_dot = (Pc * At) / sqrt(R*Tc) * sqrt(gamma) * (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))
        #
        # Solving for At:
        # At = m_dot * sqrt(R*Tc) / (Pc * sqrt(gamma) * (2/(gamma+1))^((gamma+1)/(2*(gamma-1))))
        #
        # Where:
        # - R = R_universal / MW is the specific gas constant
        # - gamma is the specific heat ratio
        # - (2/(gamma+1))^((gamma+1)/(2*(gamma-1))) is the critical flow coefficient
        #
        # Source: Huzel & Huang, Section 3-4; Anderson, "Fundamentals of Aerodynamics"
        
        R_specific = self.R_universal / self.MW  # [J/(kg·K)]
        
        # Critical flow coefficient (at throat, M=1)
        critical_flow_coeff = np.sqrt(self.gamma) * \
                             (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (2 * (self.gamma - 1)))
        
        self.At = (self.m_dot * np.sqrt(R_specific * self.Tc)) / \
                  (self.Pc * critical_flow_coeff)  # [m²]
        
        self.Dt = np.sqrt(4 * self.At / np.pi)  # Throat diameter [m]
        
        # ====================================================================
        # CALCULATION 3: Exit Area (Ae) and Expansion Ratio
        # ====================================================================
        # For optimal expansion at sea level (Pa = 1 atm = 101325 Pa),
        # the exit area is calculated from the isentropic relation:
        # Pe/Pc = (1 + (gamma-1)/2 * Me²)^(-gamma/(gamma-1))
        #
        # Solving for Me (exit Mach) with Pe = Pa:
        # Me = sqrt(2/(gamma-1) * ((Pc/Pa)^((gamma-1)/gamma) - 1))
        #
        # Then, from the isentropic area ratio:
        # Ae/At = (1/Me) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * Me²))^((gamma+1)/(2*(gamma-1)))
        #
        # Source: Huzel & Huang, Section 3-4
        
        Pa = 101325  # Ambient pressure at sea level [Pa]
        Pc_over_Pa = self.Pc / Pa
        
        # Calculate exit Mach for optimal expansion
        self.Me = np.sqrt((2 / (self.gamma - 1)) * 
                          ((Pc_over_Pa) ** ((self.gamma - 1) / self.gamma) - 1))
        
        # Area ratio using isentropic formula
        area_ratio_coeff = (2 / (self.gamma + 1)) * \
                          (1 + (self.gamma - 1) / 2 * self.Me ** 2)
        self.epsilon = (1 / self.Me) * \
                       (area_ratio_coeff ** ((self.gamma + 1) / (2 * (self.gamma - 1))))
        
        self.Ae = self.At * self.epsilon  # [m²]
        self.De = np.sqrt(4 * self.Ae / np.pi)  # Exit diameter [m]
        
        # ====================================================================
        # CALCULATION 4: Combustion Chamber Dimensions
        # ====================================================================
        # The L* (L-star) parameter is defined as:
        # L* = Vc / At
        # where Vc is the combustion chamber volume.
        #
        # L* is related to the residence time of gases in the chamber, necessary
        # to complete combustion. Typical values:
        # - LOX/RP-1: 0.8 - 1.5 m
        # - LOX/H2: 1.0 - 2.5 m
        # - Hypergolic: 0.6 - 1.2 m
        #
        # Source: Huzel & Huang, Section 8-3
        
        # L* selection based on propellant type
        L_star_values = {
            "LOX/Kerosene (RP-1)": 1.0,
            "LOX/Liquid Hydrogen": 1.5,
            "LOX/Methane": 1.2,
            "N2O4/MMH": 0.8
        }
        self.L_star = L_star_values.get(self.propellant_name, 1.0)  # [m]
        
        # Chamber volume
        self.Vc = self.L_star * self.At  # [m³]
        
        # Assume a cylindrical chamber with typical length/diameter ratio
        # Typically Dc/Dt ≈ 3-5 to ensure good combustion conditions
        Dc_over_Dt = 4.0  # Typical value
        self.Dc = self.Dt * Dc_over_Dt  # [m]
        
        # Cylindrical chamber length (ignore convergent section for simplicity)
        # Vc ≈ π * (Dc/2)² * Lc_cylindrical
        self.Lc_cylindrical = self.Vc / (np.pi * (self.Dc / 2) ** 2)  # [m]
        
        # Approximate total length (chamber + convergent)
        # Convergent typically has an angle of 30-45°
        self.L_convergent = (self.Dc - self.Dt) / (2 * np.tan(np.radians(30)))  # [m]
        self.L_total = self.Lc_cylindrical + self.L_convergent  # [m]
        
        # ====================================================================
        # CALCULATION 5: Divergent Nozzle Length
        # ====================================================================
        # For a conical nozzle with divergence angle θ (typically 15°):
        # L_divergent = (De - Dt) / (2 * tan(θ))
        #
        # For a bell nozzle (contour nozzle), the length is ~60-80% of the conical one
        # We assume a simplified optimal bell nozzle (Rao contour)
        #
        # Source: Huzel & Huang, Section 3-4
        
        theta_divergent = 15  # Divergence angle [°]
        self.L_divergent = (self.De - self.Dt) / (2 * np.tan(np.radians(theta_divergent)))  # [m]
        self.L_nozzle_total = self.L_convergent + self.L_divergent  # [m]
    
    def get_summary(self):
        """
        Returns a dictionary with all calculated parameters.
        """
        return {
            "Total Mass Flow": f"{self.m_dot:.3f} kg/s",
            "Oxidizer Mass Flow": f"{self.m_dot_oxidizer:.3f} kg/s",
            "Fuel Mass Flow": f"{self.m_dot_fuel:.3f} kg/s",
            "Effective Isp": f"{self.Isp_effective:.1f} s",
            "Throat Area (At)": f"{self.At*10000:.2f} cm²",
            "Throat Diameter (Dt)": f"{self.Dt*1000:.2f} mm",
            "Exit Area (Ae)": f"{self.Ae*10000:.2f} cm²",
            "Exit Diameter (De)": f"{self.De*1000:.2f} mm",
            "Expansion Ratio (ε)": f"{self.epsilon:.2f}",
            "Exit Mach": f"{self.Me:.2f}",
            "Chamber Volume (Vc)": f"{self.Vc*1000:.2f} L",
            "Chamber Diameter (Dc)": f"{self.Dc*1000:.2f} mm",
            "Chamber Length": f"{self.Lc_cylindrical*1000:.2f} mm",
            "Total Engine Length": f"{(self.L_total + self.L_divergent)*1000:.2f} mm",
            "L* (L-star)": f"{self.L_star:.2f} m"
        }


# ============================================================================
# CLASS: CostEstimator
# ============================================================================
class CostEstimator:
    """
    Class for estimating engine production costs.
    
    Implements a simplified model based on:
    - Material mass (mainly Inconel for chamber/nozzle)
    - Machining costs
    - Fixed costs (injection systems, turbopumps, etc.)
    """
    
    def __init__(self, engine: RocketEngine):
        """
        Initialize the estimator with a sized engine.
        
        Parameters:
        ----------
        engine : RocketEngine
            Instance of the already calculated engine
        """
        self.engine = engine
        
        # Cost parameters (typical values for research/development engines)
        self.material_density = 8200  # Inconel 718 density [kg/m³]
        self.cost_per_kg_material = 150  # $/kg for machined material
        self.machining_factor = 3.0  # Multiplicative factor for complex machining
        self.base_cost = 50000  # Base fixed costs [$]
        self.complexity_factor = 1.5  # Additional complexity factor
        
        # Initialize total_cost to None (will be calculated when needed)
        self.total_cost = None
        self.total_mass = None
    
    def estimate_mass(self):
        """
        Estimates engine mass based on geometric dimensions.
        
        Calculates mass as sum of:
        - Combustion chamber (cylinder)
        - Convergent section
        - Divergent section (nozzle)
        - Wall thickness (assumed constant)
        """
        wall_thickness = 0.003  # Typical wall thickness [m] (3 mm)
        
        # Cylindrical chamber mass
        volume_chamber = np.pi * ((self.engine.Dc / 2 + wall_thickness) ** 2 - 
                                  (self.engine.Dc / 2) ** 2) * self.engine.Lc_cylindrical
        mass_chamber = volume_chamber * self.material_density
        
        # Convergent mass (truncated cone)
        R_inner_top = self.engine.Dc / 2
        R_inner_bot = self.engine.Dt / 2
        R_outer_top = R_inner_top + wall_thickness
        R_outer_bot = R_inner_bot + wall_thickness
        
        volume_convergent = (np.pi * self.engine.L_convergent / 3) * \
                           (R_outer_top ** 2 + R_outer_top * R_outer_bot + R_outer_bot ** 2) - \
                           (np.pi * self.engine.L_convergent / 3) * \
                           (R_inner_top ** 2 + R_inner_top * R_inner_bot + R_inner_bot ** 2)
        mass_convergent = volume_convergent * self.material_density
        
        # Divergent mass (truncated cone)
        R_inner_exit = self.engine.De / 2
        R_outer_exit = R_inner_exit + wall_thickness
        
        volume_divergent = (np.pi * self.engine.L_divergent / 3) * \
                          (R_outer_exit ** 2 + R_outer_exit * R_outer_bot + R_outer_bot ** 2) - \
                          (np.pi * self.engine.L_divergent / 3) * \
                          (R_inner_exit ** 2 + R_inner_exit * R_inner_bot + R_inner_bot ** 2)
        mass_divergent = volume_divergent * self.material_density
        
        self.total_mass = mass_chamber + mass_convergent + mass_divergent
        return self.total_mass
    
    def estimate_cost(self):
        """
        Estimates total engine cost.
        
        Simplified formula:
        Cost = (Mass * Cost_per_kg * Machining_factor * Complexity_factor) + Fixed_costs
        
        Fixed costs include:
        - Injection system
        - Regenerative cooling
        - Production equipment
        - Testing
        """
        mass = self.estimate_mass()
        
        # Material and machining cost
        material_cost = mass * self.cost_per_kg_material * \
                       self.machining_factor * self.complexity_factor
        
        # Variable costs (scale with dimensions)
        # Injection system and cooling
        injection_cost = 20000 * (self.engine.At * 10000) ** 0.5  # Scales with throat area
        cooling_cost = 15000 * (self.engine.L_total * 1000) ** 0.7  # Scales with length
        
        self.total_cost = material_cost + self.base_cost + injection_cost + cooling_cost
        return self.total_cost
    
    def get_cost_breakdown(self):
        """
        Returns a detailed cost breakdown.
        """
        mass = self.estimate_mass()
        material_cost = mass * self.cost_per_kg_material * \
                       self.machining_factor * self.complexity_factor
        injection_cost = 20000 * (self.engine.At * 10000) ** 0.5
        cooling_cost = 15000 * (self.engine.L_total * 1000) ** 0.7
        
        # Calculate total cost (also assign to self.total_cost for compatibility)
        self.total_cost = material_cost + self.base_cost + injection_cost + cooling_cost
        
        return {
            "Total Mass": f"{mass:.2f} kg",
            "Material + Machining Cost": f"${material_cost:,.0f}",
            "Base Fixed Costs": f"${self.base_cost:,.0f}",
            "Injection System": f"${injection_cost:,.0f}",
            "Cooling System": f"${cooling_cost:,.0f}",
            "Total Cost": f"${self.total_cost:,.0f}"
        }


# ============================================================================
# FUNCTION: Engine Visualization
# ============================================================================
def plot_engine_cross_section(engine: RocketEngine):
    """
    Generates a 2D plot of the engine cross-section.
    
    Shows:
    - Combustion chamber (cylinder)
    - Convergent section
    - Throat
    - Divergent section (nozzle)
    
    Parameters:
    ----------
    engine : RocketEngine
        Engine instance to visualize
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Created matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Scale for visualization (everything in mm)
    scale = 1000
    
    # Scaled geometric parameters
    Dc = engine.Dc * scale
    Dt = engine.Dt * scale
    De = engine.De * scale
    Lc = engine.Lc_cylindrical * scale
    L_conv = engine.L_convergent * scale
    L_div = engine.L_divergent * scale
    
    # Initial x position (reference point)
    x_start = 0
    
    # ========================================================================
    # 1. COMBUSTION CHAMBER (Cylinder)
    # ========================================================================
    chamber = Rectangle((x_start, -Dc/2), Lc, Dc, 
                       linewidth=2, edgecolor='darkblue', facecolor='lightblue', 
                       alpha=0.7, label='Combustion Chamber')
    ax.add_patch(chamber)
    
    # ========================================================================
    # 2. CONVERGENT SECTION (Truncated cone)
    # ========================================================================
    x_conv_start = x_start + Lc
    x_conv_end = x_conv_start + L_conv
    
    conv_points = np.array([
        [x_conv_start, -Dc/2],
        [x_conv_start, Dc/2],
        [x_conv_end, Dt/2],
        [x_conv_end, -Dt/2]
    ])
    convergent = Polygon(conv_points, closed=True,
                        linewidth=2, edgecolor='blue', facecolor='skyblue',
                        alpha=0.7, label='Convergent')
    ax.add_patch(convergent)
    
    # ========================================================================
    # 3. THROAT (reference circle)
    # ========================================================================
    x_throat = x_conv_end
    throat = Circle((x_throat, 0), Dt/2,
                   linewidth=3, edgecolor='red', facecolor='none',
                   label='Throat (Dt)')
    ax.add_patch(throat)
    
    # ========================================================================
    # 4. DIVERGENT SECTION (Nozzle - Truncated cone)
    # ========================================================================
    x_div_start = x_conv_end
    x_div_end = x_div_start + L_div
    
    div_points = np.array([
        [x_div_start, -Dt/2],
        [x_div_start, Dt/2],
        [x_div_end, De/2],
        [x_div_end, -De/2]
    ])
    divergent = Polygon(div_points, closed=True,
                       linewidth=2, edgecolor='green', facecolor='lightgreen',
                       alpha=0.7, label='Divergent (Nozzle)')
    ax.add_patch(divergent)
    
    # ========================================================================
    # Annotations and Labels
    # ========================================================================
    # Center line (axis)
    total_length = Lc + L_conv + L_div
    ax.plot([0, total_length], [0, 0], 'k--', linewidth=1, alpha=0.3, label='Axis')
    
    # Dimension annotations
    ax.annotate(f'Dt = {Dt:.1f} mm', 
               xy=(x_throat, Dt/2 + 5), 
               xytext=(x_throat, Dt/2 + 20),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10, ha='center', color='red', weight='bold')
    
    ax.annotate(f'De = {De:.1f} mm',
               xy=(x_div_end, De/2 + 5),
               xytext=(x_div_end, De/2 + 30),
               arrowprops=dict(arrowstyle='->', color='green'),
               fontsize=10, ha='center', color='green', weight='bold')
    
    ax.annotate(f'Dc = {Dc:.1f} mm',
               xy=(Lc/2, Dc/2 + 5),
               xytext=(Lc/2, Dc/2 + 20),
               arrowprops=dict(arrowstyle='->', color='darkblue'),
               fontsize=10, ha='center', color='darkblue', weight='bold')
    
    # Plot configuration
    ax.set_xlim(-10, total_length + 20)
    ax.set_ylim(-De/2 - 60, De/2 + 60)
    ax.set_aspect('equal')
    ax.set_xlabel('Length [mm]', fontsize=12, weight='bold')
    ax.set_ylabel('Diameter [mm]', fontsize=12, weight='bold')
    ax.set_title('Rocket Engine Cross-Section', fontsize=14, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================
def main():
    """
    Main function for the Streamlit interface.
    """
    st.set_page_config(
        page_title="RocketSizer - Rocket Engine Sizing",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 RocketSizer")
    st.markdown("### Educational Software for Liquid Propellant Rocket Engine Sizing")
    st.markdown("---")
    
    # ========================================================================
    # SIDEBAR: User Input
    # ========================================================================
    st.sidebar.header("⚙️ Input Parameters")
    
    # Desired thrust
    thrust_kN = st.sidebar.slider(
        "Desired Thrust (kN)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="Engine thrust in kilonewtons"
    )
    
    # Chamber pressure
    Pc_bar = st.sidebar.slider(
        "Combustion Chamber Pressure (bar)",
        min_value=10.0,
        max_value=200.0,
        value=50.0,
        step=1.0,
        help="Combustion chamber pressure in bar (1 bar = 100 kPa)"
    )
    
    # Propellant selection
    propellant_options = list(RocketEngine.PROPELLANT_PROPERTIES.keys())
    propellant_name = st.sidebar.selectbox(
        "Propellant",
        options=propellant_options,
        index=0,
        help="Type of propellant to use"
    )
    
    # O/F ratio
    OF_ratio = st.sidebar.slider(
        "Mixture Ratio (O/F)",
        min_value=1.0,
        max_value=20.0,
        value=2.5 if "Hydrogen" in propellant_name else 2.5,
        step=0.1,
        help="Ratio between oxidizer mass and fuel mass"
    )
    
    # Selected propellant information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Propellant Properties")
    props = RocketEngine.PROPELLANT_PROPERTIES[propellant_name]
    st.sidebar.write(f"**Flame Temperature:** {props['Tc']} K")
    st.sidebar.write(f"**Molecular Weight:** {props['MW']} kg/kmol")
    st.sidebar.write(f"**Gamma (γ):** {props['gamma']}")
    st.sidebar.write(f"**Theoretical Isp:** {props['Isp_theoretical']} s")
    
    # ========================================================================
    # CALCULATION AND RESULTS
    # ========================================================================
    try:
        # Create engine instance and calculate
        engine = RocketEngine(thrust_kN, Pc_bar, propellant_name, OF_ratio)
        cost_estimator = CostEstimator(engine)
        
        # Column layout for results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📐 Engine Dimensions")
            
            # Visualization
            fig = plot_engine_cross_section(engine)
            st.pyplot(fig)
        
        with col2:
            st.header("📊 Calculated Parameters")
            
            summary = engine.get_summary()
            for key, value in summary.items():
                st.metric(key, value)
        
        # ====================================================================
        # COST ANALYSIS
        # ====================================================================
        st.markdown("---")
        st.header("💰 Cost Analysis")
        
        cost_breakdown = cost_estimator.get_cost_breakdown()
        
        cost_col1, cost_col2 = st.columns(2)
        
        with cost_col1:
            st.subheader("Cost Breakdown")
            for key, value in cost_breakdown.items():
                if "Total" in key:
                    st.markdown(f"### {key}: {value}")
                else:
                    st.write(f"**{key}:** {value}")
        
        with cost_col2:
            st.subheader("📈 Cost Chart")
            # Extract numeric values for the chart
            labels = []
            values = []
            for key, value in cost_breakdown.items():
                if "Total" not in key and "$" in value:
                    labels.append(key.replace("Cost", "").replace("System", "").strip())
                    # Extract number from string like "$150,000"
                    num_str = value.replace("$", "").replace(",", "")
                    values.append(float(num_str))
            
            if values:
                fig_cost, ax_cost = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
                bars = ax_cost.barh(labels, values, color=colors)
                ax_cost.set_xlabel('Cost [$]', fontsize=11, weight='bold')
                ax_cost.set_title('Cost Breakdown', fontsize=12, weight='bold')
                ax_cost.grid(True, alpha=0.3, axis='x')
                
                # Add values on bars
                for bar in bars:
                    width = bar.get_width()
                    ax_cost.text(width, bar.get_y() + bar.get_height()/2,
                               f'${width:,.0f}',
                               ha='left', va='center', fontsize=9, weight='bold')
                
                plt.tight_layout()
                st.pyplot(fig_cost)
        
        # ====================================================================
        # COST WARNING
        # ====================================================================
        # Make sure cost has been calculated
        if cost_estimator.total_cost is None:
            cost_estimator.get_cost_breakdown()  # Calculate if not already done
        
        total_cost_value = cost_estimator.total_cost
        if total_cost_value > 5000000:
            st.warning(f"⚠️ **Warning:** The estimated cost (${total_cost_value:,.0f}) exceeds the $500,0000 threshold. "
                      "Consider reducing thrust or optimizing design parameters.")
        elif total_cost_value > 200000:
            st.info(f"ℹ️ The estimated cost is ${total_cost_value:,.0f}. "
                   "A project in this price range is typical for research and development engines.")
        
        # ====================================================================
        # TECHNICAL NOTES
        # ====================================================================
        with st.expander("📚 Technical Notes and References"):
            st.markdown("""
            ### Main Formulas Used
            
            1. **Thrust Equation:**
               - F = Isp × g₀ × ṁ
               - Source: Huzel & Huang, Section 3-3
            
            2. **Isentropic Flow (Throat Area):**
               - ṁ = (Pc × At) / √(R×Tc) × √(γ) × (2/(γ+1))^((γ+1)/(2(γ-1)))
               - Source: Huzel & Huang, Section 3-4
            
            3. **Expansion Ratio:**
               - ε = Ae/At = (1/Me) × [(2/(γ+1)) × (1 + (γ-1)/2 × Me²)]^((γ+1)/(2(γ-1)))
               - Source: Anderson, "Fundamentals of Aerodynamics"
            
            4. **L* Parameter:**
               - L* = Vc / At
               - Typical values: 0.8-2.5 m depending on propellant
               - Source: Huzel & Huang, Section 8-3
            
            ### Assumptions and Limitations
            
            - Effective Isp = 92% of theoretical Isp (real losses)
            - Optimal expansion at sea level (Pe = Pa = 1 atm)
            - Simplified cylindrical chamber
            - Simplified conical nozzle (not optimal Rao contour)
            - Cost estimates based on typical values for R&D, not serial production
            
            ### Bibliographic References
            
            - **Huzel & Huang**, "Design of Liquid Propellant Rocket Engines", NASA SP-125, 1971
            - **Sutton & Biblarz**, "Rocket Propulsion Elements", 9th Edition
            - **Anderson**, "Fundamentals of Aerodynamics", 6th Edition
            """)
    
    except Exception as e:
        st.error(f"❌ Calculation error: {str(e)}")
        st.info("Please verify that all input parameters are valid.")


if __name__ == "__main__":
    main()
