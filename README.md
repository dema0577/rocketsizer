# 🚀 RocketSizer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Educational software for sizing liquid propellant rocket engines and estimating production costs.

## 📋 Description

RocketSizer is an interactive Python application built with Streamlit that allows you to:
- **Size** a liquid propellant rocket engine based on input parameters
- **Visualize** the engine geometry (chamber, convergent, throat, divergent)
- **Estimate** production costs based on mass and complexity

The implemented formulas are based on classical space propulsion texts (Huzel & Huang, Sutton & Biblarz, Anderson).

## ✨ Features

- 🔬 **Physics-Based Calculations**: Implements fundamental rocket propulsion equations
- 📊 **Interactive Interface**: User-friendly Streamlit web interface
- 📐 **Visualization**: 2D cross-section plot of engine geometry
- 💰 **Cost Estimation**: Simplified cost model for R&D projects
- 🎓 **Educational**: Detailed comments and references for learning

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rocket-sizer.git
cd rocket-sizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Usage

Launch the application with:
```bash
streamlit run app.py
```

The interface will automatically open in your browser (typically at `http://localhost:8501`).

## 📊 Input Parameters

The application requires the following input parameters:

- **Desired Thrust**: Engine thrust in kilonewtons (1-1000 kN)
- **Combustion Chamber Pressure**: Pressure in bar (10-200 bar)
- **Propellant**: Choice among different combinations:
  - LOX/Kerosene (RP-1)
  - LOX/Liquid Hydrogen
  - LOX/Methane
  - N2O4/MMH
- **Mixture Ratio (O/F)**: Ratio between oxidizer mass and fuel mass

## 📐 Calculations Performed

The software automatically calculates:

- **Mass Flow Rates**: Total, oxidizer, and fuel mass flow rates
- **Throat Dimensions**: Area (At) and diameter (Dt)
- **Exit Dimensions**: Area (Ae) and diameter (De)
- **Expansion Ratio** (ε)
- **Chamber Dimensions**: Volume and dimensions based on L* parameter
- **Characteristic Length** (L*)
- **Cost Estimates**: Mass and production cost estimates

## 🏗️ Project Structure

```
rocket-sizer/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── LICENSE            # MIT License
└── .gitignore         # Git ignore file
```

## 📚 Bibliographic References

- **Huzel & Huang**, "Design of Liquid Propellant Rocket Engines", NASA SP-125, 1971
- **Sutton & Biblarz**, "Rocket Propulsion Elements", 9th Edition
- **Anderson**, "Fundamentals of Aerodynamics", 6th Edition

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This is **educational software** for educational purposes. The cost estimates are simplified and based on typical values for research and development projects. This software is not intended for actual rocket engine design or production purposes.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@dema0577]

## 🙏 Acknowledgments

- NASA for the technical documentation and references
- The open-source community for the amazing tools and libraries

---

⭐ If you find this project helpful, please give it a star!
