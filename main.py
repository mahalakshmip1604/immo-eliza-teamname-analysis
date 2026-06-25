import runpy

if __name__ == "__main__":
    # Runs the file directly from top to bottom
    runpy.run_path("analysis/epc_combined_efficiency_price_per_sqm.py")
    runpy.run_path("analysis/epc_individual_efficiency_price_per_sqm.py")
