import runpy
import sys
from pathlib import Path

def ask_user(question: str) -> bool:
    """Ask user a yes/no question and return their response."""
    response = input(f"\n{question} (yes/no): ").strip().lower()
    return response in ["yes", "y"]

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    
    # Ask user about data cleaning
    if ask_user("Do you want to run data cleaning?"):
        cleaning_scripts = [
            script_dir / "data" / "cleaned" / "clean_sale_data.py",
            script_dir / "data" / "cleaned" / "clean_rent_data.py",
        ]
        
        print("\n" + "="*60)
        print("RUNNING DATA CLEANING PIPELINE")
        print("="*60)
        
        for script in cleaning_scripts:
            if not script.exists():
                print(f"❌ Script not found: {script}")
                sys.exit(1)
            try:
                print(f"\n▶️  Running {script.name}...")
                runpy.run_path(str(script))
                print(f"✅ {script.name} completed")
            except Exception as e:
                print(f"❌ {script.name} failed: {e}")
                sys.exit(1)
    
    # Run EPC analysis scripts
    analysis_scripts = [
        script_dir / "analysis" / "epc_combined_efficiency_price_per_sqm.py",
        script_dir / "analysis" / "epc_individual_efficiency_price_per_sqm.py",
        script_dir / "analysis" / "alex_analysis.py",
    ]
    
    print("\n" + "="*60)
    print("RUNNING EPC ANALYSIS PIPELINE")
    print("="*60)
    
    for script in analysis_scripts:
        if not script.exists():
            print(f"❌ Script not found: {script}")
            sys.exit(1)
        try:
            print(f"\n▶️  Running {script.name}...")
            runpy.run_path(str(script))
            print(f"✅ {script.name} completed")
        except Exception as e:
            print(f"❌ {script.name} failed: {e}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ ALL PIPELINES COMPLETED SUCCESSFULLY")
    print("="*60)