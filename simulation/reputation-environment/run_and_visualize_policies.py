"""Script to run policy simulation and then visualize the results.

This demonstrates the complete workflow:
1. Run policy simulation with different configurations
2. Generate visualizations of the results
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def run_policy_simulation():
    """Run the policy simulation to generate results."""
    print("=" * 60)
    print("STEP 1: Running Policy Simulation")
    print("=" * 60)
    
    try:
        # Import and run the policy simulation
        from run_policy_simulation import run_simulation_with_policies, compare_policy_performances
        
        print("Running single simulation with balanced policies...")
        run_simulation_with_policies()
        
        print("\n" + "=" * 60)
        print("Comparing different policy distributions...")
        compare_policy_performances()
        
        print("\nPolicy simulation completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Error importing policy simulation modules: {e}")
        print("Make sure run_policy_simulation.py exists and is properly configured.")
        return False
    except Exception as e:
        print(f"Error running policy simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_policy_results():
    """Visualize the policy simulation results."""
    print("\n" + "=" * 60)
    print("STEP 2: Visualizing Policy Simulation Results")
    print("=" * 60)
    
    try:
        # Import and run the visualization script
        from visualize_policy_simulation import main as run_visualizations
        
        run_visualizations()
        print("\nPolicy visualization completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Error importing visualization modules: {e}")
        print("Make sure visualize_policy_simulation.py exists and all required packages are installed.")
        return False
    except Exception as e:
        print(f"Error running visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_requirements():
    """Check if required packages are installed."""
    print("Checking requirements...")
    
    required_packages = ['matplotlib', 'numpy', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All required packages are installed!")
    return True


def main():
    """Main function to run the complete workflow."""
    print("Policy Simulation and Visualization Workflow")
    print("=" * 60)
    
    # Check requirements first
    if not check_requirements():
        print("\nPlease install missing packages before continuing.")
        return
    
    # Step 1: Run policy simulation
    if run_policy_simulation():
        print("\n✓ Policy simulation completed")
        
        # Step 2: Visualize results
        if visualize_policy_results():
            print("\n✓ Visualization completed")
            print("\n" + "=" * 60)
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\nGenerated files:")
            print("- Policy simulation results in various .json files")
            print("- Visualizations saved to visualizations/ directory")
            print("\nYou can now:")
            print("1. Examine the JSON results files for detailed data")
            print("2. View the generated plots in the visualizations/ directory")
            print("3. Modify the visualization scripts to create custom plots")
        else:
            print("\n✗ Visualization failed")
    else:
        print("\n✗ Policy simulation failed")
        print("Cannot proceed to visualization step.")


if __name__ == "__main__":
    main()
