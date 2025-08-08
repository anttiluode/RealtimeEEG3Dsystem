#!/usr/bin/env python3
"""
Launch Real EEG 3D Brain Analysis
Quick launcher for real-time real EEG analysis
"""

import sys
import argparse
from pathlib import Path
import subprocess

def find_eeg_files():
    """Find EEG files in current directory"""
    current_dir = Path('.')
    eeg_files = []
    
    # Common EEG file extensions
    extensions = ['.edf', '.bdf', '.fif', '.gdf', '.set']
    
    for ext in extensions:
        files = list(current_dir.glob(f'*{ext}'))
        eeg_files.extend(files)
    
    return eeg_files

def main():
    print("🧠 REAL EEG 3D BRAIN ANALYZER")
    print("="*40)
    
    parser = argparse.ArgumentParser(description='Launch real-time real EEG analysis')
    parser.add_argument('--file', help='EEG file path')
    parser.add_argument('--duration', type=int, default=30, help='Analysis duration (seconds)')
    parser.add_argument('--fast', action='store_true', help='Fast mode (smaller grid)')
    parser.add_argument('--list', action='store_true', help='List available EEG files')
    
    args = parser.parse_args()
    
    if args.list:
        eeg_files = find_eeg_files()
        if eeg_files:
            print("📂 Found EEG files:")
            for i, f in enumerate(eeg_files, 1):
                print(f"   {i}. {f}")
        else:
            print("❌ No EEG files found in current directory")
        return
    
    # Find EEG file
    eeg_file = None
    
    if args.file:
        eeg_file = Path(args.file)
        if not eeg_file.exists():
            print(f"❌ File not found: {eeg_file}")
            return
    else:
        # Auto-find EEG files
        eeg_files = find_eeg_files()
        if eeg_files:
            print("📂 Found EEG files:")
            for i, f in enumerate(eeg_files, 1):
                print(f"   {i}. {f}")
            
            if len(eeg_files) == 1:
                eeg_file = eeg_files[0]
                print(f"✅ Using: {eeg_file}")
            else:
                try:
                    choice = input(f"\nSelect file (1-{len(eeg_files)}): ")
                    idx = int(choice) - 1
                    if 0 <= idx < len(eeg_files):
                        eeg_file = eeg_files[idx]
                        print(f"✅ Selected: {eeg_file}")
                    else:
                        print("❌ Invalid selection")
                        return
                except (ValueError, KeyboardInterrupt):
                    print("❌ Invalid input")
                    return
        else:
            print("❌ No EEG files found!")
            print("💡 Place your .edf, .bdf, .fif, or .gdf file in this directory")
            return
    
    # Set parameters
    grid_size = 16 if args.fast else 20
    update_rate = 0.3 if args.fast else 0.5
    
    print(f"\n🚀 LAUNCHING ANALYSIS:")
    print(f"   File: {eeg_file}")
    print(f"   Duration: {args.duration}s")
    print(f"   Grid: {grid_size}³ = {grid_size**3:,} neurons")
    print(f"   Mode: {'Fast' if args.fast else 'Standard'}")
    
    # Build command
    cmd = [
        sys.executable, 
        'realtime_eeg_3d_system.py',
        str(eeg_file),
        '--duration', str(args.duration),
        '--update-rate', str(update_rate),
        '--grid-size', str(grid_size)
    ]
    
    try:
        print(f"\n🎬 Starting real-time analysis...")
        print(f"   Press Ctrl+C to stop")
        print(f"   3D visualizations will be saved as HTML files")
        
        result = subprocess.run(cmd, check=True)
        print(f"✅ Analysis completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
    except KeyboardInterrupt:
        print(f"\n⏹️  Analysis stopped by user")
    except FileNotFoundError:
        print(f"❌ realtime_eeg_3d_system.py not found!")
        print(f"💡 Make sure the script is in the current directory")

def show_help():
    """Show usage examples"""
    print("🚀 USAGE EXAMPLES:")
    print("-"*30)
    print()
    print("1. Auto-detect and select EEG file:")
    print("   python launch_real_eeg.py")
    print()
    print("2. Specify EEG file:")
    print("   python launch_real_eeg.py --file data.edf")
    print()
    print("3. Fast analysis (smaller grid):")
    print("   python launch_real_eeg.py --fast")
    print()
    print("4. Long analysis:")
    print("   python launch_real_eeg.py --duration 60")
    print()
    print("5. List available EEG files:")
    print("   python launch_real_eeg.py --list")
    print()
    print("📊 What you'll see:")
    print("   • Real-time 3D neural field from your EEG")
    print("   • Theta-gamma phase-amplitude coupling")
    print("   • Dendrite neurons spiking in 3D space")
    print("   • Brain-like patterns emerging from real neural data")
    print()
    print("🧠 Expected results:")
    print("   • PAC strength: 0.01-0.1 (higher = better coupling)")
    print("   • Active neurons: 100-1000 spikes per frame")
    print("   • 3D visualizations saved as HTML files")
    print("   • Real brain wave interference patterns!")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_help()
        print()
        main()
    else:
        main()