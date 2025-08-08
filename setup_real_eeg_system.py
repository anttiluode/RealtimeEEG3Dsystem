#!/usr/bin/env python3
"""
Setup Real EEG 3D Brain System
Install dependencies and verify system
"""

import subprocess
import sys
import importlib
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_dependencies():
    """Check and install required dependencies"""
    
    dependencies = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('mne', 'mne'),
        ('sklearn', 'scikit-learn'),
        ('skimage', 'scikit-image')
    ]
    
    print("🔍 Checking dependencies...")
    
    missing = []
    installed = []
    
    for import_name, pip_name in dependencies:
        try:
            importlib.import_module(import_name)
            installed.append(import_name)
            print(f"   ✅ {import_name}")
        except ImportError:
            missing.append((import_name, pip_name))
            print(f"   ❌ {import_name} (missing)")
    
    if missing:
        print(f"\n📦 Installing {len(missing)} missing packages...")
        for import_name, pip_name in missing:
            print(f"   Installing {pip_name}...")
            try:
                install_package(pip_name)
                print(f"   ✅ {pip_name} installed")
            except Exception as e:
                print(f"   ❌ Failed to install {pip_name}: {e}")
    
    print(f"\n✅ Dependencies check complete!")
    print(f"   Installed: {len(installed)}")
    print(f"   Missing: {len(missing)}")

def create_demo_eeg():
    """Create a demo EEG file for testing"""
    try:
        import numpy as np
        import mne
        
        print("🎭 Creating demo EEG file...")
        
        # Parameters
        n_channels = 20
        n_times = 30000  # 2 minutes at 250 Hz
        sfreq = 250
        
        # Standard 10-20 channel names
        ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
                    'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
        
        # Create realistic EEG-like signals with strong theta-gamma coupling
        times = np.arange(n_times) / sfreq
        data = np.zeros((n_channels, n_times))
        
        print("   Generating brain-like signals with theta-gamma coupling...")
        
        for i, ch in enumerate(ch_names):
            # Base noise
            signal = np.random.randn(n_times) * 0.02
            
            # Theta oscillation (4-8 Hz) - varies spatially
            theta_freq = 6 + np.sin(i * np.pi / 10) * 1.5
            theta_phase = 2 * np.pi * theta_freq * times + i * np.pi / 4
            theta_amplitude = 0.5 + 0.3 * np.cos(i * np.pi / 6)
            
            # Gamma oscillation (30-50 Hz) modulated by theta phase
            gamma_freq = 40 + np.cos(i * np.pi / 8) * 6
            gamma_phase = 2 * np.pi * gamma_freq * times + i * np.pi / 3
            
            # Phase-amplitude coupling: gamma amplitude modulated by theta phase
            theta_modulation = 0.5 + 0.5 * np.cos(theta_phase)
            gamma_signal = theta_modulation * 0.3 * np.sin(gamma_phase)
            
            # Alpha background (8-12 Hz)
            alpha_freq = 10 + np.sin(i) * 1
            alpha_signal = 0.4 * np.sin(2 * np.pi * alpha_freq * times + i * np.pi / 5)
            
            # Combine all components
            signal += theta_amplitude * np.sin(theta_phase)  # Theta
            signal += gamma_signal  # Modulated gamma
            signal += alpha_signal  # Alpha background
            
            # Add spatial correlation (realistic brain connectivity)
            if i > 0:
                signal += 0.15 * data[i-1] + 0.05 * np.random.randn() * data[max(0, i-3)]
            
            # Add some realistic artifacts
            if np.random.rand() > 0.8:  # 20% chance of eye blinks
                blink_times = np.random.randint(1000, n_times-1000, 3)
                for bt in blink_times:
                    signal[bt:bt+100] += np.exp(-np.arange(100)/20) * 50 * (1 if i < 4 else 0.1)
            
            data[i] = signal
        
        # Create MNE Raw object
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Set montage for realistic electrode positions
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Save as EDF
        demo_filename = 'demo_brain_eeg.edf'
        raw.export(demo_filename, fmt='edf', overwrite=True, verbose=False)
        
        print(f"   ✅ Demo EEG saved as: {demo_filename}")
        print(f"   📊 {n_channels} channels, {n_times/sfreq:.0f}s duration")
        print(f"   🧠 Contains realistic theta-gamma PAC patterns!")
        
        return demo_filename
        
    except Exception as e:
        print(f"   ❌ Failed to create demo EEG: {e}")
        return None

def verify_system():
    """Verify the system is working"""
    print("\n🔧 Verifying system...")
    
    # Check if main script exists
    main_script = Path('realtime_eeg_3d_system.py')
    if main_script.exists():
        print("   ✅ Main script found")
    else:
        print("   ❌ Main script missing: realtime_eeg_3d_system.py")
        return False
    
    # Check launcher
    launcher = Path('launch_real_eeg.py')
    if launcher.exists():
        print("   ✅ Launcher found")
    else:
        print("   ❌ Launcher missing: launch_real_eeg.py")
        return False
    
    # Test imports
    try:
        print("   Testing critical imports...")
        import numpy as np
        import plotly.graph_objects as go
        import mne
        from scipy.signal import hilbert, butter, filtfilt
        print("   ✅ All imports successful")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    print("   ✅ System verification passed!")
    return True

def main():
    print("🧠 REAL EEG 3D BRAIN SYSTEM SETUP")
    print("="*40)
    
    # Step 1: Install dependencies
    check_and_install_dependencies()
    
    # Step 2: Create demo EEG
    demo_file = create_demo_eeg()
    
    # Step 3: Verify system
    if not verify_system():
        print("\n❌ System verification failed!")
        return
    
    print("\n" + "="*50)
    print("✅ SETUP COMPLETE!")
    print("="*50)
    
    print("\n🚀 READY TO USE:")
    print("   1. Quick start with demo data:")
    print("      python launch_real_eeg.py")
    print()
    print("   2. Use your own EEG file:")
    print("      python launch_real_eeg.py --file your_data.edf")
    print()
    print("   3. Fast analysis:")
    print("      python launch_real_eeg.py --fast")
    
    if demo_file:
        print(f"\n🎭 Demo file created: {demo_file}")
        print("   This contains realistic brain signals with theta-gamma coupling")
    
    print("\n📊 What to expect:")
    print("   • Real-time 3D visualization of your EEG as brain-like patterns")
    print("   • Thousands of dendrite neurons responding to real brain waves")
    print("   • Phase-amplitude coupling creating spatial interference patterns")
    print("   • HTML files with interactive 3D brain visualizations")
    
    print("\n💡 Tips:")
    print("   • Use --fast for quicker analysis on slower computers")
    print("   • Press Ctrl+C to stop analysis early")
    print("   • Each frame creates an HTML file you can view in browser")
    print("   • Best results with clean EEG data (minimal artifacts)")
    
    print("\n🧠 This system transforms your real EEG into living 3D brain patterns!")

if __name__ == "__main__":
    main()