#!/usr/bin/env python3
"""
Real-time Real EEG 3D Brain System
Uses actual EEG data streaming in real-time with your dendrite neuron model
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import mne
from scipy.signal import hilbert, filtfilt, butter
import time
import threading
import queue
from collections import deque
import argparse
from pathlib import Path

# Enhanced Dendrite class from your code
class Dendrite:
    def __init__(self, length=3, decay=0.8):
        self.segments = np.zeros(length)
        self.decay = decay
        self.history = deque(maxlen=10)  # Track recent activity
    
    def integrate(self, input_current):
        self.segments[0] += input_current
        for i in range(1, len(self.segments)):
            self.segments[i] += self.segments[i-1] * 0.5
        output = np.sum(self.segments)
        self.segments *= self.decay
        self.history.append(output)
        return output

# Enhanced Neuron with real neural dynamics
class RealtimeNeuron:
    def __init__(self, pos, num_dendrites=4, threshold=1.0, decay=0.9):
        self.pos = pos
        self.dendrites = [Dendrite(length=np.random.randint(2, 5), 
                                  decay=0.7 + np.random.rand()*0.2) 
                         for _ in range(num_dendrites)]
        self.potential = 0.0
        self.threshold = threshold + np.random.rand() * 0.5  # Variable thresholds
        self.decay = decay
        self.spiked = False
        self.spike_history = deque(maxlen=50)
        self.refractory = 0
        
    def update(self, field_input, dt=0.01):
        # Refractory period
        if self.refractory > 0:
            self.refractory -= dt
            return 0.0
            
        # Dendritic integration
        dend_inputs = sum(d.integrate(field_input / len(self.dendrites)) 
                         for d in self.dendrites)
        
        # Membrane potential update with leak
        leak = -0.1 * self.potential  # Leak current
        self.potential += dt * (dend_inputs + leak)
        
        # Spike generation
        self.spiked = self.potential >= self.threshold
        
        if self.spiked:
            self.spike_history.append(time.time())
            self.potential = -0.2  # Hyperpolarization
            self.refractory = 0.002  # 2ms refractory
            return 1.0  # Strong feedback
        
        return max(0, self.potential * 0.1)  # Subthreshold feedback

class RealTimeEEGProcessor:
    """Real-time EEG data processor"""
    
    def __init__(self, eeg_file, buffer_size=2.0):
        self.eeg_file = eeg_file
        self.buffer_size = buffer_size  # seconds
        self.raw = None
        self.current_sample = 0
        self.data_buffer = deque(maxlen=int(buffer_size * 250))  # Assume 250Hz
        self.is_streaming = False
        
        # Load EEG file
        self.load_eeg_file()
        
    def load_eeg_file(self):
        """Load real EEG file"""
        print(f"📂 Loading real EEG data from {self.eeg_file}")
        
        try:
            if self.eeg_file.suffix.lower() == '.edf':
                self.raw = mne.io.read_raw_edf(self.eeg_file, preload=True, verbose=False)
            elif self.eeg_file.suffix.lower() == '.fif':
                self.raw = mne.io.read_raw_fif(self.eeg_file, preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported format: {self.eeg_file.suffix}")
            
            # Preprocessing
            self.raw.pick_types(eeg=True, exclude='bads')
            
            # Smart filtering based on sampling rate
            sfreq = self.raw.info['sfreq']
            nyquist = sfreq / 2
            high_freq = min(100, nyquist - 1)  # Stay below Nyquist
            
            print(f"   Sampling rate: {sfreq} Hz, filtering 1-{high_freq} Hz")
            self.raw.filter(1, high_freq, verbose=False)
            
            # Only resample if needed
            if sfreq != 250:
                target_sfreq = min(250, sfreq)  # Don't upsample
                print(f"   Resampling from {sfreq} Hz to {target_sfreq} Hz")
                self.raw.resample(target_sfreq, verbose=False)
            
            # Clean channel names
            new_names = {ch: ch.rstrip('.').upper() for ch in self.raw.ch_names}
            self.raw.rename_channels(new_names)
            
            # Set montage for 3D positions
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                self.raw.set_montage(montage, match_case=False, on_missing='ignore')
                self.ch_xyz = self._get_3d_positions()
            except:
                self.ch_xyz = self._create_mock_positions()
            
            print(f"✅ EEG loaded: {len(self.raw.ch_names)} channels, {self.raw.times[-1]:.1f}s")
            print(f"📡 Channels: {', '.join(self.raw.ch_names[:8])}{'...' if len(self.raw.ch_names) > 8 else ''}")
            
        except Exception as e:
            print(f"❌ Error loading EEG: {e}")
            raise
    
    def _get_3d_positions(self):
        """Get real 3D electrode positions"""
        positions = []
        for ch in self.raw.info['chs']:
            if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0):
                positions.append(ch['loc'][:3])
            else:
                positions.append(self._approximate_position(ch['ch_name']))
        
        positions = np.array(positions)
        # Scale to reasonable brain size
        if positions.shape[0] > 0:
            max_pos = np.max(np.abs(positions))
            if max_pos > 0:
                positions = positions / max_pos * 0.08  # ±8cm
        
        return positions
    
    def _approximate_position(self, ch_name):
        """Approximate positions for missing channels"""
        approx_pos = {
            'FP1': (-0.03, 0.08, 0.04), 'FP2': (0.03, 0.08, 0.04),
            'F3': (-0.04, 0.05, 0.06), 'F4': (0.04, 0.05, 0.06),
            'C3': (-0.04, 0.0, 0.07), 'C4': (0.04, 0.0, 0.07),
            'P3': (-0.04, -0.05, 0.06), 'P4': (0.04, -0.05, 0.06),
            'O1': (-0.03, -0.08, 0.04), 'O2': (0.03, -0.08, 0.04),
            'FZ': (0.0, 0.05, 0.07), 'CZ': (0.0, 0.0, 0.08), 
            'PZ': (0.0, -0.05, 0.07), 'OZ': (0.0, -0.08, 0.05)
        }
        return np.array(approx_pos.get(ch_name.upper(), (0, 0, 0.05)))
    
    def _create_mock_positions(self):
        """Create mock positions if montage fails"""
        n_ch = len(self.raw.ch_names)
        angles = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
        positions = np.column_stack([
            0.06 * np.cos(angles),
            0.06 * np.sin(angles),
            0.04 * np.ones(n_ch)
        ])
        return positions
    
    def get_realtime_eeg_data(self, window_samples=500):
        """Get real-time EEG data window"""
        if self.current_sample + window_samples >= self.raw.n_times:
            self.current_sample = 0  # Loop back to start
            
        # Extract current window
        data = self.raw.get_data(start=self.current_sample, 
                                stop=self.current_sample + window_samples)
        
        self.current_sample += window_samples // 4  # 75% overlap for smooth streaming
        
        return data
    
    def extract_realtime_pac(self, data):
        """Extract phase-amplitude coupling from real-time data"""
        sf = self.raw.info['sfreq']
        
        # Adjust frequency bands based on sampling rate
        nyquist = sf / 2
        
        # Theta (4-8 Hz) for slow phase - always safe
        theta_low, theta_high = 4, min(8, nyquist - 2)
        theta_b, theta_a = butter(4, [theta_low/(sf/2), theta_high/(sf/2)], btype='band')
        theta_filtered = filtfilt(theta_b, theta_a, data, axis=1)
        theta_phase = np.angle(hilbert(theta_filtered, axis=1))
        
        # Gamma (30-50 Hz) for fast amplitude - adjust if needed
        gamma_low = min(30, nyquist - 10)
        gamma_high = min(50, nyquist - 2)
        
        # If sampling rate too low for gamma, use beta instead
        if gamma_low >= gamma_high:
            gamma_low, gamma_high = min(15, nyquist - 10), min(25, nyquist - 2)
            print(f"   Using beta band ({gamma_low}-{gamma_high} Hz) instead of gamma")
        
        gamma_b, gamma_a = butter(4, [gamma_low/(sf/2), gamma_high/(sf/2)], btype='band')
        gamma_filtered = filtfilt(gamma_b, gamma_a, data, axis=1)
        gamma_amplitude = np.abs(hilbert(gamma_filtered, axis=1))
        
        # Take mean across time for spatial projection
        phases = np.mean(theta_phase, axis=1)
        amplitudes = np.mean(gamma_amplitude, axis=1)
        
        return phases, amplitudes

def project_holonomic_field(phases, amps, ch_xyz, grid_pts):
    """Your original holonomic projection method"""
    diff = grid_pts[None] - ch_xyz[:, None]
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    dirv = np.nan_to_num(diff / dist)
    k = 2 * np.pi / 0.06  # 6cm wavelength
    phase = k * (dirv * dist).sum(-1) + phases[:, None]
    field = (amps[:, None] * np.cos(phase)).sum(0)
    return field

class RealTime3DBrainVisualizer:
    """Real-time 3D brain visualization with your neuron model"""
    
    def __init__(self, eeg_processor, grid_size=24):
        self.eeg_processor = eeg_processor
        self.grid_size = grid_size
        
        # Setup 3D grid
        self.setup_3d_grid()
        
        # Create neurons at grid points
        self.setup_neurons()
        
        # Visualization data
        self.field_history = deque(maxlen=10)
        self.neuron_activity_history = deque(maxlen=50)
        self.pac_strength_history = deque(maxlen=100)
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
    def setup_3d_grid(self):
        """Setup 3D grid for neurons and field"""
        extent = 0.08  # ±8cm
        lin = np.linspace(-extent, extent, self.grid_size)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
        
        self.grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        self.grid_shape = (self.grid_size, self.grid_size, self.grid_size)
        
        print(f"🧊 3D Grid: {self.grid_size}³ = {len(self.grid_pts):,} points")
    
    def setup_neurons(self):
        """Create neurons at grid points"""
        print(f"🧠 Creating {len(self.grid_pts):,} neurons with dendrites...")
        
        self.neurons = []
        for i, pos in enumerate(self.grid_pts):
            # Vary neuron parameters based on position
            distance_from_center = np.linalg.norm(pos)
            
            # More dendrites in central regions
            num_dendrites = 3 + int(3 * np.exp(-distance_from_center * 10))
            
            # Variable thresholds
            threshold = 0.8 + 0.4 * np.random.rand()
            
            neuron = RealtimeNeuron(pos, num_dendrites, threshold)
            self.neurons.append(neuron)
            
            if i % 5000 == 0:
                print(f"   Created {i:,}/{len(self.grid_pts):,} neurons...")
        
        print(f"✅ Neural network ready!")
    
    def process_frame(self):
        """Process one frame of real EEG data"""
        # Get real-time EEG data
        eeg_data = self.eeg_processor.get_realtime_eeg_data()
        
        # Extract PAC components
        phases, amplitudes = self.eeg_processor.extract_realtime_pac(eeg_data)
        
        # Project to 3D field
        field_raw = project_holonomic_field(phases, amplitudes, 
                                           self.eeg_processor.ch_xyz, 
                                           self.grid_pts)
        
        # Normalize field
        if field_raw.ptp() > 0:
            field = (field_raw - field_raw.min()) / field_raw.ptp()
        else:
            field = np.zeros_like(field_raw)
        
        # Update neurons with field input
        neuron_outputs = np.zeros(len(self.neurons))
        active_neurons = 0
        
        for i, neuron in enumerate(self.neurons):
            output = neuron.update(field[i])
            neuron_outputs[i] = output
            if neuron.spiked:
                active_neurons += 1
        
        # Ephaptic-like feedback (neurons influence field)
        enhanced_field = field + neuron_outputs * 0.3
        
        # Compute PAC strength
        pac_strength = np.mean(amplitudes * np.abs(np.cos(phases)))
        
        # Store history
        self.field_history.append(enhanced_field)
        self.neuron_activity_history.append(active_neurons / len(self.neurons))
        self.pac_strength_history.append(pac_strength)
        
        # Frame statistics
        self.frame_count += 1
        current_time = time.time()
        fps = self.frame_count / (current_time - self.start_time)
        
        return {
            'field': enhanced_field,
            'neuron_spikes': neuron_outputs,
            'active_neurons': active_neurons,
            'pac_strength': pac_strength,
            'frame_count': self.frame_count,
            'fps': fps,
            'phases': phases,
            'amplitudes': amplitudes
        }
    
    def create_realtime_visualization(self, data):
        """Create real-time 3D visualization"""
        field = data['field']
        neuron_spikes = data['neuron_spikes']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"3D Neural Field (Frame {data['frame_count']})",
                f"Active Neurons: {data['active_neurons']}",
                f"PAC Strength: {data['pac_strength']:.3f}",
                "Real-time Statistics"
            ],
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # 1. 3D Neural Field
        threshold = np.mean(field) + 0.8 * np.std(field)
        active_field = field > threshold
        
        if np.any(active_field):
            active_coords = self.grid_pts[active_field]
            active_values = field[active_field]
            
            # Subsample for performance
            if len(active_coords) > 3000:
                indices = np.random.choice(len(active_coords), 3000, replace=False)
                active_coords = active_coords[indices]
                active_values = active_values[indices]
            
            fig.add_trace(
                go.Scatter3d(
                    x=active_coords[:, 0] * 100,  # Convert to cm
                    y=active_coords[:, 1] * 100,
                    z=active_coords[:, 2] * 100,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=active_values,
                        colorscale='Viridis',
                        opacity=0.7,
                        showscale=False
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Spiking Neurons
        spiking = neuron_spikes > 0.5
        if np.any(spiking):
            spike_coords = self.grid_pts[spiking]
            
            # Subsample spiking neurons
            if len(spike_coords) > 1000:
                indices = np.random.choice(len(spike_coords), 1000, replace=False)
                spike_coords = spike_coords[indices]
            
            fig.add_trace(
                go.Scatter3d(
                    x=spike_coords[:, 0] * 100,
                    y=spike_coords[:, 1] * 100,
                    z=spike_coords[:, 2] * 100,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='red',
                        opacity=0.9,
                        showscale=False
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. PAC Strength over time
        if len(self.pac_strength_history) > 1:
            fig.add_trace(
                go.Scatter(
                    y=list(self.pac_strength_history),
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Neural Activity over time
        if len(self.neuron_activity_history) > 1:
            fig.add_trace(
                go.Scatter(
                    y=list(self.neuron_activity_history),
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Real-time EEG 3D Brain Analysis - {data['fps']:.1f} FPS",
            height=800,
            showlegend=False
        )
        
        # Set 3D scene properties
        scene_props = dict(
            xaxis_title="X (cm)",
            yaxis_title="Y (cm)",
            zaxis_title="Z (cm)",
            aspectmode='cube'
        )
        fig.update_scenes(scene_props)
        
        return fig
    
    def run_realtime_stream(self, duration=60, update_interval=0.5):
        """Run real-time streaming analysis"""
        print(f"🎬 Starting real-time EEG 3D brain analysis...")
        print(f"   Duration: {duration}s, Update interval: {update_interval}s")
        print(f"   Neural network: {len(self.neurons):,} neurons")
        print(f"   EEG channels: {len(self.eeg_processor.raw.ch_names)}")
        
        end_time = time.time() + duration
        frame_count = 0
        
        try:
            while time.time() < end_time:
                frame_count += 1
                frame_start = time.time()
                
                # Process frame
                data = self.process_frame()
                
                print(f"Frame {frame_count:3d}: "
                     f"PAC={data['pac_strength']:.3f}, "
                     f"Active={data['active_neurons']:4d}, "
                     f"FPS={data['fps']:.1f}")
                
                # Create visualization every few frames
                if frame_count % 3 == 0:  # Every 3rd frame
                    try:
                        fig = self.create_realtime_visualization(data)
                        filename = f"realtime_brain_frame_{frame_count:04d}.html"
                        pyo.plot(fig, filename=filename, auto_open=False)
                    except Exception as e:
                        print(f"   Viz error: {e}")
                
                # Control timing
                frame_time = time.time() - frame_start
                sleep_time = max(0, update_interval - frame_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"   ⚠️  Frame took {frame_time:.2f}s (target: {update_interval:.2f}s)")
        
        except KeyboardInterrupt:
            print("\n⏹️  Stream stopped by user")
        
        print(f"✅ Real-time analysis complete: {frame_count} frames processed")
        return frame_count

def main():
    parser = argparse.ArgumentParser(description='Real-time Real EEG 3D Brain System')
    parser.add_argument('eeg_file', help='Path to real EEG file (.edf, .fif)')
    parser.add_argument('--duration', type=float, default=30, help='Stream duration (seconds)')
    parser.add_argument('--update-rate', type=float, default=0.5, help='Update interval (seconds)')
    parser.add_argument('--grid-size', type=int, default=20, help='3D grid size (N³ neurons)')
    
    args = parser.parse_args()
    
    # Validate EEG file
    eeg_path = Path(args.eeg_file)
    if not eeg_path.exists():
        print(f"❌ EEG file not found: {eeg_path}")
        return
    
    print("🧠 REAL-TIME REAL EEG 3D BRAIN SYSTEM")
    print("="*50)
    print(f"EEG file: {eeg_path}")
    print(f"Grid size: {args.grid_size}³ = {args.grid_size**3:,} neurons")
    print(f"Duration: {args.duration}s")
    print(f"Update rate: {args.update_rate}s")
    
    try:
        # Initialize EEG processor
        print(f"\n📡 Initializing real EEG processor...")
        eeg_processor = RealTimeEEGProcessor(eeg_path)
        
        # Initialize 3D visualizer
        print(f"\n🎨 Initializing 3D brain visualizer...")
        visualizer = RealTime3DBrainVisualizer(eeg_processor, args.grid_size)
        
        # Run real-time stream
        print(f"\n🚀 Starting real-time analysis...")
        frame_count = visualizer.run_realtime_stream(args.duration, args.update_rate)
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total frames: {frame_count}")
        print(f"   Average PAC: {np.mean(list(visualizer.pac_strength_history)):.3f}")
        print(f"   Peak neural activity: {max(visualizer.neuron_activity_history):.1%}")
        
        print(f"\n🎯 Check the generated HTML files for 3D visualizations!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()