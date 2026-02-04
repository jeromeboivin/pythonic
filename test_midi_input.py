#!/usr/bin/env python3
"""
Simple MIDI input test script to capture all MIDI events including pitch bend.

This script can be used to verify that your MIDI controller (e.g., Arturia Keystep mk2)
is sending pitch bend and other MIDI messages correctly.

Usage:
    python test_midi_input.py              # Auto-detect controller
    python test_midi_input.py "Port Name"  # Use specific port
"""
import mido
import sys

def list_midi_ports():
    """List all available MIDI input ports."""
    print("Available MIDI input ports:")
    ports = mido.get_input_names()
    for i, port in enumerate(ports):
        print(f"  [{i}] {port}")
    return ports

def capture_midi_events(port_name=None):
    """Capture and display all MIDI events."""
    ports = list_midi_ports()
    
    if not ports:
        print("\nNo MIDI input ports found!")
        return
    
    if port_name is None:
        # Try to find Arturia Keystep
        for port in ports:
            if 'keystep' in port.lower() or 'arturia' in port.lower():
                port_name = port
                break
        
        if port_name is None:
            # Use first available port
            port_name = ports[0]
    
    print(f"\nOpening MIDI port: {port_name}")
    print("Listening for MIDI events... (Press Ctrl+C to stop)\n")
    print("Event types to look for:")
    print("  - note_on/note_off: Key presses")
    print("  - pitchwheel: Pitch bend wheel (-8192 to 8191)")
    print("  - control_change: Mod wheel, sustain, etc.")
    print("-" * 60)
    
    try:
        with mido.open_input(port_name) as inport:
            for msg in inport:
                # Format the message nicely
                if msg.type == 'pitchwheel':
                    # Pitch bend range is -8192 to 8191
                    normalized = msg.pitch / 8192.0  # -1.0 to ~1.0
                    print(f"PITCHWHEEL: channel={msg.channel}, pitch={msg.pitch:+6d}, normalized={normalized:+.3f}")
                elif msg.type == 'note_on':
                    print(f"NOTE_ON:    channel={msg.channel}, note={msg.note:3d}, velocity={msg.velocity:3d}")
                elif msg.type == 'note_off':
                    print(f"NOTE_OFF:   channel={msg.channel}, note={msg.note:3d}, velocity={msg.velocity:3d}")
                elif msg.type == 'control_change':
                    cc_names = {
                        1: 'Mod Wheel',
                        7: 'Volume',
                        10: 'Pan',
                        64: 'Sustain',
                        74: 'Filter Cutoff',
                    }
                    cc_name = cc_names.get(msg.control, f'CC{msg.control}')
                    print(f"CC:         channel={msg.channel}, control={msg.control:3d} ({cc_name}), value={msg.value:3d}")
                else:
                    print(f"{msg.type.upper():12s}: {msg}")
    except KeyboardInterrupt:
        print("\n\nStopped listening.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    port_name = sys.argv[1] if len(sys.argv) > 1 else None
    capture_midi_events(port_name)
