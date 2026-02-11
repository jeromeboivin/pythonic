"""
Custom Widgets for Pythonic GUI
Knobs, faders, and buttons styled like the real application

Style features:
- Hint window shows parameter name and value when dragging
- Shift key for fine adjustments
- Ctrl/Cmd click to reset to default
- Alt click for linear vs circular mode
"""

import tkinter as tk
from tkinter import ttk
import math
import numpy as np


class HintWindow:
    """
    Floating hint window that displays parameter name and value
    """
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls, root):
        if cls._instance is None or not cls._instance.window.winfo_exists():
            cls._instance = cls(root)
        return cls._instance
    
    def __init__(self, root):
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)  # No window decorations
        self.window.attributes('-topmost', True)
        self.window.withdraw()  # Start hidden
        
        self.frame = tk.Frame(self.window, bg='#222233', bd=1, relief='solid')
        self.frame.pack()
        
        self.name_label = tk.Label(self.frame, text="", 
                                   font=('Segoe UI', 8), 
                                   fg='#aaaacc', bg='#222233')
        self.name_label.pack(padx=5, pady=(3, 0))
        
        self.value_label = tk.Label(self.frame, text="", 
                                    font=('Segoe UI', 10, 'bold'), 
                                    fg='#ffffff', bg='#222233')
        self.value_label.pack(padx=5, pady=(0, 3))
    
    def show(self, name, value, x, y):
        """Show hint at position with given name and value"""
        self.name_label.config(text=name)
        self.value_label.config(text=value)
        
        # Position above the cursor
        self.window.geometry(f"+{x-30}+{y-50}")
        self.window.deiconify()
    
    def hide(self):
        """Hide the hint window"""
        self.window.withdraw()


class RotaryKnob(tk.Canvas):
    """
    Rotary knob widget (metallic knobs)
    
    Features:
    - Hint window shows parameter name and value when dragging
    - Shift key for fine adjustments (1/10th sensitivity)
    - Ctrl/Cmd click to reset to default value
    - Alt click to toggle between circular and linear drag mode
    - Mouse wheel for adjustment
    - Double-click to reset to center
    - Logarithmic scaling for frequency/time parameters
    """
    
    # Logarithmic scaling types
    LOG_FREQUENCY = 'frequency'  # For frequency knobs (20-20000 Hz)
    LOG_TIME = 'time'            # For attack/decay (0-10000 ms)
    LOG_RATE = 'rate'            # For rate knobs (1-2000 Hz)
    LOG_Q = 'q'                  # For filter Q (0.5-20)
    
    def __init__(self, parent, size=50, min_val=0, max_val=100, default=50,
                 label="", unit="", logarithmic=False, command=None, command_end=None, **kwargs):
        super().__init__(parent, width=size, height=size + 20,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default  # Store default for reset
        self.label = label
        self.unit = unit  # Unit string for display (Hz, dB, %, etc.)
        self.logarithmic = logarithmic  # Can be True, False, or a LOG_* type string
        self.command = command
        self.command_end = command_end  # Called on mouse release after drag
        
        # Initialize value (convert default if logarithmic)
        self.value = default
        
        # Knob appearance
        self.knob_color = '#8888aa'
        self.knob_highlight = '#aaaacc'
        self.indicator_color = '#4466ff'
        self.bg_color = '#3a3a4a'
        
        # Interaction state
        self.dragging = False
        self.drag_start_y = 0
        self.drag_start_x = 0  # For circular mode
        self.drag_start_value = 0
        self.linear_mode = True  # Default to linear (vertical) drag
        
        # Draw initial state
        self._draw()
        
        # Bind events
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Double-Button-1>', self._on_double_click)
        self.bind('<MouseWheel>', self._on_scroll)
        self.bind('<Button-4>', self._on_scroll_linux)  # Linux scroll up
        self.bind('<Button-5>', self._on_scroll_linux)  # Linux scroll down
        self.bind('<Control-Button-1>', self._on_ctrl_click)  # Ctrl+click to reset
    
    def _get_formatted_value(self):
        """Get value formatted with appropriate unit"""
        val = self.value
        
        # Auto-detect unit from label if not specified
        unit = self.unit
        if not unit:
            label_lower = self.label.lower()
            if 'freq' in label_lower or 'rate' in label_lower:
                unit = 'Hz'
            elif 'gain' in label_lower or 'level' in label_lower:
                unit = 'dB'
            elif 'pan' in label_lower:
                unit = ''
                if val > 0:
                    return f"R{abs(val):.0f}"
                elif val < 0:
                    return f"L{abs(val):.0f}"
                else:
                    return "C"
            elif 'attack' in label_lower or 'decay' in label_lower:
                unit = 'ms'
        
        # Format based on range and value
        if abs(val) >= 1000:
            return f"{val/1000:.2f}k{unit}"
        elif abs(val) >= 100:
            return f"{val:.0f}{unit}"
        elif abs(val) >= 10:
            return f"{val:.1f}{unit}"
        else:
            return f"{val:.2f}{unit}"
    
    def _get_log_floor(self):
        """Get the minimum value for logarithmic calculations.
        Uses a smart floor based on the parameter type."""
        if self.min_val > 0:
            return self.min_val
        
        # Smart floor based on label/type
        label_lower = self.label.lower() if self.label else ""
        if 'attack' in label_lower or 'decay' in label_lower:
            return 0.1  # 0.1ms for time parameters
        elif 'freq' in label_lower or 'rate' in label_lower:
            return 1.0  # 1Hz for frequency parameters
        else:
            return 0.001  # Default floor
    
    def _value_to_normalized(self, value):
        """Convert actual value to normalized 0-1 position (for display)"""
        if not self.logarithmic:
            # Linear scaling
            value_range = self.max_val - self.min_val
            if value_range != 0:
                return (value - self.min_val) / value_range
            return 0.5
        else:
            # Logarithmic scaling
            min_val = self._get_log_floor()
            max_val = self.max_val
            value = max(value, min_val)
            
            # Use log scale
            return np.log(value / min_val) / np.log(max_val / min_val)
    
    def _normalized_to_value(self, normalized):
        """Convert normalized 0-1 position to actual value"""
        normalized = max(0.0, min(1.0, normalized))
        
        if not self.logarithmic:
            # Linear scaling
            return self.min_val + normalized * (self.max_val - self.min_val)
        else:
            # Logarithmic scaling
            min_val = self._get_log_floor()
            max_val = self.max_val
            
            # Exponential mapping: output = min * (max/min)^normalized
            return min_val * np.power(max_val / min_val, normalized)
    
    def _draw(self):
        """Draw the knob"""
        self.delete('all')
        
        cx = self.size // 2
        cy = self.size // 2
        radius = self.size // 2 - 4
        
        # Draw outer ring (arc showing range)
        self.create_arc(4, 4, self.size - 4, self.size - 4,
                       start=225, extent=-270, style='arc',
                       outline='#555566', width=2)
        
        # Draw knob body (gradient effect with ovals)
        for i in range(3):
            offset = i * 2
            shade = int(0x88 + i * 0x11)
            color = f'#{shade:02x}{shade:02x}{shade + 0x22:02x}'
            self.create_oval(4 + offset, 4 + offset,
                           self.size - 4 - offset, self.size - 4 - offset,
                           fill=color, outline='')
        
        # Calculate indicator angle using normalized position
        # Range is 225° (bottom-left) to -45° (bottom-right) = 270° sweep
        normalized = self._value_to_normalized(self.value)
        angle = math.radians(225 - normalized * 270)
        
        # Draw indicator line
        indicator_length = radius - 6
        ix = cx + math.cos(angle) * indicator_length
        iy = cy - math.sin(angle) * indicator_length
        
        self.create_line(cx, cy, ix, iy,
                        fill=self.indicator_color, width=3, capstyle='round')
        
        # Draw center dot
        self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                        fill='#666688', outline='')
        
        # Draw label
        if self.label:
            self.create_text(cx, self.size + 10,
                           text=self.label, fill='#aaaacc',
                           font=('Segoe UI', 7))
    
    def _on_click(self, event):
        """Start dragging - Alt toggles linear/circular mode"""
        # Check for Alt key (toggle drag mode)
        if event.state & 0x20000:  # Alt key on Linux
            self.linear_mode = not self.linear_mode
        
        self.dragging = True
        self.drag_start_y = event.y
        self.drag_start_x = event.x
        self.drag_start_value = self.value
        
        # Capture pre-drag state for undo
        if self.command_end:
            self.command_end('start')
        
        # Show hint window
        self._show_hint()
    
    def _on_ctrl_click(self, event):
        """Ctrl+click resets to default value"""
        self.set_value(self.default_val)
        return "break"  # Prevent normal click handling
    
    def _on_drag(self, event):
        """Handle drag to change value - Shift for fine control"""
        if self.dragging:
            # Check for Shift key (fine adjustment)
            fine_mode = event.state & 0x1  # Shift key
            sensitivity_multiplier = 0.1 if fine_mode else 1.0
            
            if self.linear_mode:
                # Linear/vertical drag - work in normalized space for log scaling
                delta_y = self.drag_start_y - event.y
                
                # Get start position in normalized space
                start_normalized = self._value_to_normalized(self.drag_start_value)
                
                # Sensitivity: 200 pixels = full normalized range (0-1)
                delta_normalized = (delta_y / 200.0) * sensitivity_multiplier
                new_normalized = start_normalized + delta_normalized
                new_normalized = max(0.0, min(1.0, new_normalized))
                
                # Convert back to actual value
                new_value = self._normalized_to_value(new_normalized)
            else:
                # Circular drag mode
                cx, cy = self.size // 2, self.size // 2
                
                # Calculate angle from center
                dx = event.x - cx
                dy = cy - event.y  # Inverted for screen coords
                angle = math.atan2(dy, dx)
                
                # Convert angle to normalized 0-1 range
                angle_deg = math.degrees(angle)
                if angle_deg < -45:
                    angle_deg += 360
                normalized = (225 - angle_deg) / 270
                normalized = max(0, min(1, normalized))
                
                # Convert to actual value (respects log scaling)
                new_value = self._normalized_to_value(normalized)
            
            self.set_value(new_value)
            self._show_hint()
    
    def _on_release(self, event):
        """Stop dragging"""
        was_dragging = self.dragging
        self.dragging = False
        # Hide hint window
        try:
            hint = HintWindow.get_instance(self.winfo_toplevel())
            hint.hide()
        except:
            pass
        # Notify end of interaction — commit undo snapshot
        if was_dragging and self.command_end:
            self.command_end('end')
    
    def _show_hint(self):
        """Show the hint window with parameter name and value"""
        try:
            hint = HintWindow.get_instance(self.winfo_toplevel())
            # Get screen position
            x = self.winfo_rootx() + self.size // 2
            y = self.winfo_rooty()
            hint.show(self.label, self._get_formatted_value(), x, y)
        except:
            pass
    
    def _on_double_click(self, event):
        """Reset to default on double-click"""
        self.set_value(self.default_val)
    
    def _on_scroll(self, event):
        """Handle mouse wheel (Windows/Mac)"""
        delta = event.delta / 120  # Windows scroll delta
        # Work in normalized space for proper log scaling
        current_normalized = self._value_to_normalized(self.value)
        new_normalized = current_normalized + delta / 100.0
        new_normalized = max(0.0, min(1.0, new_normalized))
        self.set_value(self._normalized_to_value(new_normalized))
    
    def _on_scroll_linux(self, event):
        """Handle mouse wheel on Linux"""
        delta = 1 if event.num == 4 else -1
        # Work in normalized space for proper log scaling
        current_normalized = self._value_to_normalized(self.value)
        new_normalized = current_normalized + delta / 100.0
        new_normalized = max(0.0, min(1.0, new_normalized))
        self.set_value(self._normalized_to_value(new_normalized))
    
    def set_value(self, value):
        """Set knob value"""
        self.value = max(self.min_val, min(self.max_val, value))
        self._draw()
        
        if self.command:
            self.command(self.value)
    
    def get_value(self):
        """Get current value"""
        return self.value


class VerticalSlider(tk.Canvas):
    """
    Vertical slider/fader.
    
    Features:
    - Hint window shows value when dragging
    - Shift key for fine adjustments
    - Ctrl+click to reset to default
    - Logarithmic scaling for time parameters
    """
    
    def __init__(self, parent, width=30, height=100, min_val=0, max_val=100,
                 default=50, label="", unit="", logarithmic=False, command=None, command_end=None, **kwargs):
        super().__init__(parent, width=width, height=height + 20,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.slider_width = width
        self.slider_height = height
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default  # Store for reset
        self.value = default
        self.label = label
        self.unit = unit
        self.logarithmic = logarithmic  # Support logarithmic scaling
        self.command = command
        self.command_end = command_end  # Called on mouse release after drag
        
        # Track dimensions
        self.track_x = width // 2
        self.track_top = 10
        self.track_bottom = height - 10
        self.track_range = self.track_bottom - self.track_top
        
        # Handle dimensions
        self.handle_width = width - 8
        self.handle_height = 20
        
        # Interaction
        self.dragging = False
        self.drag_start_y = 0
        self.drag_start_value = 0
        
        self._draw()
        
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<MouseWheel>', self._on_scroll)
        self.bind('<Button-4>', self._on_scroll_linux)
        self.bind('<Button-5>', self._on_scroll_linux)
        self.bind('<Control-Button-1>', self._on_ctrl_click)
        self.bind('<Double-Button-1>', self._on_double_click)
    
    def _get_formatted_value(self):
        """Format value with unit"""
        val = self.value
        unit = self.unit or '%'
        if abs(val) >= 100:
            return f"{val:.0f}{unit}"
        elif abs(val) >= 10:
            return f"{val:.1f}{unit}"
        else:
            return f"{val:.2f}{unit}"
    
    def _get_log_floor(self):
        """Get the minimum value for logarithmic calculations.
        Uses a smart floor based on the parameter type."""
        if self.min_val > 0:
            return self.min_val
        
        # Smart floor based on label/type
        label_lower = self.label.lower() if self.label else ""
        if 'attack' in label_lower or 'decay' in label_lower:
            return 0.1  # 0.1ms for time parameters
        elif 'freq' in label_lower or 'rate' in label_lower:
            return 1.0  # 1Hz for frequency parameters
        else:
            return 0.1  # Default floor for sliders
    
    def _value_to_normalized(self, value):
        """Convert actual value to normalized 0-1 position (for display)"""
        if not self.logarithmic:
            # Linear scaling
            value_range = self.max_val - self.min_val
            if value_range != 0:
                return (value - self.min_val) / value_range
            return 0.5
        else:
            # Logarithmic scaling
            min_val = self._get_log_floor()
            max_val = self.max_val
            value = max(value, min_val)
            
            return np.log(value / min_val) / np.log(max_val / min_val)
    
    def _normalized_to_value(self, normalized):
        """Convert normalized 0-1 position to actual value"""
        normalized = max(0.0, min(1.0, normalized))
        
        if not self.logarithmic:
            # Linear scaling
            return self.min_val + normalized * (self.max_val - self.min_val)
        else:
            # Logarithmic scaling
            min_val = self._get_log_floor()
            max_val = self.max_val
            
            return min_val * np.power(max_val / min_val, normalized)
    
    def _draw(self):
        """Draw the slider"""
        self.delete('all')
        
        # Draw track
        self.create_rectangle(
            self.track_x - 3, self.track_top,
            self.track_x + 3, self.track_bottom,
            fill='#222233', outline='#555566'
        )
        
        # Calculate handle position using normalized value
        normalized = self._value_to_normalized(self.value)
        handle_y = self.track_bottom - normalized * self.track_range
        
        # Draw handle
        hx1 = self.track_x - self.handle_width // 2
        hx2 = self.track_x + self.handle_width // 2
        hy1 = handle_y - self.handle_height // 2
        hy2 = handle_y + self.handle_height // 2
        
        # Handle body
        self.create_rectangle(hx1, hy1, hx2, hy2,
                            fill='#7788aa', outline='#99aacc')
        
        # Handle grip lines
        for i in range(3):
            ly = handle_y - 4 + i * 4
            self.create_line(hx1 + 4, ly, hx2 - 4, ly,
                           fill='#556688')
        
        # Draw label
        if self.label:
            self.create_text(self.slider_width // 2, self.slider_height + 10,
                           text=self.label, fill='#aaaacc',
                           font=('Segoe UI', 7))
    
    def _on_click(self, event):
        """Handle click - start dragging"""
        self.dragging = True
        self.drag_start_y = event.y
        self.drag_start_value = self.value
        # Capture pre-drag state for undo
        if self.command_end:
            self.command_end('start')
        self._show_hint()
    
    def _on_ctrl_click(self, event):
        """Ctrl+click resets to default value"""
        self.set_value(self.default_val)
        return "break"
    
    def _on_double_click(self, event):
        """Double-click resets to default value"""
        self.set_value(self.default_val)
    
    def _on_drag(self, event):
        """Handle drag - Shift for fine control"""
        if self.dragging:
            # Check for Shift key (fine adjustment)
            fine_mode = event.state & 0x1
            sensitivity_multiplier = 0.1 if fine_mode else 1.0
            
            # Work in normalized space for proper log scaling
            delta_y = self.drag_start_y - event.y
            start_normalized = self._value_to_normalized(self.drag_start_value)
            
            # Sensitivity: full track range = full normalized range
            delta_normalized = (delta_y / self.track_range) * sensitivity_multiplier
            new_normalized = start_normalized + delta_normalized
            new_normalized = max(0.0, min(1.0, new_normalized))
            
            new_value = self._normalized_to_value(new_normalized)
            self.set_value(new_value)
            self._show_hint()
    
    def _on_release(self, event):
        """Handle release"""
        was_dragging = self.dragging
        self.dragging = False
        try:
            hint = HintWindow.get_instance(self.winfo_toplevel())
            hint.hide()
        except:
            pass
        # Notify end of interaction — commit undo snapshot
        if was_dragging and self.command_end:
            self.command_end('end')

    def _show_hint(self):
        """Show hint window with current value"""
        try:
            hint = HintWindow.get_instance(self.winfo_toplevel())
            x = self.winfo_rootx() + self.slider_width // 2
            y = self.winfo_rooty()
            hint.show(self.label, self._get_formatted_value(), x, y)
        except:
            pass
    
    def _on_scroll(self, event):
        """Handle scroll (Windows/Mac)"""
        delta = event.delta / 120
        # Work in normalized space for proper log scaling
        current_normalized = self._value_to_normalized(self.value)
        new_normalized = current_normalized + delta / 50.0
        new_normalized = max(0.0, min(1.0, new_normalized))
        self.set_value(self._normalized_to_value(new_normalized))
    
    def _on_scroll_linux(self, event):
        """Handle scroll on Linux"""
        delta = 1 if event.num == 4 else -1
        # Work in normalized space for proper log scaling
        current_normalized = self._value_to_normalized(self.value)
        new_normalized = current_normalized + delta / 50.0
        new_normalized = max(0.0, min(1.0, new_normalized))
        self.set_value(self._normalized_to_value(new_normalized))
    
    def _update_from_mouse(self, y):
        """Update value from mouse position"""
        # Clamp y to track range
        y = max(self.track_top, min(self.track_bottom, y))
        
        # Convert to normalized (inverted: top = max)
        normalized = 1.0 - (y - self.track_top) / self.track_range
        
        # Convert to actual value (respects log scaling)
        value = self._normalized_to_value(normalized)
        
        self.set_value(value)
    
    def set_value(self, value):
        """Set slider value"""
        self.value = max(self.min_val, min(self.max_val, value))
        self._draw()
        
        if self.command:
            self.command(self.value)
    
    def get_value(self):
        """Get current value"""
        return self.value


class CircularButton(tk.Canvas):
    """
    Circular button
    """
    
    def __init__(self, parent, text="", size=35, command=None, 
                 bg_color='#4a4a5a', fg_color='#ccccee', **kwargs):
        super().__init__(parent, width=size, height=size,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.size = size
        self.text = text
        self.command = command
        self.btn_bg_color = bg_color
        self.btn_fg_color = fg_color
        self.active = False
        
        self._draw()
        
        self.bind('<Button-1>', self._on_click)
    
    def _draw(self):
        """Draw circular button"""
        self.delete('all')
        
        # Outer ring
        self.create_oval(2, 2, self.size - 2, self.size - 2,
                        fill=self.btn_bg_color, outline='#666688', width=2)
        
        # Inner highlight
        self.create_oval(4, 4, self.size - 4, self.size - 4,
                        fill='', outline='#888899', width=1)
        
        # Text/symbol
        self.create_text(self.size // 2, self.size // 2,
                        text=self.text,
                        fill=self.btn_fg_color, 
                        font=('Segoe UI', 12, 'bold'))
        
        # Glow if active
        if self.active:
            self.create_oval(4, 4, self.size - 4, self.size - 4,
                            fill='', outline='#44ff88', width=2)
    
    def _on_click(self, event):
        if self.command:
            self.command()
    
    def set_active(self, active):
        """Set active state (glowing)"""
        self.active = active
        self._draw()
    
    def set_bg_color(self, color):
        """Update background color"""
        self.btn_bg_color = color
        self._draw()


class ChannelButton(tk.Canvas):
    """
    Channel selection button with LED indicator
    """
    
    def __init__(self, parent, channel_num, size=30, command=None, **kwargs):
        super().__init__(parent, width=size, height=size,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.size = size
        self.channel_num = channel_num
        self.selected = False
        self.triggered = False
        self.muted = False
        self.command = command
        
        self._draw()
        
        self.bind('<Button-1>', self._on_click)
    
    def _draw(self):
        """Draw the button"""
        self.delete('all')
        
        # Button background
        if self.selected:
            bg_color = '#5566aa'
        else:
            bg_color = '#4a4a5a'
        
        self.create_rectangle(2, 2, self.size - 2, self.size - 2,
                            fill=bg_color, outline='#666688')
        
        # Channel number
        self.create_text(self.size // 2, self.size // 2,
                        text=str(self.channel_num + 1),
                        fill='#ccccee', font=('Segoe UI', 10, 'bold'))
        
        # LED indicator
        if self.triggered:
            led_color = '#44ff44'
        elif self.muted:
            led_color = '#ff4444'
        elif self.selected:
            led_color = '#4488ff'
        else:
            led_color = '#333344'
        
        self.create_oval(self.size - 10, 2, self.size - 2, 10,
                        fill=led_color, outline='')
    
    def _on_click(self, event):
        """Handle click"""
        if self.command:
            self.command(self.channel_num, event)
    
    def set_selected(self, selected):
        """Set selected state"""
        self.selected = selected
        self._draw()
    
    def set_triggered(self, triggered):
        """Set triggered state (flash)"""
        self.triggered = triggered
        self._draw()
    
    def set_muted(self, muted):
        """Set muted state"""
        self.muted = muted
        self._draw()


class WaveformSelector(tk.Canvas):
    """
    Waveform selector with visual icons
    """
    
    def __init__(self, parent, command=None, **kwargs):
        super().__init__(parent, width=80, height=30,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.selected = 0  # 0=sine, 1=triangle, 2=sawtooth
        self.command = command
        
        self._draw()
        
        self.bind('<Button-1>', self._on_click)
    
    def _draw(self):
        """Draw waveform selector"""
        self.delete('all')
        
        # Draw three waveform icons
        icons = ['sine', 'triangle', 'sawtooth']
        
        for i, waveform in enumerate(icons):
            x = 13 + i * 26
            y = 15
            
            # Selection highlight
            if i == self.selected:
                self.create_rectangle(x - 10, y - 12, x + 10, y + 12,
                                     fill='#5566aa', outline='')
            
            # Draw waveform icon
            if waveform == 'sine':
                # Sine wave
                points = []
                for j in range(16):
                    px = x - 7 + j * 0.9
                    py = y - 6 * math.sin(j * math.pi / 8)
                    points.extend([px, py])
                self.create_line(points, fill='#aaccff', width=2, smooth=True)
                
            elif waveform == 'triangle':
                # Triangle wave
                self.create_line(x - 7, y, x - 2, y - 8, x + 3, y + 8, x + 7, y,
                               fill='#aaccff', width=2)
                
            elif waveform == 'sawtooth':
                # Sawtooth wave
                self.create_line(x - 7, y + 6, x, y - 6, x, y + 6, x + 7, y - 6,
                               fill='#aaccff', width=2)
    
    def _on_click(self, event):
        """Handle click"""
        # Determine which waveform was clicked
        idx = (event.x - 3) // 26
        idx = max(0, min(2, idx))
        
        self.selected = idx
        self._draw()
        
        if self.command:
            self.command(idx)
    
    def set_value(self, value):
        """Set selected waveform"""
        self.selected = max(0, min(2, value))
        self._draw()
    
    def get_value(self):
        """Get selected waveform"""
        return self.selected


class ModeSelector(tk.Canvas):
    """
    Three-way mode selector (like pitch mod mode)
    """
    
    def __init__(self, parent, options=None, command=None, width=90, **kwargs):
        super().__init__(parent, width=width, height=22,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.options = options or ['A', 'B', 'C']
        self.selected = 0
        self.command = command
        self.btn_width = width // len(self.options)
        
        self._draw()
        
        self.bind('<Button-1>', self._on_click)
    
    def _draw(self):
        """Draw mode selector"""
        self.delete('all')
        
        for i, opt in enumerate(self.options):
            x1 = i * self.btn_width
            x2 = x1 + self.btn_width
            
            # Background
            if i == self.selected:
                self.create_rectangle(x1 + 1, 1, x2 - 1, 20,
                                     fill='#5566aa', outline='#7788cc')
            else:
                self.create_rectangle(x1 + 1, 1, x2 - 1, 20,
                                     fill='#4a4a5a', outline='#666688')
            
            # Label
            self.create_text((x1 + x2) // 2, 11,
                           text=opt, fill='#ccccee',
                           font=('Segoe UI', 7))
    
    def _on_click(self, event):
        """Handle click"""
        idx = event.x // self.btn_width
        idx = max(0, min(len(self.options) - 1, idx))
        
        self.selected = idx
        self._draw()
        
        if self.command:
            self.command(idx)
    
    def set_value(self, value):
        """Set selected option"""
        self.selected = max(0, min(len(self.options) - 1, value))
        self._draw()
    
    def get_value(self):
        """Get selected option"""
        return self.selected


class ToggleButton(tk.Canvas):
    """
    Toggle button with LED indicator
    """
    
    def __init__(self, parent, text="", width=50, height=25,
                 command=None, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.text = text
        self.btn_width = width
        self.btn_height = height
        self.enabled = False
        self.command = command
        
        self._draw()
        
        self.bind('<Button-1>', self._on_click)
    
    def _draw(self):
        """Draw toggle button"""
        self.delete('all')
        
        # Button background
        if self.enabled:
            bg_color = '#5566aa'
            led_color = '#44ff88'
        else:
            bg_color = '#4a4a5a'
            led_color = '#333344'
        
        self.create_rectangle(2, 2, self.btn_width - 2, self.btn_height - 2,
                            fill=bg_color, outline='#666688')
        
        # LED
        self.create_oval(self.btn_width - 15, 5, self.btn_width - 5, 15,
                        fill=led_color, outline='')
        
        # Text
        self.create_text(self.btn_width // 2 - 5, self.btn_height // 2,
                        text=self.text, fill='#ccccee',
                        font=('Segoe UI', 8))
    
    def _on_click(self, event):
        """Handle click"""
        self.enabled = not self.enabled
        self._draw()
        
        if self.command:
            self.command(self.enabled)
    
    def set_value(self, enabled):
        """Set enabled state"""
        self.enabled = enabled
        self._draw()
    
    def get_value(self):
        """Get enabled state"""
        return self.enabled

class PatternEditor(tk.Canvas):
    """
    Pattern editor widget showing triggers, accents, fills, substeps, and length for a pattern channel
    Supports probability mode where clicking adjusts step probability instead of triggers
    """

    def __init__(self, parent, channel_id=0, pattern_length=16, num_steps=16,
                 command=None, all_channels_command=None, length_change_callback=None, **kwargs):
        """
        Initialize pattern editor
        
        Args:
            channel_id: Which drum channel this editor represents
            pattern_length: Total steps in pattern (all channels)
            num_steps: Number of steps to display
            command: Callback when pattern changes (channel_id, step_index, lane_type, value)
            all_channels_command: Callback for multi-channel operations (step_index, lane_type, value, muted_channels)
            length_change_callback: Callback when pattern length is changed (new_length)
        """
        # Calculate dimensions - compact steps for better fit
        step_width = 32  # Compact steps
        lane_height = 14  # Compact lanes
        num_lanes = 6  # triggers, accents, fills, substeps, length display, step numbers
        
        width = num_steps * step_width + 40  # 40px for lane labels
        height = num_lanes * lane_height + 10
        
        super().__init__(parent, width=width, height=height,
                        bg='#2a2a3a', highlightthickness=1,
                        highlightbackground='#4a4a5a', **kwargs)
        
        self.channel_id = channel_id
        self.pattern_length = pattern_length
        self.num_steps = num_steps
        self.step_width = step_width
        self.lane_height = lane_height
        self.command = command
        self.all_channels_command = all_channels_command
        self.length_change_callback = length_change_callback
        
        # Pattern state
        self.triggers = [False] * num_steps
        self.accents = [False] * num_steps
        self.fills = [False] * num_steps
        self.probabilities = [100] * num_steps  # Per-step probability (0-100)
        self.substeps = [''] * num_steps  # Per-step substep pattern ('o' = play, '-' = don't play)
        
        # Display state
        self.current_position = 0  # Current playback position (green highlight)
        
        # Probability mode (when enabled, clicking adjusts probability)
        self.probability_mode = False
        
        # Lane names
        self.lanes = ['trig', 'acc', 'fill', 'sub']
        
        # Mouse state
        self.dragging = False
        self.drag_lane = None
        self.drag_start_x = 0
        self._drag_start_step = None
        self._drag_start_y = 0
        
        # Draw initial
        self._draw()
        
        # Bind events
        self.bind('<Button-1>', self._on_click)
        self.bind('<Control-Button-1>', self._on_ctrl_click)
        self.bind('<Shift-Button-1>', self._on_shift_click)
        self.bind('<Button-3>', self._on_right_click)  # Right-click for substeps menu
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
    
    def set_pattern_data(self, triggers, accents, fills, probabilities=None, substeps=None):
        """Update pattern data from pattern manager"""
        self.triggers = list(triggers)
        self.accents = list(accents)
        self.fills = list(fills)
        if probabilities is not None:
            self.probabilities = list(probabilities)
        else:
            self.probabilities = [100] * len(triggers)
        if substeps is not None:
            self.substeps = list(substeps)
        else:
            self.substeps = [''] * len(triggers)
        self._draw()
    
    def set_probability_mode(self, enabled: bool):
        """Enable/disable probability editing mode"""
        self.probability_mode = enabled
        self._draw()
    
    def set_current_position(self, position):
        """Update current playback position"""
        self.current_position = position
        self._draw()
    
    def _get_lane_at_y(self, y):
        """Determine which lane was clicked (trig, acc, fill, sub, or length)"""
        relative_y = y - 5
        lane = int(relative_y / self.lane_height)
        if lane >= 0 and lane < 4:
            return self.lanes[lane]
        elif lane == 4:
            return 'length'
        return None
    
    def _get_step_at_x(self, x):
        """Determine which step was clicked"""
        relative_x = x - 40
        if relative_x < 0:
            return None
        step = int(relative_x / self.step_width)
        if 0 <= step < self.num_steps:
            return step
        return None
    
    def _on_click(self, event):
        """Handle click start"""
        step = self._get_step_at_x(event.x)
        
        # In probability mode, clicking on any step adjusts probability
        if self.probability_mode and step is not None:
            self._drag_start_step = step
            self._drag_start_y = event.y
            self.dragging = True
            return
        
        lane = self._get_lane_at_y(event.y)
        
        if lane == 'sub' and step is not None:
            # Click on substeps lane - show substeps editor popup
            self._show_substeps_menu(event.x_root, event.y_root, step)
        elif lane == 'length' and step is not None and self.length_change_callback:
            # Click on length lane sets the pattern length to clicked step + 1
            new_length = step + 1
            self.pattern_length = new_length
            self.length_change_callback(new_length)
            self._draw()
        elif lane and step is not None:
            self.dragging = True
            self.drag_lane = lane
            self.drag_start_x = event.x
            self._toggle_step(lane, step)
    
    def _on_ctrl_click(self, event):
        """Handle Ctrl+Click: toggle both trigger and accent for a step"""
        lane = self._get_lane_at_y(event.y)
        step = self._get_step_at_x(event.x)
        
        if lane and step is not None and lane == 'trig':
            # Toggle trigger
            self.triggers[step] = not self.triggers[step]
            # Also toggle accent if trigger is on
            if self.triggers[step]:
                self.accents[step] = not self.accents[step]
            else:
                # Clear accent and fill if trigger is turned off
                self.accents[step] = False
                self.fills[step] = False
            
            self._draw()
            
            # Call command callbacks
            if self.command:
                self.command(self.channel_id, step, 'trig', self.triggers[step])
                self.command(self.channel_id, step, 'acc', self.accents[step])
    
    def _on_shift_click(self, event):
        """Handle Shift+Click: apply to all unmuted channels (requires all_channels_command)"""
        lane = self._get_lane_at_y(event.y)
        step = self._get_step_at_x(event.x)
        
        if lane and step is not None and self.all_channels_command:
            # Determine new value based on current state
            if lane == 'trig':
                new_value = not self.triggers[step]
                self.triggers[step] = new_value
            elif lane == 'acc':
                new_value = not self.accents[step]
                self.accents[step] = new_value
            elif lane == 'fill':
                new_value = not self.fills[step]
                self.fills[step] = new_value
            else:
                return
            
            self._draw()
            
            # Call the all-channels callback with empty muted_channels set (all channels active)
            self.all_channels_command(step, lane, new_value, set())
    
    def _on_right_click(self, event):
        """Handle right-click: show substeps menu for any step"""
        step = self._get_step_at_x(event.x)
        if step is not None:
            self._show_substeps_menu(event.x_root, event.y_root, step)
    
    def _show_substeps_menu(self, x, y, step):
        """Show context menu for editing substeps at the given step"""
        import tkinter as tk
        menu = tk.Menu(self, tearoff=0)
        
        current_substeps = self.substeps[step] if step < len(self.substeps) else ''
        
        # Clear substeps option
        menu.add_command(label="No substeps" + (" ✓" if current_substeps == '' else ""),
                        command=lambda: self._set_substeps(step, ''))
        menu.add_separator()
        
        # Common substep patterns
        patterns = [
            ('oo', '2 subs: ○○ (both play)'),
            ('o-', '2 subs: ○- (1st only)'),
            ('-o', '2 subs: -○ (2nd only)'),
            ('ooo', '3 subs: ○○○ (all play)'),
            ('oo-', '3 subs: ○○- (1st, 2nd)'),
            ('o-o', '3 subs: ○-○ (1st, 3rd)'),
            ('o--', '3 subs: ○-- (1st only)'),
            ('-oo', '3 subs: -○○ (2nd, 3rd)'),
            ('--o', '3 subs: --○ (3rd only)'),
            ('oooo', '4 subs: ○○○○ (all play)'),
            ('o-o-', '4 subs: ○-○- (alternating)'),
            ('-o-o', '4 subs: -○-○ (alternating)'),
            ('ooo-', '4 subs: ○○○- (first 3)'),
            ('o---', '4 subs: ○--- (1st only)'),
        ]
        
        for pattern, label in patterns:
            check = " ✓" if current_substeps == pattern else ""
            menu.add_command(label=label + check,
                           command=lambda p=pattern: self._set_substeps(step, p))
        
        menu.add_separator()
        menu.add_command(label="Custom...", command=lambda: self._show_custom_substeps_dialog(step))
        
        menu.tk_popup(x, y)
    
    def _set_substeps(self, step, pattern):
        """Set substeps pattern for a step"""
        if step < len(self.substeps):
            self.substeps[step] = pattern
            self._draw()
            if self.command:
                self.command(self.channel_id, step, 'sub', pattern)
    
    def _show_custom_substeps_dialog(self, step):
        """Show dialog for entering custom substeps pattern"""
        import tkinter as tk
        from tkinter import simpledialog
        
        current = self.substeps[step] if step < len(self.substeps) else ''
        result = simpledialog.askstring(
            "Custom Substeps",
            f"Enter substep pattern for step {step + 1}:\n" +
            "Use 'o' for play, '-' for skip\n" +
            f"Examples: 'oo-', 'o-o-', 'ooo'\n" +
            f"Current: '{current}'",
            initialvalue=current,
            parent=self
        )
        if result is not None:
            # Validate: only allow 'o', 'O', '-' characters
            clean = ''.join(c if c in 'oO-' else '' for c in result).lower()
            self._set_substeps(step, clean)

    
    def _on_drag(self, event):
        """Handle drag to select multiple steps or adjust probability"""
        if self.dragging:
            # In probability mode, vertical drag adjusts probability
            if self.probability_mode and self._drag_start_step is not None:
                step = self._drag_start_step
                # Calculate probability based on vertical position
                # Drag up = higher probability, drag down = lower
                delta_y = self._drag_start_y - event.y
                # 100 pixels of drag = 100% change
                new_prob = self.probabilities[step] + int(delta_y)
                new_prob = max(0, min(100, new_prob))
                if new_prob != self.probabilities[step]:
                    self.probabilities[step] = new_prob
                    self._drag_start_y = event.y
                    self._draw()
                    if self.command:
                        self.command(self.channel_id, step, 'prob', new_prob)
            elif self.drag_lane:
                # Normal mode: allow continuous clicking across multiple steps
                step = self._get_step_at_x(event.x)
                if step is not None:
                    self._toggle_step(self.drag_lane, step)
    
    def _on_release(self, event):
        """Handle drag end"""
        self.dragging = False
        self.drag_lane = None
        self._drag_start_step = None
    
    def _toggle_step(self, lane, step):
        """Toggle a step in a lane"""
        if lane == 'trig':
            self.triggers[step] = not self.triggers[step]
            # Clear accent/fill if trigger is off
            if not self.triggers[step]:
                self.accents[step] = False
                self.fills[step] = False
        elif lane == 'acc':
            if self.triggers[step]:  # Only allow accent if trigger is on
                self.accents[step] = not self.accents[step]
        elif lane == 'fill':
            if self.triggers[step]:  # Only allow fill if trigger is on
                self.fills[step] = not self.fills[step]
        
        self._draw()
        
        # Call command callback
        if self.command:
            self.command(self.channel_id, step, lane, 
                        self.triggers[step] if lane == 'trig' else
                        self.accents[step] if lane == 'acc' else
                        self.fills[step])
    
    def _draw(self):
        """Draw the pattern editor"""
        self.delete('all')
        
        # Draw background - highlight in probability mode
        bg_color = '#2a3a3a' if self.probability_mode else '#2a2a3a'
        self.create_rectangle(0, 0, self.winfo_width(), self.winfo_height(),
                             fill=bg_color, outline='#4a4a5a')
        
        # Draw lane labels on left
        for i, lane_name in enumerate(self.lanes):
            y = 5 + i * self.lane_height + self.lane_height // 2
            # Highlight "trig" label in probability mode
            label_color = '#66ddaa' if (self.probability_mode and i == 0) else '#8888aa'
            self.create_text(20, y, text=lane_name, fill=label_color,
                           font=('Segoe UI', 6), anchor='center')
        
        # Draw length indicator lane label
        y = 5 + 4 * self.lane_height + self.lane_height // 2
        self.create_text(20, y, text='len', fill='#8888aa',
                       font=('Segoe UI', 6), anchor='center')
        
        # Draw steps for each lane
        x_start = 40
        
        # Draw trigger lane (with probability visualization)
        self._draw_trigger_lane_with_prob()
        
        # Draw accent lane
        self._draw_lane(1, self.accents, '#ffaa44', '#ff8822')
        
        # Draw fill lane
        self._draw_lane(2, self.fills, '#4488ff', '#2266aa')
        
        # Draw substeps lane
        self._draw_substeps_lane()
        
        # Draw length indicator lane
        self._draw_length_lane()
        
        # Draw step numbers row
        self._draw_step_numbers()
        
        # Draw vertical grid lines
        for step in range(self.num_steps + 1):
            x = x_start + step * self.step_width
            self.create_line(x, 0, x, self.winfo_height() - self.lane_height,
                           fill='#3a3a4a', dash=(2,))
        
        # Draw horizontal grid lines
        for lane in range(5):  # 5 grid lines for 5 data lanes
            y = 5 + (lane + 1) * self.lane_height
            self.create_line(x_start, y, x_start + self.num_steps * self.step_width, y,
                           fill='#3a3a4a')
        
        # Draw current position indicator
        current_x = x_start + self.current_position * self.step_width + self.step_width // 2
        self.create_line(current_x, 0, current_x, self.winfo_height(),
                       fill='#44ff88', width=2)
    
    def _draw_trigger_lane_with_prob(self):
        """Draw trigger lane with probability visualization"""
        lane_index = 0
        x_start = 40
        y_base = 5 + lane_index * self.lane_height
        
        for step in range(len(self.triggers)):
            is_on = self.triggers[step]
            prob = self.probabilities[step] if step < len(self.probabilities) else 100
            
            x = x_start + step * self.step_width + 2
            y = y_base + 2
            w = self.step_width - 4
            h = self.lane_height - 4
            
            # Base color depends on trigger state
            if is_on:
                # Interpolate color based on probability
                # 100% = bright blue (#4488ff), 0% = dark (#223344)
                if self.probability_mode:
                    # In probability mode, show gradient from red (0%) to green (100%)
                    r = int(0x88 + (0x44 - 0x88) * prob / 100)
                    g = int(0x44 + (0xaa - 0x44) * prob / 100)
                    b = 0x55
                    color = f'#{r:02x}{g:02x}{b:02x}'
                else:
                    # Normal mode: brighter blue for higher probability
                    intensity = 0.4 + 0.6 * (prob / 100.0)
                    r = int(0x44 * intensity)
                    g = int(0x88 * intensity)
                    b = int(0xff * intensity)
                    color = f'#{r:02x}{g:02x}{b:02x}'
            else:
                color = '#3366cc'
            
            # Draw pill-shaped step button
            radius = min(w, h) // 2
            outline = '#66ddaa' if self.probability_mode else '#555566'
            self._draw_rounded_rect(x, y, x + w, y + h, radius, fill=color, outline=outline)
            
            # In probability mode, show percentage text on triggered steps
            if self.probability_mode and is_on and prob < 100:
                text_x = x + w // 2
                text_y = y + h // 2
                self.create_text(text_x, text_y, text=f'{prob}', 
                               fill='#ffffff', font=('Segoe UI', 6, 'bold'))
    
    def _draw_lane(self, lane_index, data, color_on, color_off):
        """Draw a single lane of steps"""
        x_start = 40
        y_base = 5 + lane_index * self.lane_height
        
        for step, is_on in enumerate(data):
            x = x_start + step * self.step_width + 2
            y = y_base + 2
            w = self.step_width - 4
            h = self.lane_height - 4
            
            # Color based on state
            color = color_on if is_on else color_off
            
            # Highlight if trigger is off (accent/fill only meaningful with trigger)
            if lane_index > 0 and not self.triggers[step]:
                color = '#444455'
            
            # Draw pill-shaped (rounded) step button
            radius = min(w, h) // 2
            self._draw_rounded_rect(x, y, x + w, y + h, radius, fill=color, outline='#555566')
    
    def _draw_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Draw a rounded rectangle (pill shape)"""
        # Clamp radius to half the smaller dimension
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        
        # Create points for a rounded rectangle using polygon
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        
        # Use smooth polygon for rounded corners
        self.create_polygon(points, smooth=True, **kwargs)
    
    def _draw_length_lane(self):
        """Draw the pattern length indicator"""
        lane_index = 4
        x_start = 40
        y_base = 5 + lane_index * self.lane_height
        
        # Show steps 0 to pattern_length
        for step in range(self.num_steps):
            x = x_start + step * self.step_width + 2
            y = y_base + 2
            w = self.step_width - 4
            h = self.lane_height - 4
            
            # Check if this step is within pattern length
            if step < self.pattern_length:
                # Fill color for active pattern length
                color = '#2255aa'
            else:
                # Dimmed for outside pattern length
                color = '#222233'
            
            # Draw pill-shaped step like other lanes
            radius = min(w, h) // 2
            self._draw_rounded_rect(x, y, x + w, y + h, radius, fill=color, outline='#555566')
    
    def _draw_step_numbers(self):
        """Draw the step numbers row (1-16)"""
        lane_index = 5
        x_start = 40
        y_base = 5 + lane_index * self.lane_height
        
        for step in range(self.num_steps):
            x = x_start + step * self.step_width + self.step_width // 2
            y = y_base + self.lane_height // 2
            step_num = step + 1
            self.create_text(x, y, text=str(step_num),
                           fill='#8888aa', font=('Segoe UI', 6))
    
    def get_triggers(self):
        """Get current triggers"""
        return list(self.triggers)
    
    def get_accents(self):
        """Get current accents"""
        return list(self.accents)
    
    def get_fills(self):
        """Get current fills"""
        return list(self.fills)
    
    def get_probabilities(self):
        """Get current probabilities"""
        return list(self.probabilities)
    
    def get_substeps(self):
        """Get current substeps patterns"""
        return list(self.substeps)
    
    def _draw_substeps_lane(self):
        """Draw the substeps lane showing substep patterns"""
        lane_index = 3
        x_start = 40
        y_base = 5 + lane_index * self.lane_height
        
        for step in range(self.num_steps):
            substep_pattern = self.substeps[step] if step < len(self.substeps) else ''
            has_trigger = self.triggers[step] if step < len(self.triggers) else False
            
            x = x_start + step * self.step_width + 2
            y = y_base + 2
            w = self.step_width - 4
            h = self.lane_height - 4
            
            # Background color based on whether substeps are set
            if substep_pattern and has_trigger:
                # Has substeps and trigger - show active
                color = '#aa44aa'  # Purple for active substeps
            elif substep_pattern:
                # Has substeps but no trigger - dimmed
                color = '#553355'
            else:
                # No substeps - default
                color = '#3a3a4a'
            
            # Draw pill-shaped step button
            radius = min(w, h) // 2
            self._draw_rounded_rect(x, y, x + w, y + h, radius, fill=color, outline='#555566')
            
            # Draw substep pattern visualization inside the cell
            if substep_pattern:
                self._draw_substep_dots(x, y, w, h, substep_pattern)
    
    def _draw_substep_dots(self, x, y, w, h, pattern):
        """Draw small dots representing substep pattern inside a cell"""
        if not pattern:
            return
        
        num_subs = len(pattern)
        dot_spacing = w / (num_subs + 1)
        dot_radius = min(3, (h - 4) // 2)
        
        for i, char in enumerate(pattern):
            dot_x = x + dot_spacing * (i + 1)
            dot_y = y + h // 2
            
            if char == 'o' or char == 'O':
                # Filled dot for 'play'
                self.create_oval(dot_x - dot_radius, dot_y - dot_radius,
                               dot_x + dot_radius, dot_y + dot_radius,
                               fill='#ffffff', outline='')
            else:
                # Empty dot outline for 'skip'
                self.create_oval(dot_x - dot_radius, dot_y - dot_radius,
                               dot_x + dot_radius, dot_y + dot_radius,
                               fill='', outline='#888888', width=1)


class MatrixEditor(tk.Canvas):
    """
    Matrix Editor widget for editing all channels simultaneously
    Shows a grid of channels (rows) vs steps (columns)
    """

    def __init__(self, parent, num_channels=8, num_steps=16, 
                 command=None, **kwargs):
        """
        Initialize matrix editor
        
        Args:
            num_channels: Number of drum channels
            num_steps: Number of pattern steps
            command: Callback when pattern changes (channel_id, step_index, value)
        """
        step_width = 20
        channel_height = 16
        
        width = num_steps * step_width + 35
        height = num_channels * channel_height + 8
        
        super().__init__(parent, width=width, height=height,
                        bg='#2a2a3a', highlightthickness=1,
                        highlightbackground='#4a4a5a', **kwargs)
        
        self.num_channels = num_channels
        self.num_steps = num_steps
        self.step_width = step_width
        self.channel_height = channel_height
        self.command = command
        
        # Pattern state (triggers only in matrix view)
        self.matrix = [[False] * num_steps for _ in range(num_channels)]
        
        # Display state
        self.current_position = 0
        
        # Draw initial
        self._draw()
        
        # Bind events
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        
        # Interaction state
        self.dragging = False
        self.drag_channel = None
    
    def set_matrix_data(self, matrix):
        """Update matrix data from pattern manager"""
        self.matrix = [list(row) for row in matrix]
        self._draw()
    
    def set_current_position(self, position):
        """Update current playback position"""
        self.current_position = position
        self._draw()
    
    def _get_channel_at_y(self, y):
        """Determine which channel was clicked"""
        relative_y = y - 5
        if relative_y < 0:
            return None
        channel = int(relative_y / self.channel_height)
        if 0 <= channel < self.num_channels:
            return channel
        return None
    
    def _get_step_at_x(self, x):
        """Determine which step was clicked"""
        relative_x = x - 35
        if relative_x < 0:
            return None
        step = int(relative_x / self.step_width)
        if 0 <= step < self.num_steps:
            return step
        return None
    
    def _on_click(self, event):
        """Handle click start"""
        channel = self._get_channel_at_y(event.y)
        step = self._get_step_at_x(event.x)
        
        if channel is not None and step is not None:
            self.dragging = True
            self.drag_channel = channel
            self._toggle_cell(channel, step)
    
    def _on_drag(self, event):
        """Handle drag to select multiple cells"""
        if self.dragging and self.drag_channel is not None:
            step = self._get_step_at_x(event.x)
            if step is not None:
                self._toggle_cell(self.drag_channel, step)
    
    def _on_release(self, event):
        """Handle drag end"""
        self.dragging = False
        self.drag_channel = None
    
    def _toggle_cell(self, channel, step):
        """Toggle a cell in the matrix"""
        self.matrix[channel][step] = not self.matrix[channel][step]
        self._draw()
        
        if self.command:
            self.command(channel, step, self.matrix[channel][step])
    
    def _draw(self):
        """Draw the matrix editor"""
        self.delete('all')
        
        # Draw background
        self.create_rectangle(0, 0, self.winfo_width(), self.winfo_height(),
                             fill='#2a2a3a', outline='#4a4a5a')
        
        # Draw channel labels on left
        for ch in range(self.num_channels):
            y = 5 + ch * self.channel_height + self.channel_height // 2
            self.create_text(14, y, text=f"ch{ch+1}", fill='#8888aa',
                           font=('Segoe UI', 6), anchor='center')
        
        # Draw step numbers on top
        x_start = 35
        for step in range(self.num_steps):
            if (step + 1) % 4 == 1:  # Every 4 steps
                x = x_start + step * self.step_width + self.step_width // 2
                self.create_text(x, -5, text=str(step + 1), fill='#6688ff',
                               font=('Segoe UI', 6), anchor='center')
        
        # Draw matrix cells
        for ch in range(self.num_channels):
            for step in range(self.num_steps):
                x = x_start + step * self.step_width + 2
                y = 5 + ch * self.channel_height + 2
                w = self.step_width - 4
                h = self.channel_height - 4
                
                # Color based on trigger state
                if self.matrix[ch][step]:
                    color = '#4488ff'
                    self.create_rectangle(x, y, x + w, y + h,
                                        fill=color, outline='#555566', width=1)
                    # Draw indicator
                    self.create_oval(x + 2, y + 2, x + w - 2, y + h - 2,
                                   outline='#ffffff', width=1)
                else:
                    color = '#333344'
                    self.create_rectangle(x, y, x + w, y + h,
                                        fill=color, outline='#555566', width=1)
        
        # Draw vertical grid lines
        for step in range(self.num_steps + 1):
            x = x_start + step * self.step_width
            self.create_line(x, 0, x, self.winfo_height(),
                           fill='#3a3a4a', dash=(2,))
        
        # Draw horizontal grid lines
        for ch in range(self.num_channels + 1):
            y = 5 + ch * self.channel_height
            self.create_line(x_start, y, x_start + self.num_steps * self.step_width, y,
                           fill='#3a3a4a')
        
        # Draw current position indicator
        current_x = x_start + self.current_position * self.step_width + self.step_width // 2
        self.create_line(current_x, 0, current_x, self.winfo_height(),
                       fill='#44ff88', width=2)
    
    def get_matrix(self):
        """Get current matrix state"""
        return [list(row) for row in self.matrix]