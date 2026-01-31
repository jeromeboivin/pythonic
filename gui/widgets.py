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
    """
    
    def __init__(self, parent, size=50, min_val=0, max_val=100, default=50,
                 label="", unit="", logarithmic=False, command=None, **kwargs):
        super().__init__(parent, width=size, height=size + 20,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default  # Store default for reset
        self.value = default
        self.label = label
        self.unit = unit  # Unit string for display (Hz, dB, %, etc.)
        self.logarithmic = logarithmic
        self.command = command
        
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
        
        # Calculate indicator angle
        # Range is 225° (bottom-left) to -45° (bottom-right) = 270° sweep
        value_range = self.max_val - self.min_val
        if value_range != 0:
            normalized = (self.value - self.min_val) / value_range
        else:
            normalized = 0.5
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
            
            value_range = self.max_val - self.min_val
            
            if self.linear_mode:
                # Linear/vertical drag
                delta_y = self.drag_start_y - event.y
                # Sensitivity: 200 pixels = full range
                sensitivity = (value_range / 200.0) * sensitivity_multiplier
                new_value = self.drag_start_value + delta_y * sensitivity
            else:
                # Circular drag mode
                cx, cy = self.size // 2, self.size // 2
                
                # Calculate angle from center
                dx = event.x - cx
                dy = cy - event.y  # Inverted for screen coords
                angle = math.atan2(dy, dx)
                
                # Convert angle to value (225° to -45° range)
                # Normalize angle to 0-1 range
                angle_deg = math.degrees(angle)
                if angle_deg < -45:
                    angle_deg += 360
                normalized = (225 - angle_deg) / 270
                normalized = max(0, min(1, normalized))
                
                new_value = self.min_val + normalized * value_range
            
            self.set_value(new_value)
            self._show_hint()
    
    def _on_release(self, event):
        """Stop dragging"""
        self.dragging = False
        # Hide hint window
        try:
            hint = HintWindow.get_instance(self.winfo_toplevel())
            hint.hide()
        except:
            pass
    
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
        step = (self.max_val - self.min_val) / 100
        self.set_value(self.value + delta * step)
    
    def _on_scroll_linux(self, event):
        """Handle mouse wheel on Linux"""
        delta = 1 if event.num == 4 else -1
        step = (self.max_val - self.min_val) / 100
        self.set_value(self.value + delta * step)
    
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
    """
    
    def __init__(self, parent, width=30, height=100, min_val=0, max_val=100,
                 default=50, label="", unit="", command=None, **kwargs):
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
        self.command = command
        
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
    
    def _draw(self):
        """Draw the slider"""
        self.delete('all')
        
        # Draw track
        self.create_rectangle(
            self.track_x - 3, self.track_top,
            self.track_x + 3, self.track_bottom,
            fill='#222233', outline='#555566'
        )
        
        # Calculate handle position
        value_range = self.max_val - self.min_val
        if value_range != 0:
            normalized = (self.value - self.min_val) / value_range
        else:
            normalized = 0.5
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
            
            # Calculate value change
            delta_y = self.drag_start_y - event.y
            value_range = self.max_val - self.min_val
            sensitivity = (value_range / self.track_range) * sensitivity_multiplier
            
            new_value = self.drag_start_value + delta_y * sensitivity
            self.set_value(new_value)
            self._show_hint()
    
    def _on_release(self, event):
        """Handle release"""
        self.dragging = False
        try:
            hint = HintWindow.get_instance(self.winfo_toplevel())
            hint.hide()
        except:
            pass
    
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
        step = (self.max_val - self.min_val) / 50
        self.set_value(self.value + delta * step)
    
    def _on_scroll_linux(self, event):
        """Handle scroll on Linux"""
        delta = 1 if event.num == 4 else -1
        step = (self.max_val - self.min_val) / 50
        self.set_value(self.value + delta * step)
    
    def _update_from_mouse(self, y):
        """Update value from mouse position"""
        # Clamp y to track range
        y = max(self.track_top, min(self.track_bottom, y))
        
        # Convert to value (inverted: top = max)
        normalized = 1.0 - (y - self.track_top) / self.track_range
        value = self.min_val + normalized * (self.max_val - self.min_val)
        
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
            self.command(self.channel_num)
    
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
        super().__init__(parent, width=90, height=40,
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
            x = 15 + i * 30
            y = 20
            
            # Selection highlight
            if i == self.selected:
                self.create_rectangle(x - 12, y - 15, x + 12, y + 15,
                                     fill='#5566aa', outline='')
            
            # Draw waveform icon
            if waveform == 'sine':
                # Sine wave
                points = []
                for j in range(20):
                    px = x - 8 + j * 0.8
                    py = y - 8 * math.sin(j * math.pi / 10)
                    points.extend([px, py])
                self.create_line(points, fill='#aaccff', width=2, smooth=True)
                
            elif waveform == 'triangle':
                # Triangle wave
                self.create_line(x - 8, y, x - 2, y - 10, x + 4, y + 10, x + 8, y,
                               fill='#aaccff', width=2)
                
            elif waveform == 'sawtooth':
                # Sawtooth wave
                self.create_line(x - 8, y + 8, x, y - 8, x, y + 8, x + 8, y - 8,
                               fill='#aaccff', width=2)
    
    def _on_click(self, event):
        """Handle click"""
        # Determine which waveform was clicked
        idx = (event.x - 3) // 30
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
        super().__init__(parent, width=width, height=30,
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
                self.create_rectangle(x1 + 2, 2, x2 - 2, 28,
                                     fill='#5566aa', outline='#7788cc')
            else:
                self.create_rectangle(x1 + 2, 2, x2 - 2, 28,
                                     fill='#4a4a5a', outline='#666688')
            
            # Label
            self.create_text((x1 + x2) // 2, 15,
                           text=opt, fill='#ccccee',
                           font=('Segoe UI', 8))
    
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
    Pattern editor widget showing triggers, accents, fills, and length for a pattern channel
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
        # Calculate dimensions - wider steps for better usability
        step_width = 45  # Wider steps for better click targets
        lane_height = 18  # Taller lanes for better visibility
        num_lanes = 5  # triggers, accents, fills, length display, step numbers
        
        width = num_steps * step_width + 50  # 50px for lane labels
        height = num_lanes * lane_height + 15
        
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
        
        # Display state
        self.current_position = 0  # Current playback position (green highlight)
        
        # Lane names
        self.lanes = ['trig', 'acc', 'fill']
        
        # Mouse state
        self.dragging = False
        self.drag_lane = None
        self.drag_start_x = 0
        
        # Draw initial
        self._draw()
        
        # Bind events
        self.bind('<Button-1>', self._on_click)
        self.bind('<Control-Button-1>', self._on_ctrl_click)
        self.bind('<Shift-Button-1>', self._on_shift_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
    
    def set_pattern_data(self, triggers, accents, fills):
        """Update pattern data from pattern manager"""
        self.triggers = list(triggers)
        self.accents = list(accents)
        self.fills = list(fills)
        self._draw()
    
    def set_current_position(self, position):
        """Update current playback position"""
        self.current_position = position
        self._draw()
    
    def _get_lane_at_y(self, y):
        """Determine which lane was clicked (trig, acc, fill, or length)"""
        relative_y = y - 5
        lane = int(relative_y / self.lane_height)
        if lane >= 0 and lane < 3:
            return self.lanes[lane]
        elif lane == 3:
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
        lane = self._get_lane_at_y(event.y)
        step = self._get_step_at_x(event.x)
        
        if lane == 'length' and step is not None and self.length_change_callback:
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

    
    def _on_drag(self, event):
        """Handle drag to select multiple steps"""
        if self.dragging and self.drag_lane:
            # Allow continuous clicking across multiple steps
            step = self._get_step_at_x(event.x)
            if step is not None:
                self._toggle_step(self.drag_lane, step)
    
    def _on_release(self, event):
        """Handle drag end"""
        self.dragging = False
        self.drag_lane = None
    
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
        
        # Draw background
        self.create_rectangle(0, 0, self.winfo_width(), self.winfo_height(),
                             fill='#2a2a3a', outline='#4a4a5a')
        
        # Draw lane labels on left
        for i, lane_name in enumerate(self.lanes):
            y = 5 + i * self.lane_height + self.lane_height // 2
            self.create_text(25, y, text=lane_name, fill='#8888aa',
                           font=('Segoe UI', 8), anchor='center')
        
        # Draw length indicator lane label
        y = 5 + 3 * self.lane_height + self.lane_height // 2
        self.create_text(25, y, text='len', fill='#8888aa',
                       font=('Segoe UI', 8), anchor='center')
        
        # Draw steps for each lane
        x_start = 50
        
        # Draw trigger lane
        self._draw_lane(0, self.triggers, '#4488ff', '#3366cc')
        
        # Draw accent lane
        self._draw_lane(1, self.accents, '#ffaa44', '#ff8822')
        
        # Draw fill lane
        self._draw_lane(2, self.fills, '#4488ff', '#2266aa')
        
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
        for lane in range(4):
            y = 5 + (lane + 1) * self.lane_height
            self.create_line(x_start, y, x_start + self.num_steps * self.step_width, y,
                           fill='#3a3a4a')
        
        # Draw current position indicator
        current_x = x_start + self.current_position * self.step_width + self.step_width // 2
        self.create_line(current_x, 0, current_x, self.winfo_height(),
                       fill='#44ff88', width=2)
    
    def _draw_lane(self, lane_index, data, color_on, color_off):
        """Draw a single lane of steps"""
        x_start = 50
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
        lane_index = 3
        x_start = 50
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
        lane_index = 4
        x_start = 50
        y_base = 5 + lane_index * self.lane_height
        
        for step in range(self.num_steps):
            x = x_start + step * self.step_width + self.step_width // 2
            y = y_base + self.lane_height // 2
            step_num = step + 1
            self.create_text(x, y, text=str(step_num),
                           fill='#8888aa', font=('Segoe UI', 7))
    
    def get_triggers(self):
        """Get current triggers"""
        return list(self.triggers)
    
    def get_accents(self):
        """Get current accents"""
        return list(self.accents)
    
    def get_fills(self):
        """Get current fills"""
        return list(self.fills)


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
        step_width = 25
        channel_height = 20
        
        width = num_steps * step_width + 40
        height = num_channels * channel_height + 10
        
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
        relative_x = x - 40
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
            self.create_text(15, y, text=f"ch{ch+1}", fill='#8888aa',
                           font=('Segoe UI', 7), anchor='center')
        
        # Draw step numbers on top
        x_start = 40
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