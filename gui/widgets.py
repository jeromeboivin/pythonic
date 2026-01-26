"""
Custom Widgets for Pythonic GUI
Knobs, faders, and buttons styled like the real application
"""

import tkinter as tk
from tkinter import ttk
import math


class RotaryKnob(tk.Canvas):
    """
    Rotary knob widget styled like Pythonic's metallic knobs
    """
    
    def __init__(self, parent, size=50, min_val=0, max_val=100, default=50,
                 label="", logarithmic=False, command=None, **kwargs):
        super().__init__(parent, width=size, height=size + 20,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.label = label
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
        self.drag_start_value = 0
        
        # Draw initial state
        self._draw()
        
        # Bind events
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Double-Button-1>', self._on_double_click)
        self.bind('<MouseWheel>', self._on_scroll)
    
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
        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
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
        """Start dragging"""
        self.dragging = True
        self.drag_start_y = event.y
        self.drag_start_value = self.value
    
    def _on_drag(self, event):
        """Handle drag to change value"""
        if self.dragging:
            # Calculate value change based on vertical movement
            delta_y = self.drag_start_y - event.y
            value_range = self.max_val - self.min_val
            
            # Sensitivity: 200 pixels = full range
            sensitivity = value_range / 200.0
            
            new_value = self.drag_start_value + delta_y * sensitivity
            self.set_value(new_value)
    
    def _on_release(self, event):
        """Stop dragging"""
        self.dragging = False
    
    def _on_double_click(self, event):
        """Reset to default on double-click"""
        default = (self.min_val + self.max_val) / 2
        self.set_value(default)
    
    def _on_scroll(self, event):
        """Handle mouse wheel"""
        delta = event.delta / 120  # Windows scroll delta
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
    Vertical slider/fader styled like Pythonic
    """
    
    def __init__(self, parent, width=30, height=100, min_val=0, max_val=100,
                 default=50, label="", command=None, **kwargs):
        super().__init__(parent, width=width, height=height + 20,
                        bg='#3a3a4a', highlightthickness=0, **kwargs)
        
        self.slider_width = width
        self.slider_height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.label = label
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
        
        self._draw()
        
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<MouseWheel>', self._on_scroll)
    
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
        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
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
        """Handle click"""
        self.dragging = True
        self._update_from_mouse(event.y)
    
    def _on_drag(self, event):
        """Handle drag"""
        if self.dragging:
            self._update_from_mouse(event.y)
    
    def _on_release(self, event):
        """Handle release"""
        self.dragging = False
    
    def _on_scroll(self, event):
        """Handle scroll"""
        delta = event.delta / 120
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
