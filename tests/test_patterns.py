"""
Pythonic Pattern System Test Suite

Tests for pattern chaining, fills, pattern operations, and playback logic.

Run with: pytest tests/test_patterns.py -v
Or: python tests/test_patterns.py
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import pytest, but allow running without it
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from pythonic.pattern_manager import PatternManager, Pattern, PatternChannel, PatternStep


class TestPatternBasics:
    """Test basic pattern creation and manipulation"""
    
    def test_pattern_manager_initialization(self):
        """PatternManager should initialize with 12 patterns (A-L)"""
        pm = PatternManager()
        assert len(pm.patterns) == 12
        assert pm.patterns[0].name == 'A'
        assert pm.patterns[11].name == 'L'
    
    def test_pattern_default_length(self):
        """Default pattern length should be 16 steps"""
        pm = PatternManager()
        assert pm.patterns[0].length == 16
        for pattern in pm.patterns:
            assert len(pattern.channels[0].steps) == 16
    
    def test_pattern_set_length(self):
        """Pattern length can be changed"""
        pm = PatternManager()
        pattern = pm.get_pattern(0)
        
        # Increase length
        pattern.set_length(32)
        assert pattern.length == 32
        assert len(pattern.channels[0].steps) == 32
        
        # Decrease length
        pattern.set_length(8)
        assert pattern.length == 8
        assert len(pattern.channels[0].steps) == 8
    
    def test_pattern_trigger_setting(self):
        """Triggers can be set and retrieved"""
        pm = PatternManager()
        pattern = pm.get_pattern(0)
        channel = pattern.get_channel(0)
        
        # Set some triggers
        channel.set_trigger(0, True)
        channel.set_trigger(4, True)
        channel.set_trigger(8, True)
        
        triggers = channel.get_triggers()
        assert triggers[0] == True
        assert triggers[1] == False
        assert triggers[4] == True
        assert triggers[8] == True
    
    def test_pattern_accent_setting(self):
        """Accents can be set and retrieved"""
        pm = PatternManager()
        pattern = pm.get_pattern(0)
        channel = pattern.get_channel(0)
        
        channel.set_accent(0, True)
        channel.set_accent(4, True)
        
        accents = channel.get_accents()
        assert accents[0] == True
        assert accents[4] == True
        assert accents[1] == False
    
    def test_pattern_fill_setting(self):
        """Fills can be set and retrieved"""
        pm = PatternManager()
        pattern = pm.get_pattern(0)
        channel = pattern.get_channel(0)
        
        channel.set_fill(0, True)
        channel.set_fill(8, True)
        
        fills = channel.get_fills()
        assert fills[0] == True
        assert fills[8] == True
        assert fills[4] == False
    
    def test_pattern_is_empty(self):
        """Pattern should report empty status correctly"""
        pm = PatternManager()
        pattern = pm.get_pattern(0)
        
        # Initially empty
        assert pattern.is_empty() == True
        
        # Add a trigger
        pattern.get_channel(0).set_trigger(0, True)
        assert pattern.is_empty() == False
        
        # Clear pattern
        pattern.clear()
        assert pattern.is_empty() == True
    
    def test_pattern_copy(self):
        """Pattern copy should create independent copy"""
        pm = PatternManager()
        pattern = pm.get_pattern(0)
        
        # Set some data
        pattern.get_channel(0).set_trigger(0, True)
        pattern.get_channel(0).set_accent(0, True)
        pattern.chained_to_next = True
        
        # Copy
        copy = pattern.copy()
        
        # Verify copy has same data
        assert copy.get_channel(0).get_triggers()[0] == True
        assert copy.get_channel(0).get_accents()[0] == True
        assert copy.chained_to_next == True
        
        # Modify original
        pattern.get_channel(0).set_trigger(0, False)
        
        # Copy should be unchanged
        assert copy.get_channel(0).get_triggers()[0] == True


class TestPatternChaining:
    """Test pattern chaining functionality"""
    
    def test_chain_patterns(self):
        """chain_patterns should link two adjacent patterns"""
        pm = PatternManager()
        
        # Chain A to B
        result = pm.chain_patterns(0, 1)
        assert result == True
        assert pm.patterns[0].chained_to_next == True
        assert pm.patterns[1].chained_from_prev == True
    
    def test_chain_patterns_non_adjacent_fails(self):
        """chain_patterns should fail for non-adjacent patterns"""
        pm = PatternManager()
        
        # Try to chain A to C (non-adjacent)
        result = pm.chain_patterns(0, 2)
        assert result == False
        assert pm.patterns[0].chained_to_next == False
    
    def test_chain_patterns_invalid_index(self):
        """chain_patterns should handle invalid indices"""
        pm = PatternManager()
        
        # Invalid indices
        assert pm.chain_patterns(-1, 0) == False
        assert pm.chain_patterns(11, 12) == False
    
    def test_unchain_patterns(self):
        """unchain_patterns should remove chain link"""
        pm = PatternManager()
        
        # Chain then unchain
        pm.chain_patterns(0, 1)
        result = pm.unchain_patterns(0, 1)
        
        assert result == True
        assert pm.patterns[0].chained_to_next == False
        assert pm.patterns[1].chained_from_prev == False
    
    def test_toggle_chain_to_next(self):
        """toggle_chain_to_next should toggle chain state"""
        pm = PatternManager()
        
        # Toggle on
        result = pm.toggle_chain_to_next(0)
        assert result == True
        assert pm.patterns[0].chained_to_next == True
        assert pm.patterns[1].chained_from_prev == True
        
        # Toggle off
        result = pm.toggle_chain_to_next(0)
        assert result == False
        assert pm.patterns[0].chained_to_next == False
        assert pm.patterns[1].chained_from_prev == False
    
    def test_toggle_chain_from_prev(self):
        """toggle_chain_from_prev should toggle chain state from the other direction"""
        pm = PatternManager()
        
        # Toggle on (chain B from A)
        result = pm.toggle_chain_from_prev(1)
        assert result == True
        assert pm.patterns[0].chained_to_next == True
        assert pm.patterns[1].chained_from_prev == True
    
    def test_toggle_chain_boundary_cases(self):
        """Toggle chain should handle boundary cases"""
        pm = PatternManager()
        
        # Can't chain from prev on pattern A (index 0)
        result = pm.toggle_chain_from_prev(0)
        assert result == False
        
        # Can't chain to next on pattern L (index 11)
        result = pm.toggle_chain_to_next(11)
        assert result == False
    
    def test_get_chain_start(self):
        """get_chain_start should find first pattern in chain"""
        pm = PatternManager()
        
        # Single pattern (not in chain)
        assert pm.get_chain_start(0) == 0
        assert pm.get_chain_start(5) == 5
        
        # Chain A-B-C
        pm.chain_patterns(0, 1)
        pm.chain_patterns(1, 2)
        
        assert pm.get_chain_start(0) == 0
        assert pm.get_chain_start(1) == 0
        assert pm.get_chain_start(2) == 0
        
        # Pattern D is not in chain
        assert pm.get_chain_start(3) == 3
    
    def test_get_chain_end(self):
        """get_chain_end should find last pattern in chain"""
        pm = PatternManager()
        
        # Single pattern
        assert pm.get_chain_end(0) == 0
        
        # Chain A-B-C
        pm.chain_patterns(0, 1)
        pm.chain_patterns(1, 2)
        
        assert pm.get_chain_end(0) == 2
        assert pm.get_chain_end(1) == 2
        assert pm.get_chain_end(2) == 2
    
    def test_get_chain_patterns(self):
        """get_chain_patterns should return all patterns in chain"""
        pm = PatternManager()
        
        # Single pattern
        assert pm.get_chain_patterns(5) == [5]
        
        # Chain A-B-C
        pm.chain_patterns(0, 1)
        pm.chain_patterns(1, 2)
        
        assert pm.get_chain_patterns(0) == [0, 1, 2]
        assert pm.get_chain_patterns(1) == [0, 1, 2]
        assert pm.get_chain_patterns(2) == [0, 1, 2]
    
    def test_is_in_chain(self):
        """is_in_chain should detect if pattern is part of a chain"""
        pm = PatternManager()
        
        # Initially not in chain
        assert pm.is_in_chain(0) == False
        assert pm.is_in_chain(1) == False
        
        # Chain A-B
        pm.chain_patterns(0, 1)
        
        assert pm.is_in_chain(0) == True
        assert pm.is_in_chain(1) == True
        assert pm.is_in_chain(2) == False
    
    def test_get_next_pattern_in_chain(self):
        """get_next_pattern_in_chain should return correct next pattern"""
        pm = PatternManager()
        
        # Not in chain - returns None
        assert pm.get_next_pattern_in_chain(0) == None
        
        # Chain A-B-C
        pm.chain_patterns(0, 1)
        pm.chain_patterns(1, 2)
        
        # From A, go to B
        assert pm.get_next_pattern_in_chain(0) == 1
        # From B, go to C
        assert pm.get_next_pattern_in_chain(1) == 2
        # From C (end), loop back to A
        assert pm.get_next_pattern_in_chain(2) == 0
    
    def test_get_next_pattern_in_chain_single(self):
        """Single pattern in chain should loop to itself"""
        pm = PatternManager()
        
        # A single pattern not chained to anything just loops
        assert pm.get_next_pattern_in_chain(0) == None
    
    def test_advance_to_next_pattern(self):
        """advance_to_next_pattern should advance playback correctly"""
        pm = PatternManager()
        
        # Set up chain A-B-C
        pm.chain_patterns(0, 1)
        pm.chain_patterns(1, 2)
        
        # Start playback on A
        pm.start_playback(0)
        assert pm.playing_pattern_index == 0
        
        # Advance to B
        result = pm.advance_to_next_pattern()
        assert result == True
        assert pm.playing_pattern_index == 1
        assert pm.play_position == 0
        
        # Advance to C
        result = pm.advance_to_next_pattern()
        assert result == True
        assert pm.playing_pattern_index == 2
        
        # Advance loops back to A
        result = pm.advance_to_next_pattern()
        assert result == True
        assert pm.playing_pattern_index == 0
    
    def test_advance_to_next_pattern_no_chain(self):
        """advance_to_next_pattern on non-chained pattern should return False"""
        pm = PatternManager()
        
        # Start playback on D (not chained)
        pm.start_playback(3)
        
        # Advance should return False (no chain progression)
        result = pm.advance_to_next_pattern()
        assert result == False
        assert pm.playing_pattern_index == 3  # Still on same pattern
        assert pm.play_position == 0
    
    def test_advance_not_playing(self):
        """advance_to_next_pattern when not playing should return False"""
        pm = PatternManager()
        pm.chain_patterns(0, 1)
        
        # Not playing
        pm.is_playing = False
        result = pm.advance_to_next_pattern()
        assert result == False
    
    def test_multiple_separate_chains(self):
        """Multiple separate chains should work independently"""
        pm = PatternManager()
        
        # Chain A-B
        pm.chain_patterns(0, 1)
        # Chain D-E-F
        pm.chain_patterns(3, 4)
        pm.chain_patterns(4, 5)
        
        # Check A-B chain
        assert pm.get_chain_patterns(0) == [0, 1]
        assert pm.get_chain_patterns(1) == [0, 1]
        
        # Check D-E-F chain
        assert pm.get_chain_patterns(3) == [3, 4, 5]
        assert pm.get_chain_patterns(4) == [3, 4, 5]
        assert pm.get_chain_patterns(5) == [3, 4, 5]
        
        # C is not in any chain
        assert pm.get_chain_patterns(2) == [2]


class TestPatternSerialization:
    """Test pattern serialization with chaining"""
    
    def test_pattern_to_dict_includes_chain(self):
        """Pattern to_dict should include chaining info"""
        pattern = Pattern("A", 16, 8)
        pattern.chained_to_next = True
        pattern.chained_from_prev = True
        
        data = pattern.to_dict()
        
        assert data['chained_to_next'] == True
        assert data['chained_from_prev'] == True
    
    def test_pattern_from_dict_restores_chain(self):
        """Pattern from_dict should restore chaining info"""
        data = {
            'name': 'A',
            'length': 16,
            'chained_to_next': True,
            'chained_from_prev': False,
            'channels': [
                {'channel_id': i, 'steps': [{'trigger': False, 'accent': False, 'fill': False} for _ in range(16)]}
                for i in range(8)
            ]
        }
        
        pattern = Pattern.from_dict(data)
        
        assert pattern.chained_to_next == True
        assert pattern.chained_from_prev == False
    
    def test_pattern_manager_serialization_preserves_chains(self):
        """PatternManager serialization should preserve chains"""
        pm = PatternManager()
        
        # Set up chains
        pm.chain_patterns(0, 1)
        pm.chain_patterns(1, 2)
        pm.chain_patterns(5, 6)
        
        # Add some triggers
        pm.patterns[0].get_channel(0).set_trigger(0, True)
        
        # Serialize
        data = pm.to_dict()
        
        # Deserialize into new manager
        pm2 = PatternManager()
        pm2.from_dict(data)
        
        # Verify chains preserved
        assert pm2.patterns[0].chained_to_next == True
        assert pm2.patterns[1].chained_from_prev == True
        assert pm2.patterns[1].chained_to_next == True
        assert pm2.patterns[2].chained_from_prev == True
        assert pm2.patterns[5].chained_to_next == True
        assert pm2.patterns[6].chained_from_prev == True
        
        # Verify trigger preserved
        assert pm2.patterns[0].get_channel(0).get_triggers()[0] == True

    def test_load_from_preset_data_sets_chain_correctly(self):
        """load_from_preset_data should set chained_from_prev based on chained_to_next"""
        pm = PatternManager()
        
        # Simulate preset data where A-B-C are chained (A and B have chained=True)
        patterns_data = {
            'A': {
                'length': 16,
                'chained': True,  # Chains to B
                'channels': {}
            },
            'B': {
                'length': 16,
                'chained': True,  # Chains to C
                'channels': {}
            },
            'C': {
                'length': 16,
                'chained': False,  # End of chain
                'channels': {}
            },
            'D': {
                'length': 16,
                'chained': False,
                'channels': {}
            }
        }
        
        pm.load_from_preset_data(patterns_data)
        
        # A chains to B
        assert pm.patterns[0].chained_to_next == True
        assert pm.patterns[0].chained_from_prev == False
        
        # B is chained from A and chains to C
        assert pm.patterns[1].chained_to_next == True
        assert pm.patterns[1].chained_from_prev == True
        
        # C is chained from B but doesn't chain forward
        assert pm.patterns[2].chained_to_next == False
        assert pm.patterns[2].chained_from_prev == True
        
        # D is standalone
        assert pm.patterns[3].chained_to_next == False
        assert pm.patterns[3].chained_from_prev == False
        
        # Chain should be A-B-C
        assert pm.get_chain_patterns(0) == [0, 1, 2]
        assert pm.get_chain_patterns(1) == [0, 1, 2]
        assert pm.get_chain_patterns(2) == [0, 1, 2]
        assert pm.get_chain_patterns(3) == [3]  # D is standalone


class TestPlaybackState:
    """Test playback state management"""
    
    def test_start_playback(self):
        """start_playback should initialize playback state"""
        pm = PatternManager()
        
        pm.start_playback(3)
        
        assert pm.is_playing == True
        assert pm.playing_pattern_index == 3
        assert pm.play_position == 0
    
    def test_stop_playback(self):
        """stop_playback should reset playback state"""
        pm = PatternManager()
        
        pm.start_playback(5)
        pm.play_position = 10
        pm.stop_playback()
        
        assert pm.is_playing == False
        assert pm.play_position == 0
    
    def test_toggle_playback(self):
        """toggle_playback should toggle play state"""
        pm = PatternManager()
        
        # Toggle on
        pm.toggle_playback(2)
        assert pm.is_playing == True
        assert pm.playing_pattern_index == 2
        
        # Toggle off
        pm.toggle_playback()
        assert pm.is_playing == False
    
    def test_set_play_position(self):
        """set_play_position should wrap around pattern length"""
        pm = PatternManager()
        pm.patterns[0].set_length(8)
        pm.playing_pattern_index = 0
        
        pm.set_play_position(5)
        assert pm.play_position == 5
        
        pm.set_play_position(10)  # Should wrap
        assert pm.play_position == 2  # 10 % 8 = 2


class TestFillRate:
    """Test fill rate settings"""
    
    def test_fill_rate_default(self):
        """Default fill rate should be 4"""
        pm = PatternManager()
        assert pm.fill_rate == 4
    
    def test_fill_rate_set_valid(self):
        """set_fill_rate should set valid rates"""
        pm = PatternManager()
        
        pm.set_fill_rate(2)
        assert pm.fill_rate == 2
        
        pm.set_fill_rate(8)
        assert pm.fill_rate == 8
    
    def test_fill_rate_clamped(self):
        """set_fill_rate should clamp to valid range (2-8)"""
        pm = PatternManager()
        
        pm.set_fill_rate(1)  # Below minimum
        assert pm.fill_rate == 2
        
        pm.set_fill_rate(10)  # Above maximum
        assert pm.fill_rate == 8


class TestSwing:
    """Test swing/shuffle settings"""
    
    def test_swing_default(self):
        """Default swing should be 0"""
        pm = PatternManager()
        assert pm.swing == 0.0
    
    def test_swing_set_valid(self):
        """set_swing should set valid swing values"""
        pm = PatternManager()
        
        pm.set_swing(0.5)
        assert pm.swing == 0.5
        
        pm.set_swing(1.0)
        assert pm.swing == 1.0
    
    def test_swing_clamped(self):
        """set_swing should clamp to valid range (0-1)"""
        pm = PatternManager()
        
        pm.set_swing(-0.5)
        assert pm.swing == 0.0
        
        pm.set_swing(1.5)
        assert pm.swing == 1.0
    
    def test_get_step_time_no_swing(self):
        """get_step_time_ms with no swing should return even timing"""
        pm = PatternManager()
        pm.set_bpm(120)
        pm.set_step_rate('1/16')  # 4 steps per beat
        pm.set_swing(0.0)
        
        # At 120 BPM with 1/16 rate, each step is 125ms
        step_duration = pm.step_duration_ms
        assert abs(step_duration - 125.0) < 0.1
        
        # Step times should be evenly spaced
        t0 = pm.get_step_time_ms(0)
        t1 = pm.get_step_time_ms(1)
        t2 = pm.get_step_time_ms(2)
        t3 = pm.get_step_time_ms(3)
        
        assert abs(t0 - 0) < 0.1
        assert abs(t1 - 125) < 0.1
        assert abs(t2 - 250) < 0.1
        assert abs(t3 - 375) < 0.1
    
    def test_get_step_time_with_swing(self):
        """get_step_time_ms with swing should delay odd steps"""
        pm = PatternManager()
        pm.set_bpm(120)
        pm.set_step_rate('1/16')
        pm.set_swing(1.0)  # Maximum swing
        
        # Even steps should be at normal position
        t0 = pm.get_step_time_ms(0)
        t2 = pm.get_step_time_ms(2)
        
        assert abs(t0 - 0) < 0.1
        assert abs(t2 - 250) < 0.1  # Pair 1 starts at 250ms
        
        # Odd steps should be delayed (at 90% of pair duration with max swing)
        t1 = pm.get_step_time_ms(1)
        t3 = pm.get_step_time_ms(3)
        
        # With swing=1.0: position = 0.5 + 1.0 * 0.40 = 0.90
        # Pair duration = 250ms, so step 1 at 250 * 0.90 = 225ms
        assert t1 > 200  # Delayed significantly from 125ms
        assert t3 > 450  # Delayed from 375ms


class TestStepRate:
    """Test step rate settings"""
    
    def test_step_rate_default(self):
        """Default step rate should be 1/16"""
        pm = PatternManager()
        assert pm.step_rate == '1/16'
    
    def test_step_rate_set_valid(self):
        """set_step_rate should accept valid rates"""
        pm = PatternManager()
        
        for rate in ['1/8', '1/8T', '1/16', '1/16T', '1/32']:
            pm.set_step_rate(rate)
            assert pm.step_rate == rate
    
    def test_step_rate_invalid(self):
        """set_step_rate should ignore invalid rates"""
        pm = PatternManager()
        
        pm.set_step_rate('1/16')
        pm.set_step_rate('invalid')
        assert pm.step_rate == '1/16'  # Unchanged
    
    def test_step_duration_calculation(self):
        """Step duration should be calculated correctly for different rates"""
        pm = PatternManager()
        pm.set_bpm(120)  # 500ms per beat
        
        # 1/8: 2 steps per beat = 250ms per step
        pm.set_step_rate('1/8')
        assert abs(pm.step_duration_ms - 250.0) < 0.1
        
        # 1/16: 4 steps per beat = 125ms per step
        pm.set_step_rate('1/16')
        assert abs(pm.step_duration_ms - 125.0) < 0.1
        
        # 1/32: 8 steps per beat = 62.5ms per step
        pm.set_step_rate('1/32')
        assert abs(pm.step_duration_ms - 62.5) < 0.1


def run_tests_standalone():
    """Run tests without pytest for environments where pytest is not available"""
    import traceback
    
    test_classes = [
        TestPatternBasics,
        TestPatternChaining,
        TestPatternSerialization,
        TestPlaybackState,
        TestFillRate,
        TestSwing,
        TestStepRate,
    ]
    
    total_passed = 0
    total_failed = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        # Find all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
                total_failed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                failed_tests.append((test_class.__name__, method_name, traceback.format_exc()))
                total_failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for cls, method, error in failed_tests:
            print(f"  - {cls}.{method}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == '__main__':
    if PYTEST_AVAILABLE:
        import pytest
        exit(pytest.main([__file__, '-v']))
    else:
        exit(run_tests_standalone())