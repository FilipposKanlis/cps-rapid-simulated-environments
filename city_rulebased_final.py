#!/usr/bin/env python3
"""
Smart Home Pattern Recognition System
Monitors home sensors to learn and detect movement patterns
Modes: collect (training) | detect (recognition)
"""

from commlib.node import Node 
from commlib.transports.redis import ConnectionParameters
import os
import sys
import datetime
import json
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np
from typing import Any, Tuple, Optional, Dict, List, Deque
from pymongo import MongoClient
from collections import defaultdict, deque
import hashlib
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

UID = '684aad05aecbf959e723501c'
ENVIRONMENT = "city"
MODE = os.environ.get('MODE', 'collect').lower()

# Timing settings
WARMUP_SECONDS = 5.0
DEBOUNCE_WINDOW = 0.5

# Pattern recognition thresholds
PATTERN_MATCH_THRESHOLD = 0.6
WINDOW_SIZE = 30
SIMILARITY_THRESHOLD = 0.3
CONFIDENCE_REPORT_THRESHOLD = 0.4

# Simulation control
SIMULATION_END_VARIABLE = "simulation_end"

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

RUN_ID = f"{ENVIRONMENT}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
PATTERN_TAG = os.environ.get('PATTERN', 'all' if MODE == 'detect' else 'default')
SESSION_ID = f"{RUN_ID}_{PATTERN_TAG}_{MODE}"

print("="*60)
print(f"SMART CITY PATTERN RECOGNITION SYSTEM")
print(f"Mode: {MODE.upper()} | Session: {SESSION_ID}")
print(f"Pattern: {PATTERN_TAG}")
print("="*60)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

MONGO_URI = os.environ.get('MONGO_URI')

MONGO_DB = "thesis_db"
COLLECTION_EVENTS = f"{ENVIRONMENT}_events"
COLLECTION_PATTERNS = f"{ENVIRONMENT}_patterns"
COLLECTION_DETECTIONS = f"{ENVIRONMENT}_detections"

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB]
    mongo_events = mongo_db[COLLECTION_EVENTS]
    mongo_patterns = mongo_db[COLLECTION_PATTERNS]
    mongo_detections = mongo_db[COLLECTION_DETECTIONS]
    mongo_client.server_info()
    print(f"✓ Connected to MongoDB")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")
    sys.exit(1)

# ============================================================================
# GLOBAL STATE
# ============================================================================

START_TIME = time.time()
previous_states = {}
last_event_times = {}
shutdown_initiated = False
last_reported_confidence = {}

detection_stats = {
    "total_events": 0,
    "pattern_comparisons": [],
    "high_confidence_matches": 0,
    "low_confidence_matches": 0,
    "no_matches": 0,
    "pattern_scores": defaultdict(list),
    "final_verdict": "Unknown",
    "best_match_score": 0,
    "best_match_pattern": None
}


# ============================================================================

# ============================================================================
# MOVEMENT CONSTRAINTS (CITY) — designed to NOT interfere with the 3 patterns
# Patterns:
#   P1: Start → Parking → Stadium → Mall
#   P2: Start → Farm → Mall → Farm
#   P3: Start → Mall → Parking → Pool
# Notes:
#   - We only constrain destinations that already appear in the patterns.
#   - We avoid constraints on Farm↔Mall order except a generous temporal bound.
# ============================================================================
CONSTRAINTS = {
    "forbidden": [
        ["LA:start_alarm:activated", "LA:stadium_alarm:activated", "required_between", "LA:parking_alarm:activated"],
        ["LA:start_alarm:activated", "LA:pool_alarm:activated", "required_between", "LA:mall_alarm:activated"],
        ["LA:start_alarm:activated", "LA:pool_alarm:activated", "required_between", "LA:parking_alarm:activated"]
    ],
    "mandatory": [
        ["LA:stadium_alarm:activated", "LA:parking_alarm:activated"],
        ["LA:pool_alarm:activated", "LA:mall_alarm:activated"]
    ],
    "temporal": [
        {"pair": ["LA:parking_alarm:activated", "LA:stadium_alarm:activated"], "max_seconds": 1800},
        {"pair": ["LA:mall_alarm:activated", "LA:pool_alarm:activated"], "max_seconds": 1800},
        {"pair": ["LA:farm_alarm:activated", "LA:mall_alarm:activated"], "max_seconds": 3600},
        {"pair": ["LA:mall_alarm:activated", "LA:farm_alarm:activated"], "max_seconds": 3600}
    ],
    "hard_negatives": [
        {"invalid": "LA:start_alarm:deactivated", "requires_before": "LA:start_alarm:activated"}
    ]
}


# REFERENCE PATTERNS (CITY)
# ============================================================================

FILIPPOS_PATTERNS = {
    "pattern1": [
        "LA:start_alarm:activated",
        "LA:parking_alarm:activated",
        "LA:stadium_alarm:activated",
        "LA:mall_alarm:activated"
    ],
    "pattern2": [
        "LA:start_alarm:activated",
        "LA:farm_alarm:activated",
        "LA:mall_alarm:activated",
        "LA:farm_alarm:activated"
    ],
    "pattern3": [
        "LA:start_alarm:activated",
        "LA:mall_alarm:activated",
        "LA:parking_alarm:activated",
        "LA:pool_alarm:activated"
    ]
}

# ============================================================================
# CONSTRAINT VALIDATION
# ============================================================================

def _to_seconds(ts):
    """Convert timestamps to float seconds"""
    import datetime, time
    out = []
    for t in ts:
        if isinstance(t, (int, float)):
            out.append(float(t))
        elif isinstance(t, datetime.datetime):
            out.append(t.timestamp())
        else:
            try:
                out.append(float(t))
            except:
                out.append(time.time())
    return out

def check_constraints(buffer: List[str], timestamps: List[float], constraints: Dict) -> Tuple[str, Optional[str]]:
    """Validate movement sequence against defined constraints"""
    if not buffer:
        return ("ok", None)
    
    ts = _to_seconds(timestamps)
    event_indices = {event: i for i, event in enumerate(buffer)}
    
    # Check forbidden sequences
    for rule in constraints.get("forbidden", []):
        if len(rule) == 4 and rule[2] == "required_between":
            event_a, event_b, _, required = rule
            if event_a in event_indices and event_b in event_indices:
                idx_a = event_indices[event_a]
                idx_b = event_indices[event_b]
                if idx_a < idx_b:
                    required_found = any(buffer[i] == required for i in range(idx_a + 1, idx_b))
                    if not required_found:
                        return ("forbidden", f"{event_a} → {event_b} without {required}")
    
    # Check hard negatives
    for rule in constraints.get("hard_negatives", []):
        inv = rule.get("invalid")
        req = rule.get("requires_before")
        if inv in event_indices:
            if req not in event_indices or event_indices[req] > event_indices[inv]:
                return ("hard_negative", f"{inv} without prior {req}")
    
    # Check temporal constraints
    for entry in constraints.get("temporal", []):
        pair = entry.get("pair", [])
        if len(pair) != 2:
            continue
        event_a, event_b = pair
        max_s = entry.get("max_seconds")
        
        if event_a in event_indices and event_b in event_indices:
            idx_a = event_indices[event_a]
            idx_b = event_indices[event_b]
            if idx_b <= idx_a:
                return ("temporal_violation", f"{event_b} before {event_a}")
            
            time_diff = ts[idx_b] - ts[idx_a]
            if max_s and time_diff > max_s:
                return ("temporal_violation", f"{event_a}→{event_b} took {time_diff:.1f}s > {max_s}s")
    
    # Check mandatory sequences
    for rule in constraints.get("mandatory", []):
        if len(rule) == 4 and rule[2] == "requires_before":
            event_a, event_c, _, event_b = rule
            if event_a in event_indices and event_c in event_indices:
                idx_a = event_indices[event_a]
                idx_c = event_indices[event_c]
                if idx_a < idx_c:
                    b_found = any(buffer[i] == event_b for i in range(idx_a + 1, idx_c))
                    if not b_found:
                        return ("mandatory_violation", f"{event_a}→{event_c} requires {event_b} in between")
        elif len(rule) == 2:
            event_a, event_b = rule
            if event_a in event_indices:
                if event_b not in event_indices or event_indices[event_b] <= event_indices[event_a]:
                    return ("mandatory_pending", f"{event_a}→{event_b} not yet seen")
    
    return ("ok", None)

# ============================================================================
# PATTERN MATCHING ENGINE
# ============================================================================

class PatternMatcher:
    """Compares real-time events against reference patterns (uses ALL provided patterns)."""
    
    def __init__(self, patterns: Dict[str, List[str]]):
        # Always compare against all reference patterns
        self.patterns = patterns
        
        self.event_buffer = deque(maxlen=self._calculate_window_size())
        self.timestamps = deque(maxlen=self._calculate_window_size())
        self.comparison_results = []
        self.pattern_hashes = {name: hashlib.md5(''.join(steps).encode()).hexdigest() 
                              for name, steps in self.patterns.items() if steps}
    
    def _calculate_window_size(self) -> int:
        """Determine optimal buffer size"""
        if not self.patterns:
            return WINDOW_SIZE
        max_len = max((len(p) for p in self.patterns.values() if isinstance(p, list)), default=WINDOW_SIZE)
        return min(max(WINDOW_SIZE, int(max_len * 1.5)), 100)
    
    def add_event(self, symbol: str, timestamp: float) -> Dict[str, Any]:
        """Process new event and check pattern matches"""
        global last_reported_confidence
        
        self.event_buffer.append(symbol)
        self.timestamps.append(timestamp)
        
        buffer_list = list(self.event_buffer)
        timestamps_list = list(self.timestamps)
        
        # Validate constraints first
        status, info = check_constraints(buffer_list, timestamps_list, CONSTRAINTS)
        
        if status in ("forbidden", "hard_negative", "temporal_violation", "mandatory_violation"):
            comparison_result = {
                "timestamp": timestamp,
                "event": symbol,
                "buffer_size": len(self.event_buffer),
                "matches": {},
                "detection": "Other/ConstraintViolation",
                "reason": f"{status}: {info}"
            }
            detection_stats["no_matches"] += 1
            detection_stats["total_events"] += 1
            detection_stats["pattern_comparisons"].append(comparison_result)
            print(f"❌ Constraint violation: {info}")
            return comparison_result
        
        # Perform pattern matching
        comparison_result = self._perform_pattern_matching(symbol, timestamp, buffer_list)
        detection_stats["total_events"] += 1
        detection_stats["pattern_comparisons"].append(comparison_result)
        self._report_if_significant(comparison_result)
        self.comparison_results.append(comparison_result)
        
        return comparison_result
    
    def _perform_pattern_matching(self, symbol: str, timestamp: float, buffer_list: List[str]) -> Dict:
        """Match current buffer against all patterns"""
        comparison_result = {
            "timestamp": timestamp,
            "event": symbol,
            "buffer_size": len(self.event_buffer),
            "matches": {}
        }
        
        best_score = 0
        best_pattern = None
        
        for pattern_name, pattern_steps in self.patterns.items():
            if not pattern_steps:
                continue
            
            score = self._calculate_lcs_score(buffer_list, pattern_steps)
            comparison_result["matches"][pattern_name] = {
                "score": score,
                "total_steps": len(pattern_steps)
            }
            
            if score > best_score:
                best_score = score
                best_pattern = pattern_name
        
        if best_score > detection_stats.get("best_match_score", 0):
            detection_stats["best_match_score"] = best_score
            detection_stats["best_match_pattern"] = best_pattern
        
        # Classify detection
        if best_score >= PATTERN_MATCH_THRESHOLD:
            comparison_result["detection"] = "Filippos"
            detection_stats["high_confidence_matches"] += 1
        elif best_score >= SIMILARITY_THRESHOLD:
            comparison_result["detection"] = "Possible Filippos"
            detection_stats["low_confidence_matches"] += 1
        else:
            comparison_result["detection"] = "Other/Unknown"
            detection_stats["no_matches"] += 1
        
        comparison_result["confidence"] = best_score
        if best_pattern:
            comparison_result["best_pattern"] = best_pattern
        
        for pattern_name, match_info in comparison_result["matches"].items():
            detection_stats["pattern_scores"][pattern_name].append(match_info["score"])
        
        return comparison_result
    
    def _calculate_lcs_score(self, buffer: List[str], pattern: List[str]) -> float:
        """Calculate similarity using Longest Common Subsequence"""
        if not buffer or not pattern:
            return 0.0
        
        m, n = len(buffer), len(pattern)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if buffer[i-1] == pattern[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        base_score = lcs_length / min(m, n)
        
        # Bonus for correct ordering
        order_bonus = 0
        last_match_idx = -1
        for p_event in pattern:
            if p_event in buffer:
                idx = buffer.index(p_event)
                if idx > last_match_idx:
                    order_bonus += 0.1
                    last_match_idx = idx
        
        return min(base_score + order_bonus, 1.0)
    
    def _report_if_significant(self, comparison_result: Dict):
        """Natural-language update when confidence moves meaningfully."""
        global last_reported_confidence
        if "best_pattern" in comparison_result:
            pattern = comparison_result["best_pattern"]
            confidence = comparison_result["confidence"]
            # Top-2 summary
            pairs = sorted(
                ((k, v.get("score", 0.0)) for k, v in comparison_result.get("matches", {}).items()),
                key=lambda x: x[1], reverse=True
            )
            if confidence >= CONFIDENCE_REPORT_THRESHOLD:
                last_conf = last_reported_confidence.get(pattern, 0)
                if abs(confidence - last_conf) > 0.1:
                    top_txt = ""
                    if pairs:
                        top_pairs = ", ".join([f"{k} (~{s:.0%})" for k, s in pairs[:2]])
                        top_txt = f" Next best: {top_pairs}."
                    print(
                        f"Update: I'm currently leaning toward '{pattern}' (~{confidence:.0%} confidence)." + top_txt
                    )
                    last_reported_confidence[pattern] = confidence

# ============================================================================
# PATTERN LOADING AND SAVING
# ============================================================================

def load_reference_patterns() -> Dict[str, List[str]]:
    """Load ALL trained patterns for this ENVIRONMENT (detect mode).
    Returns {pattern_name: steps} keeping the latest doc per name.
    """
    if MODE != 'detect':
        return {}

    reference_patterns: Dict[str, List[str]] = {}
    if mongo_patterns is not None:
        try:
            cursor = mongo_patterns.find({"environment": ENVIRONMENT}).sort([("pattern_name", 1), ("timestamp", -1)])
            rows = list(cursor)
            if not rows:
                print("⚠️  No trained patterns found. Run in collect mode first.")
                return {}
            seen = set()
            for doc in rows:
                pname = doc.get("pattern_name", "unknown")
                if pname in seen:
                    continue
                steps = doc.get("steps", [])
                if steps:
                    reference_patterns[pname] = steps
                    seen.add(pname)
            for k, v in reference_patterns.items():
                print(f"✓ Loaded pattern: {k} ({len(v)} steps)")
        except Exception as e:
            print(f"Error loading patterns: {e}")
    return reference_patterns

def save_learned_pattern():
    """Save collected pattern to database"""
    if mongo_patterns is None or mongo_events is None:
        return
    
    try:
        cursor = mongo_events.find({
            "session_id": SESSION_ID,
            "type": "event"
        }).sort("timestamp", 1)
        
        events_list = list(cursor)
        events = [doc["symbol"] for doc in events_list]
        
        if not events:
            print("No events to save as pattern")
            return
        
        pattern_doc = {
            "pattern_name": PATTERN_TAG,
            "session_id": SESSION_ID,
            "timestamp": datetime.datetime.now(),
            "steps": events,
            "step_count": len(events),
            "environment": ENVIRONMENT,
            "duration_seconds": get_elapsed_time()
        }
        
        mongo_patterns.insert_one(pattern_doc)
        print(f"✅ Saved pattern '{PATTERN_TAG}' with {len(events)} steps")
        
    except Exception as e:
        print(f"Error saving pattern: {e}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def now_ts() -> str:
    """Get formatted timestamp"""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_elapsed_time() -> float:
    """Get seconds since start"""
    return time.time() - START_TIME

def should_emit_event(device_name: str, new_state: Any) -> bool:
    """Check if event should be logged (warmup, debounce, state change)"""
    if get_elapsed_time() < WARMUP_SECONDS:
        return False
    
    if previous_states.get(device_name) == new_state:
        return False
    
    current_time = time.time()
    last_time = last_event_times.get(device_name, 0)
    if current_time - last_time < DEBOUNCE_WINDOW:
        return False
    
    last_event_times[device_name] = current_time
    return True

def create_symbol(device_type: str, device_name: str, state: str) -> str:
    """Create event symbol like 'L:light_kitchen:on'"""
    return f"{device_type}:{device_name}:{state}"

def insert_session_marker(action: str):
    """Mark session start/end in database and store to MongoDB."""
    marker = {
        "type": "session_marker",
        "session_id": SESSION_ID,
        "run_id": RUN_ID,
        "pattern": PATTERN_TAG,
        "env": ENVIRONMENT,
        "mode": MODE,
        "action": action,
        "timestamp": datetime.datetime.now()
    }
    if mongo_events is not None:
        try:
            mongo_events.insert_one(marker)
            print(f"📍 Session marker saved: {action}")
        except Exception as e:
            print(f"Error inserting marker: {e}")


def create_compact_detection_report():
    """Generate a beautiful, modern visual detection report"""
    if MODE != 'detect' or not detection_stats['pattern_comparisons']:
        return
    
    # Modern color palette
    COLORS = {
        'bg': '#1e1e2e',           # Dark background
        'card': '#2a2a3e',         # Card background
        'text': '#ffffff',         # White text
        'text_secondary': '#a6adc8', # Secondary text
        'success': '#a6e3a1',      # Green
        'warning': '#f9e2af',      # Yellow
        'danger': '#f38ba8',       # Red
        'primary': '#89b4fa',      # Blue
        'purple': '#cba6f7',       # Purple
        'accent': '#94e2d5',       # Teal
        'grid': '#45475a'          # Grid lines
    }
    
    # Create figure with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10), facecolor=COLORS['bg'])
    gs = GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.4, 
                  left=0.06, right=0.94, top=0.88, bottom=0.08)
    
    # Determine verdict color
    verdict = detection_stats.get('final_verdict', 'Unknown')
    if 'FILIPPOS DETECTED' in verdict:
        verdict_color = COLORS['success']
        verdict_emoji = "✅"
    elif 'POSSIBLY' in verdict:
        verdict_color = COLORS['warning']
        verdict_emoji = "🤔"
    else:
        verdict_color = COLORS['danger']
        verdict_emoji = "❌"
    
    # Modern header with gradient effect
    header_text = fig.text(0.5, 0.96, 'SMART CITY PATTERN RECOGNITION', 
                           ha='center', fontsize=20, fontweight='bold', 
                           color=COLORS['text'], fontfamily='monospace')
    
    fig.text(0.5, 0.92, f'{verdict_emoji} {verdict}', 
             ha='center', fontsize=18, fontweight='bold', 
             color=verdict_color, fontfamily='sans-serif')
    
    # 1. Key Metrics Cards (top left - spanning 2 columns)
    ax_metrics = fig.add_subplot(gs[0:2, :2])
    ax_metrics.set_facecolor(COLORS['card'])
    ax_metrics.axis('off')
    
    total = detection_stats['total_events']
    high_conf = detection_stats['high_confidence_matches']
    low_conf = detection_stats['low_confidence_matches']
    no_match = detection_stats['no_matches']
    duration = get_elapsed_time()
    
    # Create metric cards with icons
    metrics_y = 0.8
    metrics_spacing = 0.2
    
    # Duration card
    rect1 = FancyBboxPatch((0.05, metrics_y), 0.4, 0.15, 
                           boxstyle="round,pad=0.02", 
                           facecolor=COLORS['primary'], alpha=0.3,
                           edgecolor=COLORS['primary'], linewidth=2)
    ax_metrics.add_patch(rect1)
    ax_metrics.text(0.25, metrics_y + 0.1, f'{duration:.1f}s', 
                   ha='center', fontsize=16, fontweight='bold', color=COLORS['primary'])
    ax_metrics.text(0.25, metrics_y + 0.04, 'Duration', 
                   ha='center', fontsize=10, color=COLORS['text_secondary'])
    
    # Events card
    rect2 = FancyBboxPatch((0.5, metrics_y), 0.4, 0.15,
                           boxstyle="round,pad=0.02",
                           facecolor=COLORS['accent'], alpha=0.3,
                           edgecolor=COLORS['accent'], linewidth=2)
    ax_metrics.add_patch(rect2)
    ax_metrics.text(0.7, metrics_y + 0.1, f'{total}', 
                   ha='center', fontsize=16, fontweight='bold', color=COLORS['accent'])
    ax_metrics.text(0.7, metrics_y + 0.04, 'Total Events', 
                   ha='center', fontsize=10, color=COLORS['text_secondary'])
    
    # Match statistics
    metrics_y = 0.45
    for i, (label, value, color) in enumerate([
        ('High Confidence', high_conf, COLORS['success']),
        ('Low Confidence', low_conf, COLORS['warning']),
        ('No Match', no_match, COLORS['danger'])
    ]):
        y_pos = metrics_y - i * 0.15
        pct = value/total*100 if total > 0 else 0
        
        # Progress bar
        bar_width = 0.6
        bar_x = 0.15
        
        # Background bar
        bg_rect = FancyBboxPatch((bar_x, y_pos), bar_width, 0.08,
                                 boxstyle="round,pad=0.005",
                                 facecolor=COLORS['card'], alpha=0.5,
                                 edgecolor=color, linewidth=1)
        ax_metrics.add_patch(bg_rect)
        
        # Fill bar
        fill_width = bar_width * (pct / 100)
        fill_rect = FancyBboxPatch((bar_x, y_pos), fill_width, 0.08,
                                   boxstyle="round,pad=0.005",
                                   facecolor=color, alpha=0.6,
                                   edgecolor='none')
        ax_metrics.add_patch(fill_rect)
        
        ax_metrics.text(bar_x - 0.02, y_pos + 0.04, label, 
                       ha='right', va='center', fontsize=9, color=COLORS['text_secondary'])
        ax_metrics.text(bar_x + bar_width + 0.02, y_pos + 0.04, f'{value} ({pct:.0f}%)', 
                       ha='left', va='center', fontsize=9, color=color, fontweight='bold')
    
    # Best match info
    if detection_stats.get('best_match_pattern'):
        ax_metrics.text(0.5, 0.05, 
                       f"🎯 Best Match: {detection_stats['best_match_pattern']} ({detection_stats['best_match_score']:.0%})",
                       ha='center', fontsize=11, color=COLORS['purple'], fontweight='bold')
    
    # 2. Beautiful Donut Chart (top middle)
    ax_pie = fig.add_subplot(gs[0:2, 2:4])
    ax_pie.set_facecolor(COLORS['bg'])
    
    sizes = [high_conf, low_conf, no_match]
    labels = ['High', 'Low', 'None']
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    
    if sum(sizes) > 0:
        # Create donut chart
        wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors,
                                               autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
                                               startangle=90, 
                                               wedgeprops=dict(width=0.5, edgecolor=COLORS['bg']),
                                               textprops={'fontsize': 10, 'color': COLORS['text']})
        
        # Add center circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.50, fc=COLORS['bg'])
        ax_pie.add_artist(centre_circle)
        
        ax_pie.text(0, 0, 'Detection\nBreakdown', ha='center', va='center', 
                   fontsize=10, color=COLORS['text_secondary'])
    
    ax_pie.set_title('Pattern Recognition Distribution', fontsize=11, color=COLORS['text'], pad=10)
    
    # 3. Pattern Scores with modern bars (top right)
    ax_patterns = fig.add_subplot(gs[0:2, 4:])
    ax_patterns.set_facecolor(COLORS['card'])
    
    if 'session_metrics' in detection_stats and detection_stats['session_metrics']:
        patterns = sorted(detection_stats['session_metrics'].items(), 
                         key=lambda x: x[1]['f1_overall'], reverse=True)[:4]
        
        pattern_names = [p[0][:12] + '..' if len(p[0]) > 14 else p[0] for p in patterns]
        f1_scores = [p[1]['f1_overall'] for p in patterns]
        
        y_pos = np.arange(len(pattern_names))
        
        # Create gradient-like effect with bars
        for i, (name, score) in enumerate(zip(pattern_names, f1_scores)):
            is_best = patterns[i][0] == detection_stats.get('best_match_pattern')
            color = COLORS['success'] if is_best else COLORS['primary']
            
            # Background bar
            ax_patterns.barh(i, 1, color=COLORS['bg'], alpha=0.3, height=0.6)
            # Actual score bar
            bar = ax_patterns.barh(i, score, color=color, alpha=0.8, height=0.6)
            
            # Add glow effect for best match
            if is_best:
                ax_patterns.barh(i, score, color=color, alpha=0.3, height=0.8)
            
            # Score text
            ax_patterns.text(score + 0.03, i, f'{score:.2f}', 
                           va='center', fontsize=10, color=color, fontweight='bold')
        
        ax_patterns.set_yticks(y_pos)
        ax_patterns.set_yticklabels(pattern_names, fontsize=10, color=COLORS['text'])
        ax_patterns.set_xlim(0, 1.1)
        ax_patterns.set_xlabel('F1 Score', fontsize=10, color=COLORS['text_secondary'])
        ax_patterns.set_title('Pattern Match Ranking', fontsize=11, color=COLORS['text'], pad=10)
        ax_patterns.grid(True, alpha=0.1, color=COLORS['grid'])
        ax_patterns.spines['top'].set_visible(False)
        ax_patterns.spines['right'].set_visible(False)
        ax_patterns.spines['left'].set_color(COLORS['grid'])
        ax_patterns.spines['bottom'].set_color(COLORS['grid'])
    
    # 4. Smooth Confidence Timeline (middle row)
    ax_timeline = fig.add_subplot(gs[2, :])
    ax_timeline.set_facecolor(COLORS['card'])
    
    if detection_stats['pattern_comparisons']:
        timestamps = [(comp['timestamp'] - START_TIME) for comp in detection_stats['pattern_comparisons']]
        confidences = [comp.get('confidence', 0) for comp in detection_stats['pattern_comparisons']]
        
        # Smooth the line with interpolation
        if len(timestamps) > 3:
            from scipy.interpolate import make_interp_spline
            timestamps_smooth = np.linspace(min(timestamps), max(timestamps), 300)
            spl = make_interp_spline(timestamps, confidences, k=min(3, len(timestamps)-1))
            confidences_smooth = spl(timestamps_smooth)
            
            # Main confidence line with gradient fill
            ax_timeline.plot(timestamps_smooth, confidences_smooth, 
                           color=COLORS['primary'], linewidth=2, alpha=0.9)
            ax_timeline.fill_between(timestamps_smooth, confidences_smooth, 
                                    alpha=0.3, color=COLORS['primary'])
        else:
            ax_timeline.plot(timestamps, confidences, 
                           color=COLORS['primary'], linewidth=2, alpha=0.9)
            ax_timeline.fill_between(timestamps, confidences, 
                                    alpha=0.3, color=COLORS['primary'])
        
        # Threshold lines with labels
        ax_timeline.axhline(y=PATTERN_MATCH_THRESHOLD, color=COLORS['success'], 
                          linestyle='--', alpha=0.5, linewidth=1.5)
        ax_timeline.text(max(timestamps) * 0.98, PATTERN_MATCH_THRESHOLD + 0.02, 
                        'High', ha='right', fontsize=9, color=COLORS['success'])
        
        ax_timeline.axhline(y=SIMILARITY_THRESHOLD, color=COLORS['warning'], 
                          linestyle='--', alpha=0.5, linewidth=1.5)
        ax_timeline.text(max(timestamps) * 0.98, SIMILARITY_THRESHOLD + 0.02, 
                        'Low', ha='right', fontsize=9, color=COLORS['warning'])
        
        ax_timeline.set_xlabel('Time (seconds)', fontsize=10, color=COLORS['text_secondary'])
        ax_timeline.set_ylabel('Confidence', fontsize=10, color=COLORS['text_secondary'])
        ax_timeline.set_title('Real-time Pattern Confidence Evolution', fontsize=11, color=COLORS['text'], pad=10)
        ax_timeline.set_ylim(-0.05, 1.05)
        ax_timeline.grid(True, alpha=0.1, color=COLORS['grid'], linestyle='-')
        ax_timeline.set_facecolor(COLORS['card'])
        
        # Style the spines
        for spine in ax_timeline.spines.values():
            spine.set_color(COLORS['grid'])
            spine.set_linewidth(0.5)
    
    # 5. Event Stream Heatmap (bottom row)
    ax_events = fig.add_subplot(gs[3, :])
    ax_events.set_facecolor(COLORS['card'])
    
    if detection_stats['pattern_comparisons']:
        max_events = 100
        events_data = detection_stats['pattern_comparisons'][-max_events:]
        
        # Create heatmap-like visualization
        for i, comp in enumerate(events_data):
            confidence = comp.get('confidence', 0)
            
            # Color based on confidence with gradient
            if confidence >= PATTERN_MATCH_THRESHOLD:
                color = COLORS['success']
                height = 0.8
            elif confidence >= SIMILARITY_THRESHOLD:
                color = COLORS['warning']
                height = 0.6
            else:
                color = COLORS['danger']
                height = 0.4
            
            time_pos = i / len(events_data)
            
            # Add vertical bar with glow effect
            rect = patches.Rectangle((time_pos, 0.5 - height/2), 0.008, height,
                                    facecolor=color, alpha=0.8, edgecolor='none')
            ax_events.add_patch(rect)
            
            # Add glow
            glow = patches.Rectangle((time_pos - 0.002, 0.5 - height/2 - 0.05), 
                                    0.012, height + 0.1,
                                    facecolor=color, alpha=0.2, edgecolor='none')
            ax_events.add_patch(glow)
        
        ax_events.set_xlim(0, 1)
        ax_events.set_ylim(0, 1)
        ax_events.set_xlabel('Session Timeline', fontsize=10, color=COLORS['text_secondary'])
        ax_events.set_title('Event Detection Stream Visualization', fontsize=11, color=COLORS['text'], pad=10)
        ax_events.set_yticks([])
        
        # Add legend
        legend_elements = [
            patches.Patch(color=COLORS['success'], label='High Match', alpha=0.8),
            patches.Patch(color=COLORS['warning'], label='Low Match', alpha=0.8),
            patches.Patch(color=COLORS['danger'], label='No Match', alpha=0.8)
        ]
        ax_events.legend(handles=legend_elements, loc='upper right', 
                        fontsize=9, framealpha=0.3, edgecolor=COLORS['grid'])
        
        # Style
        ax_events.spines['top'].set_visible(False)
        ax_events.spines['right'].set_visible(False)
        ax_events.spines['left'].set_visible(False)
        ax_events.spines['bottom'].set_color(COLORS['grid'])
        ax_events.spines['bottom'].set_linewidth(0.5)
    
    # Footer with session info
    footer_text = (f"Session: {SESSION_ID} | Environment: {ENVIRONMENT} | "
                  f"Pattern: {PATTERN_TAG} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=9, 
            color=COLORS['text_secondary'], fontfamily='monospace')
    
    # Add subtle branding/version
    fig.text(0.98, 0.02, 'v1.0', ha='right', fontsize=8, 
            color=COLORS['text_secondary'], alpha=0.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    # Save with high quality
    report_filename = f"detection_report_{SESSION_ID}.png"
    plt.savefig(report_filename, dpi=200, bbox_inches='tight', 
               facecolor=COLORS['bg'], edgecolor='none')
    print(f"\n🎨 Beautiful visual report saved: {report_filename}")
    
    # Also create a PDF version
    try:
        pdf_filename = f"detection_report_{SESSION_ID}.pdf"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', 
                   facecolor=COLORS['bg'], edgecolor='none')
        print(f"📄 PDF report saved: {pdf_filename}")
    except Exception as e:
        print(f"Could not save PDF version: {e}")
    
    # Display
    plt.show(block=False)
    
    return report_filename
    """Generate a compact single-page visual detection report"""
    if MODE != 'detect' or not detection_stats['pattern_comparisons']:
        return
    
    # Create figure with custom grid layout for compact display
    fig = plt.figure(figsize=(16, 9))  # 16:9 aspect ratio, fits most screens
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main title with verdict
    verdict = detection_stats.get('final_verdict', 'Unknown')
    verdict_color = '#27ae60' if 'FILIPPOS DETECTED' in verdict else '#f39c12' if 'POSSIBLY' in verdict else '#e74c3c'
    
    fig.suptitle(f'Pattern Detection Report - {SESSION_ID}', fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, verdict, ha='center', fontsize=16, fontweight='bold', color=verdict_color)
    
    # 1. Key Metrics Summary (top left)
    ax_metrics = fig.add_subplot(gs[0, :2])
    ax_metrics.axis('off')
    
    total = detection_stats['total_events']
    high_conf = detection_stats['high_confidence_matches']
    low_conf = detection_stats['low_confidence_matches']
    no_match = detection_stats['no_matches']
    duration = get_elapsed_time()
    
    metrics_text = f"""
    Duration: {duration:.1f}s | Total Events: {total}
    High Confidence: {high_conf} ({high_conf/total*100:.1f}%)
    Low Confidence: {low_conf} ({low_conf/total*100:.1f}%)
    No Match: {no_match} ({no_match/total*100:.1f}%)
    """
    
    if detection_stats.get('best_match_pattern'):
        metrics_text += f"\nBest Match: {detection_stats['best_match_pattern']} ({detection_stats['best_match_score']:.1%})"
    
    ax_metrics.text(0.1, 0.5, metrics_text.strip(), transform=ax_metrics.transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
    
    # 2. Detection Distribution Pie (top right)
    ax_pie = fig.add_subplot(gs[0, 2])
    sizes = [high_conf, low_conf, no_match]
    labels = ['High', 'Low', 'None']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors, 
                                               autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
                                               startangle=90, textprops={'fontsize': 9})
        ax_pie.set_title('Detection Distribution', fontsize=10)
    
    # 3. Pattern Scores Comparison (top right continued)
    ax_patterns = fig.add_subplot(gs[0, 3])
    if 'session_metrics' in detection_stats and detection_stats['session_metrics']:
        patterns = list(detection_stats['session_metrics'].keys())[:4]  # Limit to 4 patterns for space
        f1_scores = [detection_stats['session_metrics'][p]['f1_overall'] for p in patterns]
        
        # Shorten pattern names if needed
        short_patterns = [p[:8] + '..' if len(p) > 10 else p for p in patterns]
        
        bars = ax_patterns.barh(range(len(patterns)), f1_scores, color=['#27ae60' if p == detection_stats.get('best_match_pattern') else '#3498db' for p in patterns])
        ax_patterns.set_yticks(range(len(patterns)))
        ax_patterns.set_yticklabels(short_patterns, fontsize=9)
        ax_patterns.set_xlim(0, 1)
        ax_patterns.set_xlabel('F1 Score', fontsize=9)
        ax_patterns.set_title('Pattern Matches', fontsize=10)
        
        for i, v in enumerate(f1_scores):
            ax_patterns.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)
    
    # 4. Confidence Timeline (middle row, spanning 3 columns)
    ax_timeline = fig.add_subplot(gs[1, :3])
    if detection_stats['pattern_comparisons']:
        timestamps = [(comp['timestamp'] - START_TIME) for comp in detection_stats['pattern_comparisons']]
        confidences = [comp.get('confidence', 0) for comp in detection_stats['pattern_comparisons']]
        
        ax_timeline.plot(timestamps, confidences, 'b-', alpha=0.7, linewidth=1.5)
        ax_timeline.fill_between(timestamps, confidences, alpha=0.3)
        ax_timeline.axhline(y=PATTERN_MATCH_THRESHOLD, color='g', linestyle='--', alpha=0.5, linewidth=1)
        ax_timeline.axhline(y=SIMILARITY_THRESHOLD, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        
        ax_timeline.set_xlabel('Time (seconds)', fontsize=9)
        ax_timeline.set_ylabel('Confidence', fontsize=9)
        ax_timeline.set_title('Pattern Match Confidence Over Time', fontsize=10)
        ax_timeline.set_ylim(0, 1)
        ax_timeline.grid(True, alpha=0.3)
        ax_timeline.tick_params(labelsize=8)
    
    # 5. Precision vs Coverage (middle right)
    ax_precision = fig.add_subplot(gs[1, 3])
    if 'session_metrics' in detection_stats and detection_stats['session_metrics']:
        patterns = list(detection_stats['session_metrics'].keys())[:3]  # Top 3
        precision = [detection_stats['session_metrics'][p]['precision'] for p in patterns]
        coverage = [detection_stats['session_metrics'][p]['coverage'] for p in patterns]
        
        x = np.arange(len(patterns))
        width = 0.35
        
        ax_precision.bar(x - width/2, precision, width, label='Prec.', color='#3498db', alpha=0.8)
        ax_precision.bar(x + width/2, coverage, width, label='Cov.', color='#9b59b6', alpha=0.8)
        
        ax_precision.set_ylabel('Score', fontsize=9)
        ax_precision.set_title('Precision/Coverage', fontsize=10)
        ax_precision.set_xticks(x)
        ax_precision.set_xticklabels([p[:6] for p in patterns], fontsize=8, rotation=45)
        ax_precision.set_ylim(0, 1.1)
        ax_precision.legend(fontsize=8, loc='upper right')
        ax_precision.tick_params(labelsize=8)
    
    # 6. Event Stream Visualization (bottom row)
    ax_events = fig.add_subplot(gs[2, :])
    if detection_stats['pattern_comparisons']:
        # Take last N events that fit nicely
        max_events = 60
        events_data = detection_stats['pattern_comparisons'][-max_events:]
        
        for i, comp in enumerate(events_data):
            color = '#2ecc71' if comp.get('confidence', 0) >= PATTERN_MATCH_THRESHOLD else \
                   '#f39c12' if comp.get('confidence', 0) >= SIMILARITY_THRESHOLD else '#e74c3c'
            
            time_pos = (comp['timestamp'] - START_TIME) / duration if duration > 0 else 0
            ax_events.add_patch(plt.Rectangle((time_pos, 0), 0.015, 1, color=color, alpha=0.6))
        
        ax_events.set_xlim(0, 1)
        ax_events.set_ylim(0, 1)
        ax_events.set_xlabel('Session Progress', fontsize=9)
        ax_events.set_title('Event Stream (Green=High Match, Orange=Low Match, Red=No Match)', fontsize=10)
        ax_events.set_yticks([])
        ax_events.tick_params(labelsize=8)
        
        # Add time markers
        time_markers = np.linspace(0, duration, 5)
        ax_events.set_xticks(np.linspace(0, 1, 5))
        ax_events.set_xticklabels([f'{t:.0f}s' for t in time_markers], fontsize=8)
    
    # Add session info footer
    footer_text = f"Session: {SESSION_ID} | Environment: {ENVIRONMENT} | Mode: {MODE} | Pattern: {PATTERN_TAG} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, color='gray')
    
    # Adjust layout to be compact
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    # Save as high-quality image
    report_filename = f"detection_report_{SESSION_ID}.png"
    plt.savefig(report_filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n📊 Compact visual report saved: {report_filename}")
    
    # Optionally display (will open in a window)
    plt.show(block=False)
    
    # Also create a PDF version if you have the backend
    try:
        pdf_filename = f"detection_report_{SESSION_ID}.pdf"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📄 PDF report saved: {pdf_filename}")
    except Exception as e:
        print(f"Could not save PDF version: {e}")
    
    return report_filename
    if MODE != 'detect' or not detection_stats['pattern_comparisons']:
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Pattern Detection Report - Session: {SESSION_ID}', fontsize=16, fontweight='bold')
    
    # 1. Pattern Match Scores Over Time
    ax1 = plt.subplot(2, 3, 1)
    if detection_stats['pattern_comparisons']:
        timestamps = [comp['timestamp'] - START_TIME for comp in detection_stats['pattern_comparisons']]
        best_scores = [comp.get('confidence', 0) for comp in detection_stats['pattern_comparisons']]
        
        ax1.plot(timestamps, best_scores, 'b-', alpha=0.7, linewidth=2)
        ax1.axhline(y=PATTERN_MATCH_THRESHOLD, color='g', linestyle='--', label=f'High Confidence ({PATTERN_MATCH_THRESHOLD})')
        ax1.axhline(y=SIMILARITY_THRESHOLD, color='orange', linestyle='--', label=f'Low Confidence ({SIMILARITY_THRESHOLD})')
        ax1.fill_between(timestamps, best_scores, alpha=0.3)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Match Confidence')
        ax1.set_title('Pattern Match Confidence Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Pattern Comparison Bar Chart
    ax2 = plt.subplot(2, 3, 2)
    if 'session_metrics' in detection_stats and detection_stats['session_metrics']:
        patterns = list(detection_stats['session_metrics'].keys())
        f1_scores = [metrics['f1_overall'] for metrics in detection_stats['session_metrics'].values()]
        
        bars = ax2.bar(patterns, f1_scores)
        # Color best match differently
        if detection_stats['best_match_pattern']:
            for i, pattern in enumerate(patterns):
                if pattern == detection_stats['best_match_pattern']:
                    bars[i].set_color('green')
                else:
                    bars[i].set_color('skyblue')
        
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Pattern Match Scores')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    # 3. Detection Distribution Pie Chart
    ax3 = plt.subplot(2, 3, 3)
    sizes = [
        detection_stats['high_confidence_matches'],
        detection_stats['low_confidence_matches'],
        detection_stats['no_matches']
    ]
    labels = ['High Confidence', 'Low Confidence', 'No Match']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90)
        ax3.set_title('Detection Distribution')
    
    # 4. Coverage and Precision Metrics
    ax4 = plt.subplot(2, 3, 4)
    if 'session_metrics' in detection_stats and detection_stats['session_metrics']:
        patterns = list(detection_stats['session_metrics'].keys())
        precision = [m['precision'] for m in detection_stats['session_metrics'].values()]
        coverage = [m['coverage'] for m in detection_stats['session_metrics'].values()]
        
        x = np.arange(len(patterns))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, precision, width, label='Precision', color='#3498db')
        bars2 = ax4.bar(x + width/2, coverage, width, label='Coverage', color='#9b59b6')
        
        ax4.set_xlabel('Patterns')
        ax4.set_ylabel('Score')
        ax4.set_title('Precision vs Coverage')
        ax4.set_xticks(x)
        ax4.set_xticklabels(patterns)
        ax4.legend()
        ax4.set_ylim(0, 1.1)
    
    # 5. Event Timeline
    ax5 = plt.subplot(2, 3, (5, 6))
    if detection_stats['pattern_comparisons']:
        events = []
        times = []
        colors_timeline = []
        
        for comp in detection_stats['pattern_comparisons'][-50:]:  # Last 50 events
            events.append(comp.get('event', '').split(':')[1] if ':' in comp.get('event', '') else comp.get('event', ''))
            times.append(comp['timestamp'] - START_TIME)
            
            if comp.get('confidence', 0) >= PATTERN_MATCH_THRESHOLD:
                colors_timeline.append('#2ecc71')
            elif comp.get('confidence', 0) >= SIMILARITY_THRESHOLD:
                colors_timeline.append('#f39c12')
            else:
                colors_timeline.append('#e74c3c')
        
        y_positions = range(len(events))
        ax5.barh(y_positions, [1]*len(events), left=times, color=colors_timeline, alpha=0.6)
        ax5.set_yticks(y_positions[::max(1, len(events)//10)])
        ax5.set_yticklabels(events[::max(1, len(events)//10)], fontsize=8)
        ax5.set_xlabel('Time (seconds)')
        ax5.set_title('Event Timeline (Color = Confidence Level)')
        ax5.grid(True, alpha=0.3, axis='x')
    
    # Add verdict text box
    verdict_text = f"Final Verdict: {detection_stats['final_verdict']}\n"
    if detection_stats['best_match_pattern']:
        verdict_text += f"Best Match: {detection_stats['best_match_pattern']} ({detection_stats['best_match_score']:.1%})"
    
    fig.text(0.5, 0.02, verdict_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save the figure
    report_filename = f"detection_report_{SESSION_ID}.png"
    plt.savefig(report_filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visual report saved: {report_filename}")
    
    # Optionally display (will open in a window)
    plt.show(block=False)


def generate_html_report():
    """Generate an HTML report with styled results"""
    if MODE != 'detect':
        return
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detection Report - {SESSION_ID}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            .verdict {{ font-size: 28px; font-weight: bold; margin: 20px 0; }}
            .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px; min-width: 150px; }}
            .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
            .pattern-row {{ padding: 10px; margin: 5px 0; background: #f8f9fa; border-left: 4px solid #3498db; }}
            .best-match {{ border-left-color: #27ae60; background: #e8f8f5; }}
            .progress-bar {{ width: 100%; height: 30px; background: #ecf0f1; border-radius: 15px; overflow: hidden; }}
            .progress-fill {{ height: 100%; background: linear-gradient(90deg, #3498db, #2ecc71); transition: width 0.3s; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Smart Home Pattern Detection Report</h1>
            <div>Session: {SESSION_ID}</div>
            <div>Duration: {get_elapsed_time():.1f} seconds</div>
            <div class="verdict">{detection_stats['final_verdict']}</div>
        </div>
        
        <div class="card">
            <h2>Detection Summary</h2>
            <div class="metric">
                <div class="metric-value">{detection_stats['total_events']}</div>
                <div class="metric-label">Total Events</div>
            </div>
            <div class="metric">
                <div class="metric-value">{detection_stats['high_confidence_matches']}</div>
                <div class="metric-label">High Confidence</div>
            </div>
            <div class="metric">
                <div class="metric-value">{detection_stats['low_confidence_matches']}</div>
                <div class="metric-label">Low Confidence</div>
            </div>
            <div class="metric">
                <div class="metric-value">{detection_stats['no_matches']}</div>
                <div class="metric-label">No Match</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Pattern Comparison</h2>
    """
    
    if 'session_metrics' in detection_stats:
        for pattern_name, metrics in sorted(detection_stats['session_metrics'].items(), 
                                           key=lambda x: x[1].get('f1_overall', 0), reverse=True):
            is_best = pattern_name == detection_stats.get('best_match_pattern')
            class_name = 'pattern-row best-match' if is_best else 'pattern-row'
            
            html_content += f"""
            <div class="{class_name}">
                <h3>{pattern_name} {' (Best Match)' if is_best else ''}</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {metrics['f1_overall']*100:.1f}%"></div>
                </div>
                <p>F1 Score: {metrics['f1_overall']:.2%} | Precision: {metrics['precision']:.2%} | Coverage: {metrics['coverage']:.2%}</p>
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_filename = f"detection_report_{SESSION_ID}.html"
    with open(html_filename, 'w') as f:
        f.write(html_content)
    
    print(f"📄 HTML report saved: {html_filename}")
    
    # Optionally open in browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(html_filename)}")

    
    if mongo_events is not None:
        try:
            mongo_events.insert_one(marker)
            print(f"📍 Session marker: {action}")
        except Exception as e:
            print(f"Error inserting marker: {e}")

# ============================================================================
# EVENT PROCESSING
# ============================================================================

def log_event(device_name: str, description: str, device_type: str, state: str) -> None:
    """Log sensor event and run pattern detection"""
    ts = now_ts()
    symbol = create_symbol(device_type, device_name, state)
    
    comparison = None
    if MODE == 'detect' and pattern_matcher:
        comparison = pattern_matcher.add_event(symbol, time.time())
    
    print(f"[{ts}] {device_name}: {description} -> {symbol}")

    if mongo_events is not None:
        try:
            doc = {
                "timestamp": datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": get_elapsed_time(),
                "session_id": SESSION_ID,
                "run_id": RUN_ID,
                "pattern": PATTERN_TAG,
                "env": ENVIRONMENT,
                "mode": MODE,
                "device": device_name,
                "event": description,
                "symbol": symbol,
                "device_type": device_type,
                "state": state,
                "type": "event"
            }
            
            if comparison:
                doc["detection_result"] = comparison
            
            mongo_events.insert_one(doc)
        except Exception as e:
            print(f"MongoDB error: {e}")

def infer_device_name(topic: str, data: dict) -> str:
    """Extract device name from MQTT topic"""
    if topic.startswith('appcreator.variables'):
        return data.get('name', 'unknown_variable')
    
    parts = topic.split('.')
    if len(parts) >= 2 and parts[-1] in ('state', 'data'):
        return parts[-2]
    return parts[-1] if parts else 'unknown'

def normalize_on_off(value: Any) -> Optional[str]:
    """Convert various boolean representations to on/off"""
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, (int, float)):
        return 'on' if value != 0 else 'off'
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('on', 'true', '1', 'open', 'active'):
            return 'on'
        if normalized in ('off', 'false', '0', 'closed', 'inactive'):
            return 'off'
    return None

def extract_alarm_state(value: Any) -> Optional[bool]:
    """Extract alarm activation state from various formats"""
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict) and 'value' in value:
        if isinstance(value['value'], list):
            return len(value['value']) > 0
    return None

def derive_event(topic: str, data: dict) -> Optional[Tuple[str, str, Any, str]]:
    """Parse MQTT data into structured event"""
    device_name = infer_device_name(topic, data)
    
    # Determine device type from topic
    device_type = "UNKNOWN"
    if "linear_alarm" in topic:
        device_type = "LA"
    elif "area_alarm" in topic:
        device_type = "AA"
    elif "switch" in topic:
        device_type = "SW"
    elif "light" in topic or "leds" in topic:
        device_type = "L"
    
    value = data.get('value', data)
    
    # Check switch/light state
    if isinstance(value, dict) and 'state' in value:
        state = normalize_on_off(value['state'])
        if state:
            return (device_type, device_name, state, f"turned {state}")
    
    # Check alarm state
    alarm_active = extract_alarm_state(data.get('value'))
    if alarm_active is None:
        alarm_active = extract_alarm_state(data)
    
    if alarm_active is not None:
        state = 'activated' if alarm_active else 'deactivated'
        return (device_type, device_name, state, state)
    
    # Fallback
    state = normalize_on_off(data.get('value'))
    if state:
        return (device_type, device_name, state, f"turned {state}")
    
    return None

# ============================================================================
# MQTT CALLBACKS
# ============================================================================

def make_callback(topic: str):
    """Create MQTT callback for topic"""
    
    if topic == 'appcreator.variables':
        def variable_callback(payload: dict):
            try:
                data = payload
                if isinstance(payload, str):
                    data = json.loads(payload)
                
                var_name = data.get('name', '')
                var_value = data.get('value')
                
                # Check for simulation end signal
                if var_name == SIMULATION_END_VARIABLE:
                    if var_value in [True, 'True', 'true', 1, '1']:
                        print("\n" + "="*60)
                        print("🛑 SIMULATION ENDED")
                        print("="*60)
                        shutdown_gracefully()
                        return
                
            except Exception as e:
                if "Collection objects do not implement truth value" not in str(e):
                    print(f"Variable callback error: {e}")
        
        return variable_callback
    
    # Regular sensor callback
    def callback(payload: dict):
        try:
            data = payload
            if isinstance(payload, str):
                try:
                    data = json.loads(payload)
                except:
                    data = {'value': payload}

            result = derive_event(topic, data)
            if result:
                device_type, device_name, state, description = result
                
                if should_emit_event(device_name, state):
                    log_event(device_name, description, device_type, state)
                    previous_states[device_name] = state
                    
        except Exception as e:
            print(f"Callback error: {e}")
    
    return callback

# ============================================================================
# SUMMARY AND SHUTDOWN
# ============================================================================


# ============================
# Coverage- and Order-aware metrics
# ============================
def _collapse_runs(tokens: List[str]) -> List[str]:
    """Collapse consecutive duplicates so repeated 'still active' doesn't distort order."""
    if not tokens:
        return tokens
    out = [tokens[0]]
    for t in tokens[1:]:
        if t != out[-1]:
            out.append(t)
    return out

def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [0]*(n+1)
    for i in range(m):
        prev = 0
        for j in range(n):
            temp = dp[j+1]
            if a[i] == b[j]:
                dp[j+1] = prev + 1
            else:
                dp[j+1] = max(dp[j+1], dp[j])
            prev = temp
    return dp[-1]

def _levenshtein(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def compute_sequence_metrics(ref_tokens: List[str], obs_tokens: List[str]) -> Dict[str, float]:
    """Return len_ref, len_obs, lcs, precision, coverage, f1_overall, order_agreement."""
    ref = _collapse_runs(ref_tokens)
    obs = _collapse_runs(obs_tokens)
    len_ref, len_obs = len(ref), len(obs)
    if len_ref == 0 or len_obs == 0:
        return {"len_ref": len_ref, "len_obs": len_obs, "lcs": 0, "precision": 0.0, "coverage": 0.0, "f1_overall": 0.0, "order_agreement": 0.0}
    L = _lcs_length(ref, obs)
    precision = L/len_ref
    coverage  = L/len_obs
    f1 = 0.0 if (precision+coverage)==0 else 2*precision*coverage/(precision+coverage)
    ed = _levenshtein(ref, obs)
    denom = max(len_ref, len_obs)
    order_agreement = 1.0 - (ed/denom if denom else 0.0)
    return {"len_ref": len_ref, "len_obs": len_obs, "lcs": L, "precision": precision, "coverage": coverage, "f1_overall": f1, "order_agreement": order_agreement}

def build_observed_tokens() -> List[str]:
    """Extract observed token sequence from detection_stats['pattern_comparisons']."""
    tokens = []
    for comp in detection_stats.get("pattern_comparisons", []):
        sym = comp.get("event") or comp.get("symbol") or comp.get("event_symbol") or None
        if sym is not None:
            tokens.append(sym)
    return tokens

def compute_session_metrics(reference_patterns: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """Compute coverage- and order-aware metrics for the whole session per pattern."""
    obs_tokens = build_observed_tokens()
    result: Dict[str, Dict[str, float]] = {}
    for pname, steps in reference_patterns.items():
        ref_tokens = steps  # already token strings
        result[pname] = compute_sequence_metrics(ref_tokens, obs_tokens)
    detection_stats["session_metrics"] = result
    # Choose best by f1_overall
    best_name, best_f1 = None, 0.0
    for pname, m in result.items():
        if m["f1_overall"] > best_f1:
            best_name, best_f1 = pname, m["f1_overall"]
    detection_stats["best_match_pattern"] = best_name
    detection_stats["best_match_score"] = best_f1
    return result
def determine_final_verdict():
    """Calculate final detection verdict using coverage- and order-aware metrics."""
    total = detection_stats.get("total_events", 0)
    # Ensure session metrics are computed
    if 'session_metrics' not in detection_stats or not detection_stats['session_metrics']:
        try:
            _ = compute_session_metrics(reference_patterns)
        except Exception:
            pass
    metrics = detection_stats.get('session_metrics', {})
    if not metrics:
        return "No events to analyze"
    # Best by f1
    best_name, best = None, {"f1_overall":0.0, "order_agreement":0.0}
    for pname, m in metrics.items():
        if m.get('f1_overall',0.0) > best.get('f1_overall',0.0):
            best_name, best = pname, m
    detection_stats['best_match_pattern'] = best_name
    detection_stats['best_match_score'] = best.get('f1_overall',0.0)
    f1 = best.get('f1_overall',0.0)
    ord_ok = best.get('order_agreement',0.0)
    # Thresholds (tunable)
    F1_GOOD = float(os.environ.get('F1_GOOD', 0.60))
    F1_MAYBE = float(os.environ.get('F1_MAYBE', 0.40))
    ORD_GOOD = float(os.environ.get('ORD_GOOD', 0.60))
    ORD_MAYBE = float(os.environ.get('ORD_MAYBE', 0.50))
    if f1 >= F1_GOOD and ord_ok >= ORD_GOOD:
        detection_stats['final_verdict'] = '✅ FILIPPOS DETECTED'
    elif f1 >= F1_MAYBE and ord_ok >= ORD_MAYBE:
        detection_stats['final_verdict'] = '🤔 POSSIBLY FILIPPOS'
    else:
        detection_stats['final_verdict'] = '❌ NOT FILIPPOS (Unknown Person)'
    return detection_stats['final_verdict']

def print_detection_summary():
    """Display detection results in natural language with clear metric meanings."""
    if MODE != 'detect':
        return
    session = compute_session_metrics(reference_patterns) if 'reference_patterns' in globals() else {}
    total = detection_stats['total_events']
    if total == 0:
        print('No events detected in this session.')
        return
    verdict = determine_final_verdict()
    print(f"\nResult: {verdict}")
    if detection_stats['best_match_pattern']:
        print(
            f"Overall, your run is closest to '{detection_stats['best_match_pattern']}' "
            f"with an overall match score (F1) of {detection_stats['best_match_score']:.0%}."
        )
    if session:
        print("\nHow your run compares to each trained pattern:")
        ranked = sorted(session.items(), key=lambda kv: kv[1].get('f1_overall',0.0), reverse=True)
        for name, m in ranked:
            print(
                (
                    f"• {name}: about {m['f1_overall']:.0%} overall. "
                    f"This means we covered ~{m['precision']:.0%} of {name}'s reference steps, "
                    f"and about ~{m['coverage']:.0%} of your run is explained by {name}. "
                    f"The order of events looks ~{m['order_agreement']:.0%} similar. "
                    f"(Matched {m['lcs']} of {m['len_ref']} reference steps; your run had {m['len_obs']} steps.)"
                )
            )
    print("\nEvent-by-event view (for context):")
    print(
        f"Across {total} events: "
        f"{detection_stats['high_confidence_matches']} strong matches (~{detection_stats['high_confidence_matches']/total*100:.1f}%), "
        f"{detection_stats['low_confidence_matches']} weak matches (~{detection_stats['low_confidence_matches']/total*100:.1f}%), "
        f"{detection_stats['no_matches']} no-matches (~{detection_stats['no_matches']/total*100:.1f}%)."
    )

def save_detection_session():
    """Save detection results to database"""
    if MODE != 'detect' or mongo_detections is None:
        return
    
    verdict = determine_final_verdict()
    
    pattern_performance = {}
    if pattern_matcher:
        for pattern_name in pattern_matcher.patterns.keys():
            scores = detection_stats["pattern_scores"].get(pattern_name, [])
            if scores:
                pattern_performance[pattern_name] = {
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores)
                }
    
    detection_doc = {
        "session_id": SESSION_ID,
        "timestamp": datetime.datetime.now(),
        "duration_seconds": get_elapsed_time(),
        "pattern_tag": PATTERN_TAG,
        "final_verdict": verdict,
        "best_match_pattern": detection_stats["best_match_pattern"],
        "best_match_score": detection_stats["best_match_score"],
        "total_events": detection_stats["total_events"],
        "high_confidence_matches": detection_stats["high_confidence_matches"],
        "low_confidence_matches": detection_stats["low_confidence_matches"],
        "no_matches": detection_stats["no_matches"],
        "pattern_performance": pattern_performance,
        "session_metrics": detection_stats.get("session_metrics", {}),
        "all_comparisons": detection_stats["pattern_comparisons"][-100:]
    }
    
    try:
        mongo_detections.insert_one(detection_doc)
        print(f"✓ Detection summary saved")
    except Exception as e:
        print(f"Error saving detection: {e}")

def shutdown_gracefully():
    """Clean shutdown procedure"""
    global shutdown_initiated

    if shutdown_initiated:
        return
    shutdown_initiated = True

    print("\n" + "="*60)
    print("SHUTTING DOWN...")
    print("="*60)

    if MODE == 'collect':
        save_learned_pattern()

    if MODE == 'detect':
        print_detection_summary()
        save_detection_session()

        # Generate compact visual report
        try:
            create_compact_detection_report()  # Creates single-page report
        except Exception as e:
            print(f"Could not generate visual report: {e}")

    insert_session_marker("end")

    print(f"\n📊 Session Summary:")
    print(f"Session ID: {SESSION_ID}")
    print(f"Duration: {get_elapsed_time():.1f} seconds")
    print("="*60)
    os._exit(0)


# ============================================================================
# MQTT TOPICS (CITY) — only sensor topics + app variables at the end
# ============================================================================

TOPICS = [
    # Linear Alarms
    f"streamsim.{UID}.world.world.sensor.alarm.linear_alarm.start_alarm.data",
    f"streamsim.{UID}.world.world.sensor.alarm.linear_alarm.mall_alarm.data",
    f"streamsim.{UID}.world.world.sensor.alarm.linear_alarm.stadium_alarm.data",
    f"streamsim.{UID}.world.world.sensor.alarm.linear_alarm.farm_alarm.data",
    f"streamsim.{UID}.world.world.sensor.alarm.linear_alarm.parking_alarm.data",
    f"streamsim.{UID}.world.world.sensor.alarm.linear_alarm.pool_alarm.data",

    # Area Alarms
    f"streamsim.{UID}.world.world.sensor.alarm.area_alarm.area_alarm1.data",
    f"streamsim.{UID}.world.world.sensor.alarm.area_alarm.area_alarm2.data",
    f"streamsim.{UID}.world.world.sensor.alarm.area_alarm.area_alarm3.data",
    f"streamsim.{UID}.world.world.sensor.alarm.area_alarm.area_alarm4.data",
    f"streamsim.{UID}.world.world.sensor.alarm.area_alarm.area_alarm5.data",
    f"streamsim.{UID}.world.world.sensor.alarm.area_alarm.area_alarm6.data",
    f"streamsim.{UID}.world.world.sensor.alarm.area_alarm.area_alarm7.data",

    # App variables
    "appcreator.variables",
]

# ============================================================================
# MAIN PROGRAM
# ============================================================================

# Initialize pattern matcher for detection mode
pattern_matcher = None
if MODE == 'detect':
    reference_patterns = load_reference_patterns()
    if reference_patterns:
        pattern_matcher = PatternMatcher(reference_patterns)
        print(f"I'll compare this detection session against {len(reference_patterns)} trained patterns: {', '.join(reference_patterns.keys())}.")
    else:
        print("⚠️  No patterns loaded - run in collect mode first!")

def main():
    """Main execution loop"""
    insert_session_marker("start")
    
    conn = ConnectionParameters()
    node = Node(node_name='city_pattern_recognizer', connection_params=conn)

    print(f"\n📡 Subscribing to {len(TOPICS)} topics...")
    for topic in TOPICS:
        node.create_subscriber(topic=topic, msg_type=dict, on_message=make_callback(topic))

    print("\n" + "="*60)
    if MODE == 'collect':
        print("📝 TRAINING MODE - Recording pattern")
    else:
        print("🔍 DETECTION MODE - Identifying person")
    print("="*60 + "\n")
    
    print(f"⏳ Warmup: {WARMUP_SECONDS} seconds...")
    print(f"📊 Waiting for simulation end signal...")
    
    try:
        node.run_forever()
    except KeyboardInterrupt:
        print("\n⚠️ Manual interrupt")
        shutdown_gracefully()

if __name__ == "__main__":
    main()
