#!/usr/bin/env python3
"""
Streamlit UI for Rocket Simulation
Create, run, and visualize rocket simulations with Elodin integration

Modern, polished UI with aerospace-inspired design
"""

import sys
import os
from pathlib import Path

# Add the rocket-barrowman directory to path (but not if it's already there)
_rocket_dir = os.path.dirname(os.path.abspath(__file__))
if _rocket_dir not in sys.path:
    sys.path.insert(0, _rocket_dir)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

# Import rocket simulation components
from environment import Environment
from motor_model import Motor
from rocket_model import Rocket as RocketModel
from flight_solver import FlightSolver, FlightResult
from calisto_builder import build_calisto
from rocket_visualizer import visualize_rocket_3d, visualize_rocket_2d_side_view
from openrocket_components import (
    Rocket,
    NoseCone,
    BodyTube,
    TrapezoidFinSet,
    InnerTube,
    Parachute,
    MATERIALS,
)
from openrocket_motor import Motor as ORMotor
from ai_rocket_builder import RocketDesigner
from smart_optimizer import SmartOptimizer
from motor_scraper import ThrustCurveScraper

try:
    from mesh_renderer import (
        render_to_plotly,
        get_rocket_preview_html,
        TRIMESH_AVAILABLE,
    )
except ImportError:
    TRIMESH_AVAILABLE = False
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & THEMING
# ═══════════════════════════════════════════════════════════════════════════════

# Color palette - Aerospace inspired
COLORS = {
    "primary": "#00D4FF",  # Cyan - main accent
    "secondary": "#FF6B35",  # Orange - thrust/energy
    "success": "#00FF88",  # Green - success states
    "warning": "#FFB800",  # Amber - warnings
    "danger": "#FF3366",  # Red - errors/danger
    "background": "#0A0E17",  # Deep space blue
    "surface": "#131B2E",  # Elevated surface
    "surface2": "#1A2540",  # Higher elevation
    "text": "#E8ECF4",  # Primary text
    "text_muted": "#8B9CB6",  # Secondary text
    "border": "#2A3A5C",  # Borders
    "gradient_start": "#00D4FF",
    "gradient_end": "#7B2FFF",
}

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Rocket Flight Simulator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
/* ═══════════════════════════════════════════════════════════════════════════ */
/* IMPORTS & FONTS                                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ROOT VARIABLES                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */
:root {
    --primary: #00D4FF;
    --secondary: #FF6B35;
    --success: #00FF88;
    --warning: #FFB800;
    --danger: #FF3366;
    --bg-primary: #0A0E17;
    --bg-surface: #131B2E;
    --bg-surface2: #1A2540;
    --text-primary: #E8ECF4;
    --text-muted: #8B9CB6;
    --border: #2A3A5C;
    --gradient: linear-gradient(135deg, #00D4FF 0%, #7B2FFF 100%);
    --glow: 0 0 20px rgba(0, 212, 255, 0.3);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* GLOBAL STYLES                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */
.stApp {
    background: linear-gradient(180deg, #0A0E17 0%, #0D1321 50%, #0A0E17 100%);
    font-family: 'Outfit', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TYPOGRAPHY                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

p, span, div, label {
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
}

code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SIDEBAR                                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #131B2E 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding-top: 1rem;
}

.sidebar-header {
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* HERO SECTION                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */
.hero-container {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 47, 255, 0.1) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--gradient);
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.1rem;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid var(--primary);
    color: var(--primary);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* CARDS & CONTAINERS                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */
.glass-card {
    background: rgba(19, 27, 46, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: var(--primary);
    box-shadow: var(--glow);
}

.section-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title-icon {
    font-size: 1.5rem;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* METRICS CARDS                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(123, 47, 255, 0.08) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--primary);
    box-shadow: var(--glow);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--gradient);
}

.metric-label {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-value {
    color: var(--text-primary);
    font-size: 1.75rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.metric-unit {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-left: 0.25rem;
}

/* Colored variants */
.metric-card.altitude::before { background: var(--primary); }
.metric-card.velocity::before { background: var(--secondary); }
.metric-card.time::before { background: var(--success); }
.metric-card.mach::before { background: var(--warning); }

/* ═══════════════════════════════════════════════════════════════════════════ */
/* BUTTONS                                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */
.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
    border: none !important;
}

.stButton > button[kind="primary"] {
    background: var(--gradient) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
}

.stButton > button[kind="secondary"] {
    background: var(--bg-surface2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* Big launch button */
.launch-button {
    background: linear-gradient(135deg, #FF6B35 0%, #FF3366 100%) !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* INPUTS & FORMS                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2) !important;
}

.stSelectbox > div > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stRadio > div {
    background: var(--bg-surface);
    border-radius: 8px;
    padding: 0.5rem;
    border: 1px solid var(--border);
}

.stRadio > div > label {
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

.stRadio > div > label:hover {
    background: var(--bg-surface2) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* EXPANDERS                                                                    */
/* ═══════════════════════════════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--primary) !important;
}

.streamlit-expanderContent {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TABS                                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface);
    border-radius: 12px;
    padding: 0.5rem;
    border: 1px solid var(--border);
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-surface2) !important;
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient) !important;
    color: white !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ALERTS & MESSAGES                                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */
.stSuccess {
    background: rgba(0, 255, 136, 0.1) !important;
    border: 1px solid var(--success) !important;
    border-radius: 8px !important;
    color: var(--success) !important;
}

.stWarning {
    background: rgba(255, 184, 0, 0.1) !important;
    border: 1px solid var(--warning) !important;
    border-radius: 8px !important;
    color: var(--warning) !important;
}

.stError {
    background: rgba(255, 51, 102, 0.1) !important;
    border: 1px solid var(--danger) !important;
    border-radius: 8px !important;
    color: var(--danger) !important;
}

.stInfo {
    background: rgba(0, 212, 255, 0.1) !important;
    border: 1px solid var(--primary) !important;
    border-radius: 8px !important;
    color: var(--primary) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* DATAFRAMES                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PROGRESS & SPINNERS                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */
.stSpinner > div {
    border-color: var(--primary) !important;
}

.stProgress > div > div {
    background: var(--gradient) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* CUSTOM COMPONENTS                                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */
.status-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    box-sizing: border-box;
}

.status-badge.ready {
    background: rgba(0, 255, 136, 0.15);
    border: 1px solid var(--success);
    color: var(--success);
}

.status-badge.warning {
    background: rgba(255, 184, 0, 0.15);
    border: 1px solid var(--warning);
    color: var(--warning);
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--border) 50%, transparent 100%);
    margin: 1.5rem 0;
}

/* Animated glow effect */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
    50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
}

.glow-effect {
    animation: pulse-glow 2s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* RESPONSIVE                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero-title {
        font-size: 1.75rem;
    }
    .metric-grid {
        grid-template-columns: 1fr;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SIDEBAR EXPAND HINT                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */
/* When sidebar is collapsed, Streamlit shows an expand button in the header.   */
/* This hint tab provides additional visual cue on the left edge.               */
.sidebar-hint-tab {
    position: fixed;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    z-index: 1000000;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.3) 0%, rgba(123, 47, 255, 0.3) 100%);
    border: 1px solid var(--border);
    border-left: none;
    border-radius: 0 8px 8px 0;
    padding: 12px 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.sidebar-hint-tab:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.5) 0%, rgba(123, 47, 255, 0.5) 100%);
    border-color: var(--primary);
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    padding-right: 16px;
}

.sidebar-hint-tab svg {
    width: 16px;
    height: 16px;
    fill: var(--primary);
    transition: transform 0.2s ease;
}

.sidebar-hint-tab:hover svg {
    transform: translateX(3px);
}

/* Tooltip that appears on hover */
.sidebar-hint-tab::after {
    content: 'Click ≫ icon in header to show menu';
    position: absolute;
    left: calc(100% + 10px);
    top: 50%;
    transform: translateY(-50%);
    background: var(--bg-surface);
    border: 1px solid var(--primary);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--text-primary);
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease, transform 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.sidebar-hint-tab:hover::after {
    opacity: 1;
}
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar hint tab - rendered via st.markdown (visual only, no JS)
# Plus a hidden component that injects click handler into the parent document
import streamlit.components.v1 as components

# First, render the visual tab via st.markdown (this shows up in the main document)
st.markdown(
    """
<div class="sidebar-hint-tab" id="sidebarHintTab" title="Click to show menu">
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/>
    </svg>
</div>
""",
    unsafe_allow_html=True,
)

# Then inject JavaScript via components.html to attach the click handler
# This script runs in an iframe but manipulates the parent document
_sidebar_click_handler = """
<script>
(function() {
    function expandSidebar() {
        try {
            var parentDoc = window.parent.document;
            
            // Find Streamlit's expand button by looking for the icon text
            var buttons = parentDoc.querySelectorAll('button');
            for (var i = 0; i < buttons.length; i++) {
                var btn = buttons[i];
                if (btn.textContent && btn.textContent.indexOf('keyboard_double_arrow_right') !== -1) {
                    btn.click();
                    return true;
                }
            }
            return false;
        } catch (e) {
            console.error('Error:', e);
            return false;
        }
    }
    
    // Check if sidebar is visible by examining its transform property
    function isSidebarVisible() {
        try {
            var parentDoc = window.parent.document;
            var sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return false;
            
            var style = window.parent.getComputedStyle(sidebar);
            var transform = style.transform;
            
            // Parse the transform matrix to get the X translation
            // transform is either "none" or "matrix(a, b, c, d, tx, ty)"
            if (transform === 'none') {
                return true; // No transform means visible
            }
            
            var match = transform.match(/matrix.*\((.+)\)/);
            if (match) {
                var values = match[1].split(', ');
                var tx = parseFloat(values[4]); // translateX is the 5th value
                return tx >= 0; // Visible if translateX >= 0
            }
            return true; // Default to visible if can't parse
        } catch (e) {
            return true; // Default to visible on error
        }
    }
    
    // Update tab visibility based on sidebar state
    function updateTabVisibility() {
        try {
            var parentDoc = window.parent.document;
            var tab = parentDoc.getElementById('sidebarHintTab');
            if (tab) {
                var sidebarVisible = isSidebarVisible();
                tab.style.opacity = sidebarVisible ? '0' : '1';
                tab.style.pointerEvents = sidebarVisible ? 'none' : 'auto';
            }
        } catch (e) {}
    }
    
    function attachHandler() {
        try {
            var parentDoc = window.parent.document;
            var tab = parentDoc.getElementById('sidebarHintTab');
            if (tab && !tab.hasAttribute('data-handler-attached')) {
                tab.setAttribute('data-handler-attached', 'true');
                tab.style.cursor = 'pointer';
                tab.style.transition = 'opacity 0.3s ease';
                tab.addEventListener('click', function(e) {
                    e.preventDefault();
                    expandSidebar();
                });
                console.log('Sidebar tab handler attached');
                
                // Initial visibility check
                updateTabVisibility();
            }
        } catch (e) {
            console.error('Error attaching handler:', e);
        }
    }
    
    // Try to attach handler immediately and with delays (for dynamic loading)
    attachHandler();
    setTimeout(attachHandler, 100);
    setTimeout(attachHandler, 500);
    setTimeout(attachHandler, 1000);
    
    // Watch for DOM/style changes to update tab visibility
    var observer = new MutationObserver(function() {
        attachHandler();
        updateTabVisibility();
    });
    try {
        observer.observe(window.parent.document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class'] });
    } catch (e) {}
    
    // Also poll for sidebar state changes (backup for transform changes not caught by observer)
    setInterval(updateTabVisibility, 200);
})();
</script>
"""

# Render hidden iframe that injects the JavaScript
components.html(_sidebar_click_handler, height=0, scrolling=False)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None
if "rocket_config" not in st.session_state:
    st.session_state.rocket_config = None
if "motor_config" not in st.session_state:
    st.session_state.motor_config = None
if "motor_database" not in st.session_state:
    st.session_state.motor_database = []
if "ai_designer" not in st.session_state:
    st.session_state.ai_designer = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "design"

# Auto-load motor database on startup
if len(st.session_state.motor_database) == 0:
    try:
        scraper = ThrustCurveScraper()
        motor_db = scraper.load_motor_database()
        if motor_db and len(motor_db) >= 50:
            st.session_state.motor_database = motor_db
            openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
            st.session_state.ai_designer = RocketDesigner(motor_db, openai_api_key=openai_key)
        elif motor_db and len(motor_db) < 50:
            st.session_state.motor_database = motor_db
            openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
            st.session_state.ai_designer = RocketDesigner(motor_db, openai_api_key=openai_key)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_chart_layout(title: str, xaxis_title: str, yaxis_title: str) -> dict:
    """Create consistent chart layout."""
    return {
        "title": {
            "text": title,
            "font": {"size": 16, "color": COLORS["text"], "family": "Outfit"},
            "x": 0.5,
        },
        "xaxis": {
            "title": xaxis_title,
            "color": COLORS["text_muted"],
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "zeroline": False,
        },
        "yaxis": {
            "title": yaxis_title,
            "color": COLORS["text_muted"],
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "zeroline": False,
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(19, 27, 46, 0.5)",
        "font": {"family": "Outfit", "color": COLORS["text"]},
        "margin": {"t": 50, "b": 50, "l": 60, "r": 30},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": COLORS["surface"],
            "bordercolor": COLORS["primary"],
            "font": {"family": "JetBrains Mono", "size": 12},
        },
    }


def render_metric_card(label: str, value: str, unit: str = "", variant: str = "default"):
    """Render a styled metric card."""
    st.markdown(
        f"""
    <div class="metric-card {variant}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_hero():
    """Render the hero section."""
    st.markdown(
        """
    <div class="hero-container">
        <div class="hero-title">
            🚀 Rocket Flight Simulator
            <span class="hero-badge">6-DOF</span>
        </div>
        <div class="hero-subtitle">
            Design, simulate, and visualize high-powered rocket flights with Barrowman aerodynamics
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str):
    """Render a section header."""
    st.markdown(
        f"""
    <div class="section-title">
        <span class="section-title-icon">{icon}</span>
        {title}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_divider():
    """Render a styled divider."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROCKET BUILDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def build_custom_rocket(config: Dict[str, Any]) -> Rocket:
    """Build a custom rocket from configuration."""
    rocket = Rocket(config.get("name", "Custom Rocket"))

    # Nose cone
    if config.get("has_nose", True):
        nose_shape_str = config.get("nose_shape", "VON_KARMAN").upper()
        nose_shape_map = {
            "CONICAL": NoseCone.Shape.CONICAL,
            "OGIVE": NoseCone.Shape.OGIVE,
            "ELLIPSOID": NoseCone.Shape.ELLIPSOID,
            "PARABOLIC": NoseCone.Shape.PARABOLIC,
            "POWER_SERIES": NoseCone.Shape.POWER_SERIES,
            "HAACK": NoseCone.Shape.HAACK,
            "VON_KARMAN": NoseCone.Shape.VON_KARMAN,
        }
        nose_shape = nose_shape_map.get(nose_shape_str, NoseCone.Shape.VON_KARMAN)

        nose = NoseCone(
            name="Nose Cone",
            length=config.get("nose_length", 0.5),
            base_radius=config.get("body_radius", 0.0635),
            thickness=config.get("nose_thickness", 0.003),
            shape=nose_shape,
        )
        nose.material = MATERIALS.get(
            config.get("nose_material", "Fiberglass"), MATERIALS["Fiberglass"]
        )
        nose.position.x = 0.0
        rocket.add_child(nose)

    # Body tube
    body = BodyTube(
        name="Body Tube",
        length=config.get("body_length", 1.5),
        outer_radius=config.get("body_radius", 0.0635),
        thickness=config.get("body_thickness", 0.003),
    )
    body.material = MATERIALS.get(
        config.get("body_material", "Fiberglass"), MATERIALS["Fiberglass"]
    )
    body.position.x = config.get("nose_length", 0.5) if config.get("has_nose", True) else 0.0
    rocket.add_child(body)

    # Fins
    if config.get("has_fins", True):
        fins = TrapezoidFinSet(
            name="Fins",
            fin_count=config.get("fin_count", 4),
            root_chord=config.get("fin_root_chord", 0.12),
            tip_chord=config.get("fin_tip_chord", 0.06),
            span=config.get("fin_span", 0.11),
            sweep=config.get("fin_sweep", 0.06),
            thickness=config.get("fin_thickness", 0.005),
        )
        fins.material = MATERIALS.get(
            config.get("fin_material", "Fiberglass"), MATERIALS["Fiberglass"]
        )
        body_length = config.get("body_length", 1.5)
        nose_length = config.get("nose_length", 0.5) if config.get("has_nose", True) else 0.0
        fins.position.x = nose_length + body_length - config.get("fin_root_chord", 0.12)
        body.add_child(fins)

    # Motor mount
    if config.get("has_motor_mount", True):
        motor_mount = InnerTube(
            name="Motor Mount",
            length=config.get("motor_mount_length", 0.5),
            outer_radius=config.get("motor_mount_radius", 0.041),
            thickness=config.get("motor_mount_thickness", 0.003),
        )
        motor_mount.material = MATERIALS.get(
            config.get("motor_mount_material", "Fiberglass"), MATERIALS["Fiberglass"]
        )
        body_length = config.get("body_length", 1.5)
        nose_length = config.get("nose_length", 0.5) if config.get("has_nose", True) else 0.0
        motor_mount.position.x = nose_length + body_length - config.get("motor_mount_length", 0.5)
        motor_mount.motor_mount = True
        body.add_child(motor_mount)

    # Parachutes
    if config.get("has_main_chute", True):
        main_chute = Parachute(
            name="Main",
            diameter=config.get("main_chute_diameter", 2.91),
            cd=config.get("main_chute_cd", 1.5),
        )
        main_chute.deployment_event = config.get("main_deployment_event", "ALTITUDE")
        main_chute.deployment_altitude = config.get("main_deployment_altitude", 800.0)
        main_chute.deployment_delay = config.get("main_deployment_delay", 1.5)
        if config.get("has_nose", True):
            nose.add_child(main_chute)
        else:
            body.add_child(main_chute)

    if config.get("has_drogue", True):
        drogue = Parachute(
            name="Drogue",
            diameter=config.get("drogue_diameter", 0.99),
            cd=config.get("drogue_cd", 1.3),
        )
        drogue.deployment_event = config.get("drogue_deployment_event", "APOGEE")
        drogue.deployment_altitude = config.get("drogue_deployment_altitude", 0.0)
        drogue.deployment_delay = config.get("drogue_deployment_delay", 1.5)
        if config.get("has_nose", True):
            nose.add_child(drogue)
        else:
            body.add_child(drogue)

    rocket.calculate_reference_values()
    return rocket


def build_custom_motor(config: Dict[str, Any]) -> ORMotor:
    """Build a custom motor from configuration."""
    if "thrust_curve" in config and config["thrust_curve"]:
        thrust_curve = config["thrust_curve"]
        burn_time = config.get(
            "burn_time", max([t for t, _ in thrust_curve]) if thrust_curve else 3.9
        )
        total_impulse = config.get(
            "total_impulse",
            sum(
                (thrust_curve[i][1] + thrust_curve[i + 1][1])
                / 2
                * (thrust_curve[i + 1][0] - thrust_curve[i][0])
                for i in range(len(thrust_curve) - 1)
            )
            if len(thrust_curve) > 1
            else 0,
        )
    else:
        burn_time = config.get("burn_time", 3.9)
        max_thrust = config.get("max_thrust", 2200.0)
        avg_thrust = config.get("avg_thrust", max_thrust * 0.7)

        times = np.linspace(0, burn_time, 20).tolist()
        thrusts = []
        for t in times:
            if t < burn_time * 0.1:
                thrust = max_thrust * (t / (burn_time * 0.1))
            elif t < burn_time * 0.9:
                thrust = avg_thrust
            else:
                thrust = avg_thrust * (1 - (t - burn_time * 0.9) / (burn_time * 0.1))
            thrusts.append(max(0, thrust))

        thrust_curve = list(zip(times, thrusts))
        total_impulse = sum(
            (thrusts[i] + thrusts[i + 1]) / 2 * (times[i + 1] - times[i])
            for i in range(len(times) - 1)
        )

    motor = ORMotor(
        designation=config.get("designation", config.get("motor_name", "Custom Motor")),
        manufacturer=config.get("manufacturer", config.get("motor_manufacturer", "Custom")),
        diameter=config.get("diameter", config.get("motor_diameter", 0.075)),
        length=config.get("length", config.get("motor_length", 0.64)),
        total_mass=config.get("total_mass", config.get("motor_total_mass", 4.771)),
        propellant_mass=config.get("propellant_mass", config.get("motor_propellant_mass", 2.956)),
        thrust_curve=thrust_curve,
        burn_time=burn_time,
        total_impulse=config.get("total_impulse", total_impulse),
    )

    motor.cg_position = config.get("motor_cg_position", 0.317)
    motor.propellant_cg = config.get("motor_propellant_cg", 0.397)
    motor.inertia_axial = config.get("motor_inertia_axial", 0.002)
    motor.inertia_lateral = config.get("motor_inertia_lateral", 0.125)

    return motor


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def visualize_results(result: FlightResult):
    """Create visualizations of simulation results with styled charts."""
    if not result or not result.history:
        st.error("No simulation data to visualize")
        return

    history = result.history

    # Extract data
    times = [s.time for s in history]
    altitudes = [s.z for s in history]
    velocities = [np.linalg.norm(s.velocity) for s in history]
    downrange = [np.linalg.norm([s.x, s.y]) for s in history]
    machs = [s.mach for s in history]
    aoas = [np.degrees(s.angle_of_attack) for s in history]
    dynamic_pressures = [s.dynamic_pressure / 1000 for s in history]  # Convert to kPa

    # Key metrics
    max_alt = max(altitudes)
    max_v = max(velocities)
    max_mach = max(machs)
    apogee_time = times[altitudes.index(max_alt)]

    # Metrics row
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card altitude">
            <div class="metric-label">Max Altitude</div>
            <div class="metric-value">{max_alt:,.0f}<span class="metric-unit">m</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card velocity">
            <div class="metric-label">Max Velocity</div>
            <div class="metric-value">{max_v:,.0f}<span class="metric-unit">m/s</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card mach">
            <div class="metric-label">Max Mach</div>
            <div class="metric-value">{max_mach:.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card time">
            <div class="metric-label">Apogee Time</div>
            <div class="metric-value">{apogee_time:.1f}<span class="metric-unit">s</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Trajectory", "⚡ Performance", "🌪️ Aerodynamics", "🌍 3D Path"]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Altitude vs Time
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=altitudes,
                    mode="lines",
                    name="Altitude",
                    line=dict(color=COLORS["primary"], width=3),
                    fill="tozeroy",
                    fillcolor="rgba(0, 212, 255, 0.1)",
                )
            )
            fig.update_layout(**create_chart_layout("Altitude vs Time", "Time (s)", "Altitude (m)"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Velocity vs Time
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=velocities,
                    mode="lines",
                    name="Velocity",
                    line=dict(color=COLORS["secondary"], width=3),
                    fill="tozeroy",
                    fillcolor="rgba(255, 107, 53, 0.1)",
                )
            )
            fig.update_layout(
                **create_chart_layout("Velocity vs Time", "Time (s)", "Velocity (m/s)")
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Trajectory plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=downrange,
                y=altitudes,
                mode="lines",
                name="Trajectory",
                line=dict(color=COLORS["success"], width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[downrange[altitudes.index(max_alt)]],
                y=[max_alt],
                mode="markers",
                name="Apogee",
                marker=dict(color=COLORS["warning"], size=12, symbol="star"),
            )
        )
        fig.update_layout(
            **create_chart_layout("Flight Trajectory", "Downrange (m)", "Altitude (m)")
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Mach number
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=machs,
                    mode="lines",
                    name="Mach",
                    line=dict(color=COLORS["warning"], width=3),
                )
            )
            # Add sonic line
            fig.add_hline(
                y=1,
                line_dash="dash",
                line_color=COLORS["danger"],
                annotation_text="Mach 1",
                annotation_position="top right",
            )
            fig.update_layout(**create_chart_layout("Mach Number", "Time (s)", "Mach"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Altitude vs Velocity phase plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=velocities,
                    y=altitudes,
                    mode="lines",
                    name="Phase",
                    line=dict(color=COLORS["primary"], width=2),
                    marker=dict(
                        color=times,
                        colorscale="Viridis",
                        size=4,
                        showscale=True,
                        colorbar=dict(title="Time (s)"),
                    ),
                )
            )
            fig.update_layout(
                **create_chart_layout("Altitude-Velocity Phase", "Velocity (m/s)", "Altitude (m)")
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            # Angle of Attack
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=aoas,
                    mode="lines",
                    name="AoA",
                    line=dict(color=COLORS["danger"], width=2),
                )
            )
            fig.update_layout(**create_chart_layout("Angle of Attack", "Time (s)", "Angle (deg)"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Dynamic Pressure
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=dynamic_pressures,
                    mode="lines",
                    name="Q",
                    line=dict(color=COLORS["primary"], width=3),
                    fill="tozeroy",
                    fillcolor="rgba(0, 212, 255, 0.1)",
                )
            )
            fig.update_layout(
                **create_chart_layout("Dynamic Pressure (Max-Q)", "Time (s)", "Pressure (kPa)")
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # 3D trajectory
        x_coords = [s.x for s in history]
        y_coords = [s.y for s in history]
        z_coords = [s.z for s in history]

        fig = go.Figure(
            data=go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="lines",
                line=dict(
                    color=times,
                    colorscale=[
                        [0, COLORS["primary"]],
                        [0.5, COLORS["warning"]],
                        [1, COLORS["secondary"]],
                    ],
                    width=6,
                ),
                marker=dict(size=2),
            )
        )

        # Add apogee marker
        apogee_idx = altitudes.index(max_alt)
        fig.add_trace(
            go.Scatter3d(
                x=[x_coords[apogee_idx]],
                y=[y_coords[apogee_idx]],
                z=[z_coords[apogee_idx]],
                mode="markers",
                marker=dict(size=10, color=COLORS["warning"], symbol="diamond"),
                name="Apogee",
            )
        )

        fig.update_layout(
            title={
                "text": "3D Flight Path",
                "font": {"size": 16, "color": COLORS["text"], "family": "Outfit"},
                "x": 0.5,
            },
            scene=dict(
                xaxis=dict(title="X (m)", color=COLORS["text_muted"], gridcolor=COLORS["border"]),
                yaxis=dict(title="Y (m)", color=COLORS["text_muted"], gridcolor=COLORS["border"]),
                zaxis=dict(
                    title="Altitude (m)", color=COLORS["text_muted"], gridcolor=COLORS["border"]
                ),
                bgcolor="rgba(10, 14, 23, 0.8)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Outfit", "color": COLORS["text"]},
            height=600,
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
        )
        st.plotly_chart(fig, use_container_width=True)


def launch_elodin_editor(result: FlightResult, solver: FlightSolver):
    """Launch Elodin editor with simulation results."""
    import subprocess
    import tempfile
    import pickle

    try:
        temp_dir = Path(tempfile.gettempdir()) / "elodin_rocket_sim"
        temp_dir.mkdir(exist_ok=True)

        result_file = temp_dir / "simulation_result.pkl"
        solver_file = temp_dir / "solver.pkl"

        # Extract data
        summary = result.summary if hasattr(result, "summary") and result.summary else {}

        if not summary or "max_altitude" not in summary:
            max_alt = max(s.z for s in result.history) if result.history else 0.0
            max_v = (
                max(np.linalg.norm(s.velocity) for s in result.history) if result.history else 0.0
            )
            apogee_state = (
                next((s for s in result.history if s.z == max_alt), result.history[-1])
                if result.history
                else None
            )
            apogee_time = apogee_state.time if apogee_state else 0.0
            landing_time = result.history[-1].time if result.history else 0.0
        else:
            max_alt = summary.get("max_altitude", 0.0)
            max_v = summary.get("max_velocity", 0.0)
            apogee_time = summary.get("apogee_time", 0.0)
            landing_time = summary.get("landing_time", 0.0)

        result_data = {
            "history": [
                {
                    "time": s.time,
                    "position": s.position.tolist()
                    if isinstance(s.position, np.ndarray)
                    else s.position,
                    "velocity": s.velocity.tolist()
                    if isinstance(s.velocity, np.ndarray)
                    else s.velocity,
                    "quaternion": s.quaternion.tolist()
                    if isinstance(s.quaternion, np.ndarray)
                    else s.quaternion,
                    "angular_velocity": s.angular_velocity.tolist()
                    if isinstance(s.angular_velocity, np.ndarray)
                    else s.angular_velocity,
                    "motor_mass": s.motor_mass,
                    "angle_of_attack": s.angle_of_attack,
                    "sideslip": s.sideslip,
                    "mach": getattr(s, "mach", 0.0),
                    "dynamic_pressure": getattr(s, "dynamic_pressure", 0.0),
                    "drag_force": getattr(s, "drag_force", np.array([0.0, 0.0, 0.0])).tolist()
                    if isinstance(getattr(s, "drag_force", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "lift_force": getattr(s, "lift_force", np.array([0.0, 0.0, 0.0])).tolist()
                    if isinstance(getattr(s, "lift_force", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "parachute_drag": getattr(
                        s, "parachute_drag", np.array([0.0, 0.0, 0.0])
                    ).tolist()
                    if isinstance(getattr(s, "parachute_drag", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "moment_world": getattr(s, "moment_world", np.array([0.0, 0.0, 0.0])).tolist()
                    if isinstance(getattr(s, "moment_world", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "total_aero_force": s.total_aero_force.tolist()
                    if isinstance(s.total_aero_force, np.ndarray)
                    else s.total_aero_force,
                }
                for s in result.history
            ],
            "max_altitude": max_alt,
            "max_velocity": max_v,
            "apogee_time": apogee_time,
            "landing_time": landing_time,
        }

        mass_model = solver.mass_model
        solver_data = {
            "rocket": {
                "dry_mass": solver.rocket.dry_mass,
                "dry_cg": solver.rocket.dry_cg,
                "reference_diameter": solver.rocket.reference_diameter,
                "structural_mass": mass_model.structural_mass,
                "structural_cg": mass_model.structural_cg,
                "structural_inertia": mass_model.structural_inertia.tolist()
                if isinstance(mass_model.structural_inertia, np.ndarray)
                else list(mass_model.structural_inertia),
            },
            "motor": {
                "total_mass": solver.motor.total_mass,
                "propellant_mass": solver.motor.propellant_mass,
            },
            "environment": {"elevation": solver.environment.elevation},
            "mass_model": {
                "times": mass_model.times.tolist()
                if isinstance(mass_model.times, np.ndarray)
                else list(mass_model.times),
                "total_mass_values": mass_model.total_mass_values.tolist()
                if isinstance(mass_model.total_mass_values, np.ndarray)
                else list(mass_model.total_mass_values),
                "inertia_values": mass_model.inertia_values.tolist()
                if isinstance(mass_model.inertia_values, np.ndarray)
                else [[0.0, 0.0, 0.0]],
            },
        }

        with open(result_file, "wb") as f:
            pickle.dump(result_data, f)
        with open(solver_file, "wb") as f:
            pickle.dump(solver_data, f)

        script_dir = Path(__file__).parent.resolve()  # Use absolute path
        main_py = script_dir / "main.py"

        # Get the elodin repository root (two levels up from examples/rocket-barrowman/)
        # This ensures ELODIN_ASSETS_DIR finds the root-level assets/ directory
        elodin_root = script_dir.parent.parent.resolve()  # Absolute path to repo root

        if not main_py.exists():
            st.error(f"main.py not found at {main_py}")
            return

        # Launch editor from elodin root so assets/ directory is found
        with st.spinner("🚀 Launching Elodin Editor..."):
            # Use absolute path to main.py since we're launching from repo root
            main_py_abs = str(main_py.resolve())
            cmd = ["elodin", "editor", main_py_abs]

            import platform

            if platform.system() != "Windows":
                terminals = [
                    (["gnome-terminal", "--"], "bash -c"),
                    (["xterm", "-e"], "bash -c"),
                    (["konsole", "-e"], "bash -c"),
                ]

                # Launch from elodin root so ELODIN_ASSETS_DIR defaults to ./assets/
                cmd_str = f"cd {elodin_root} && elodin editor {main_py_abs}"

                for term_base, shell_prefix in terminals:
                    try:
                        which_result = subprocess.run(
                            ["which", term_base[0]], capture_output=True, timeout=1
                        )
                        if which_result.returncode == 0:
                            full_cmd = term_base + ["bash", "-c", cmd_str]
                            subprocess.Popen(full_cmd, cwd=str(elodin_root), start_new_session=True)
                            st.success("✅ Elodin Editor launched!")
                            return
                    except Exception:
                        continue

            # Fallback - launch from elodin root
            subprocess.Popen(cmd, cwd=str(elodin_root), start_new_session=True)
            st.success("✅ Elodin Editor launched in background")

    except Exception as e:
        st.error(f"Error launching Elodin: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════


def render_sidebar():
    """Render the sidebar with all configuration options."""
    with st.sidebar:
        # Logo / Header
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
            <div class="sidebar-header">🚀 ROCKET SIM</div>
            <div style="color: var(--text-muted); font-size: 0.85rem;">Barrowman Aerodynamics</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        render_divider()

        # ─────────────────────────────────────────────────────────────────────
        # ROCKET TYPE SELECTION
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### 🎯 Design Mode")
        rocket_type = st.radio(
            "Select design approach",
            ["Calisto (Default)", "Custom Rocket", "AI Builder"],
            label_visibility="collapsed",
        )

        render_divider()

        # ─────────────────────────────────────────────────────────────────────
        # AI BUILDER
        # ─────────────────────────────────────────────────────────────────────
        if rocket_type == "AI Builder":
            st.markdown("### 🤖 Smart Optimizer")
            st.caption("Tell us what you need - we'll find the cheapest working design")

            ai_input = st.text_area(
                "Requirements",
                placeholder="e.g., 'rocket to 5000 ft as cheap as possible' or '10k feet with 2kg payload, under $500'",
                height=100,
                label_visibility="collapsed",
            )

            max_iterations = st.slider(
                "Max simulations", 10, 50, 25, help="More = better search, slower"
            )

            if st.button("🔍 Find Cheapest Design", type="primary", use_container_width=True):
                if ai_input:
                    # Progress placeholder
                    log_container = st.empty()

                    with st.spinner("Optimizing design..."):
                        try:
                            # Initialize optimizer
                            motor_db = st.session_state.motor_database
                            if not motor_db:
                                scraper = ThrustCurveScraper()
                                motor_db = scraper.load_motor_database()
                                st.session_state.motor_database = motor_db

                            optimizer = SmartOptimizer(motor_db)

                            # Progress callback
                            progress_lines = []

                            def on_progress(i, total, design):
                                status = "✓" if design.meets_target else "✗"
                                line = f"{status} [{i}/{total}] {design.motor_designation}: {design.simulated_altitude_m:.0f}m, ${design.cost.total:.0f}"
                                progress_lines.append(line)
                                log_container.code("\n".join(progress_lines[-10:]))  # Last 10 lines

                            best, designs, log = optimizer.optimize_from_text(
                                ai_input, max_iterations=max_iterations, callback=on_progress
                            )

                            if best and best.rocket_config:
                                st.session_state.rocket_config = best.rocket_config
                                st.session_state.motor_config = best.motor_config
                                st.session_state.ai_design_just_generated = True
                                st.session_state.last_optimization_result = best
                                st.session_state.last_optimization_log = log

                                # Show results
                                st.success(f"✅ Found: {best.motor_designation}")

                                cols = st.columns(3)
                                cols[0].metric(
                                    "Altitude",
                                    f"{best.simulated_altitude_m:.0f}m",
                                    f"{best.simulated_altitude_m * 3.281:.0f} ft",
                                )
                                cols[1].metric("Total Cost", f"${best.cost.total:.0f}")
                                cols[2].metric("Tube Size", f"{best.body_diameter_m * 1000:.0f}mm")

                                # Cost breakdown
                                with st.expander("💰 Cost Breakdown"):
                                    st.markdown(f"""
                                    | Component | Cost |
                                    |-----------|------|
                                    | Motor | ${best.cost.motor_cost:.0f} |
                                    | Body Tube | ${best.cost.body_tube_cost:.0f} |
                                    | Nose Cone | ${best.cost.nose_cone_cost:.0f} |
                                    | Fins | ${best.cost.fins_cost:.0f} |
                                    | Motor Mount | ${best.cost.motor_mount_cost:.0f} |
                                    | Recovery | ${best.cost.recovery_cost:.0f} |
                                    | Avionics | ${best.cost.avionics_cost:.0f} |
                                    | Hardware | ${best.cost.hardware_cost:.0f} |
                                    | **Total** | **${best.cost.total:.0f}** |
                                    """)

                                st.rerun()
                            else:
                                st.warning("❌ No design found within tolerance")
                                with st.expander("📋 Optimization Log"):
                                    st.code("\n".join(log))

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback

                            st.code(traceback.format_exc())
                else:
                    st.warning("Enter requirements first")

            # Show last result if available
            if (
                hasattr(st.session_state, "last_optimization_result")
                and st.session_state.last_optimization_result
            ):
                with st.expander("📋 Last Optimization Log", expanded=False):
                    if hasattr(st.session_state, "last_optimization_log"):
                        st.code("\n".join(st.session_state.last_optimization_log))

            render_divider()

            # OpenAI API Key
            with st.expander("🔑 API Configuration"):
                openai_key = st.text_input(
                    "OpenAI API Key",
                    value=st.session_state.get("openai_api_key", ""),
                    type="password",
                )
                if openai_key:
                    st.session_state.openai_api_key = openai_key
                    if st.session_state.ai_designer:
                        st.session_state.ai_designer.openai_api_key = openai_key
                        st.session_state.ai_designer.use_openai = True

        # ─────────────────────────────────────────────────────────────────────
        # MOTOR DATABASE
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("🔧 Motor Database", expanded=False):
            if st.session_state.motor_database:
                st.markdown(
                    f"""
                <div class="status-badge ready">
                    ✓ {len(st.session_state.motor_database)} motors loaded
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                <div class="status-badge warning">
                    ⚠ No motors loaded
                </div>
                """,
                    unsafe_allow_html=True,
                )

            if st.button("🌐 Download Motors", type="primary", use_container_width=True):
                with st.spinner("Downloading..."):
                    try:
                        scraper = ThrustCurveScraper()
                        motors = scraper.scrape_motor_list(max_motors=10000)
                        if motors:
                            scraper.save_motor_database(motors)
                            st.session_state.motor_database = motors
                            st.success(f"✅ {len(motors)} motors")
                            st.balloons()
                    except Exception as e:
                        st.error(str(e))

            if st.button("📥 Load from Cache", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        scraper = ThrustCurveScraper()
                        motor_db = scraper.load_motor_database()
                        if motor_db:
                            st.session_state.motor_database = motor_db
                            st.success(f"✅ {len(motor_db)} motors")
                    except Exception as e:
                        st.error(str(e))

        # ─────────────────────────────────────────────────────────────────────
        # CUSTOM ROCKET CONFIGURATION
        # ─────────────────────────────────────────────────────────────────────
        if rocket_type == "Custom Rocket":
            render_divider()
            st.markdown("### 📐 Rocket Configuration")

            # Nose Cone
            with st.expander("🔺 Nose Cone", expanded=True):
                has_nose = st.checkbox("Include Nose Cone", value=True)
                if has_nose:
                    nose_length = st.number_input(
                        "Length (m)", 0.1, 2.0, 0.558, 0.01, key="nose_len"
                    )
                    nose_shape = st.selectbox(
                        "Shape", ["VON_KARMAN", "OGIVE", "CONICAL", "ELLIPSOID", "HAACK"]
                    )
                    nose_material = st.selectbox(
                        "Material", list(MATERIALS.keys()), index=3, key="nose_mat"
                    )

            # Body Tube
            with st.expander("📦 Body Tube", expanded=True):
                body_length = st.number_input("Length (m)", 0.1, 5.0, 1.5, 0.1, key="body_len")
                body_radius = st.number_input(
                    "Radius (m)", 0.01, 0.5, 0.0635, 0.001, key="body_rad"
                )
                body_material = st.selectbox(
                    "Material", list(MATERIALS.keys()), index=3, key="body_mat"
                )

            # Fins
            with st.expander("🔱 Fins", expanded=True):
                has_fins = st.checkbox("Include Fins", value=True)
                if has_fins:
                    fin_count = st.number_input("Count", 2, 8, 4, 1, key="fin_count")
                    fin_root_chord = st.number_input(
                        "Root Chord (m)", 0.01, 0.5, 0.12, 0.01, key="fin_root"
                    )
                    fin_tip_chord = st.number_input(
                        "Tip Chord (m)", 0.01, 0.5, 0.06, 0.01, key="fin_tip"
                    )
                    fin_span = st.number_input("Span (m)", 0.01, 0.5, 0.11, 0.01, key="fin_span")
                    fin_sweep = st.number_input("Sweep (m)", 0.0, 0.5, 0.06, 0.01, key="fin_sweep")

            # Parachutes
            with st.expander("🪂 Recovery", expanded=False):
                has_main_chute = st.checkbox("Main Parachute", value=True)
                if has_main_chute:
                    main_chute_diameter = st.number_input("Main Diameter (m)", 0.1, 10.0, 2.91, 0.1)
                    main_deployment_altitude = st.number_input(
                        "Deploy Altitude (m)", 0.0, 10000.0, 800.0, 10.0
                    )

                has_drogue = st.checkbox("Drogue Parachute", value=True)
                if has_drogue:
                    drogue_diameter = st.number_input("Drogue Diameter (m)", 0.1, 5.0, 0.99, 0.1)

            # Store config
            rocket_config = {
                "name": "Custom Rocket",
                "has_nose": has_nose,
                "nose_length": nose_length if has_nose else 0.5,
                "nose_thickness": 0.003,
                "nose_shape": nose_shape if has_nose else "VON_KARMAN",
                "nose_material": nose_material if has_nose else "Fiberglass",
                "body_length": body_length,
                "body_radius": body_radius,
                "body_thickness": 0.003,
                "body_material": body_material,
                "has_fins": has_fins,
                "fin_count": fin_count if has_fins else 4,
                "fin_root_chord": fin_root_chord if has_fins else 0.12,
                "fin_tip_chord": fin_tip_chord if has_fins else 0.06,
                "fin_span": fin_span if has_fins else 0.11,
                "fin_sweep": fin_sweep if has_fins else 0.06,
                "fin_thickness": 0.005,
                "fin_material": "Fiberglass",
                "has_motor_mount": True,
                "motor_mount_length": 0.5,
                "motor_mount_radius": 0.041,
                "motor_mount_thickness": 0.003,
                "motor_mount_material": "Fiberglass",
                "has_main_chute": has_main_chute,
                "main_chute_diameter": main_chute_diameter if has_main_chute else 2.91,
                "main_chute_cd": 1.5,
                "main_deployment_event": "ALTITUDE",
                "main_deployment_altitude": main_deployment_altitude if has_main_chute else 800.0,
                "main_deployment_delay": 1.5,
                "has_drogue": has_drogue,
                "drogue_diameter": drogue_diameter if has_drogue else 0.99,
                "drogue_cd": 1.3,
                "drogue_deployment_event": "APOGEE",
                "drogue_deployment_altitude": 0.0,
                "drogue_deployment_delay": 1.5,
            }
            st.session_state.rocket_config = rocket_config

        # ─────────────────────────────────────────────────────────────────────
        # MOTOR SELECTION
        # ─────────────────────────────────────────────────────────────────────
        if rocket_type in ["Custom Rocket", "AI Builder"]:
            render_divider()
            st.markdown("### 🔥 Motor")

            if st.session_state.motor_database and len(st.session_state.motor_database) > 0:
                motor_type = st.radio(
                    "Source", ["Default (M1670)", "Database", "Custom"], horizontal=True
                )

                if motor_type == "Database":
                    motors_by_class = {}
                    for motor in st.session_state.motor_database:
                        if motor.total_impulse > 0:
                            first_char = motor.designation[0] if motor.designation else "?"
                            if first_char.isalpha():
                                if first_char not in motors_by_class:
                                    motors_by_class[first_char] = []
                                motors_by_class[first_char].append(motor)

                    if motors_by_class:
                        selected_class = st.selectbox("Class", sorted(motors_by_class.keys()))
                        motors_in_class = sorted(
                            motors_by_class[selected_class], key=lambda m: m.total_impulse
                        )
                        motor_display = [
                            f"{m.designation} - {m.total_impulse:.0f} N·s" for m in motors_in_class
                        ]
                        selected_idx = st.selectbox(
                            "Motor",
                            range(len(motor_display)),
                            format_func=lambda i: motor_display[i],
                        )
                        selected_motor = motors_in_class[selected_idx]

                        motor_config = {
                            "designation": selected_motor.designation,
                            "manufacturer": selected_motor.manufacturer,
                            "total_impulse": selected_motor.total_impulse,
                            "max_thrust": selected_motor.max_thrust,
                            "avg_thrust": selected_motor.avg_thrust,
                            "burn_time": selected_motor.burn_time,
                            "diameter": selected_motor.diameter,
                            "length": selected_motor.length,
                            "total_mass": selected_motor.total_mass,
                            "propellant_mass": selected_motor.propellant_mass,
                            "thrust_curve": selected_motor.thrust_curve,
                        }
                        st.session_state.motor_config = motor_config

                        st.caption(
                            f"📊 {selected_motor.avg_thrust:.0f} N avg | {selected_motor.burn_time:.2f}s burn"
                        )

                elif motor_type == "Default (M1670)":
                    st.session_state.motor_config = None
            else:
                st.session_state.motor_config = None

        # ─────────────────────────────────────────────────────────────────────
        # LAUNCH CONDITIONS
        # ─────────────────────────────────────────────────────────────────────
        render_divider()
        st.markdown("### 🌍 Environment")

        with st.expander("Launch Conditions", expanded=False):
            elevation = st.number_input("Elevation (m)", 0.0, 5000.0, 1400.0, 10.0)
            rail_length = st.number_input("Rail Length (m)", 0.5, 20.0, 5.2, 0.1)
            inclination_deg = st.number_input("Inclination (°)", 0.0, 90.0, 5.0, 1.0)
            heading_deg = st.number_input("Heading (°)", 0.0, 360.0, 0.0, 1.0)
            max_time = st.number_input("Max Time (s)", 10.0, 1000.0, 200.0, 10.0)
            dt = st.number_input("Time Step (s)", 0.001, 0.1, 0.01, 0.001, format="%.3f")

        return rocket_type, elevation, rail_length, inclination_deg, heading_deg, max_time, dt


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Main application."""
    # Render sidebar and get configuration
    rocket_type, elevation, rail_length, inclination_deg, heading_deg, max_time, dt = (
        render_sidebar()
    )

    # Hero section
    render_hero()

    # Main content layout
    col_main, col_actions = st.columns([4, 1])

    with col_main:
        # Design Preview Section
        if st.session_state.rocket_config:
            config = st.session_state.rocket_config

            st.markdown(
                """
            <div class="section-card">
                <div class="section-title">
                    <span class="section-title-icon">🎨</span>
                    Rocket Design
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Quick specs
            total_length = config.get("nose_length", 0) + config.get("body_length", 0)
            diameter_mm = config.get("body_radius", 0) * 2 * 1000

            spec_cols = st.columns(4)
            with spec_cols[0]:
                st.metric("Length", f"{total_length:.2f} m")
            with spec_cols[1]:
                st.metric("Diameter", f"{diameter_mm:.0f} mm")
            with spec_cols[2]:
                st.metric("Fins", config.get("fin_count", 0))
            with spec_cols[3]:
                motor_name = (
                    st.session_state.motor_config.get("designation", "M1670")
                    if st.session_state.motor_config
                    else "M1670"
                )
                st.metric("Motor", motor_name)

            st.markdown("</div>", unsafe_allow_html=True)

            # Visualization tabs
            if TRIMESH_AVAILABLE:
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(
                    ["🎮 3D Interactive", "🌐 Plotly 3D", "📐 2D Side"]
                )
            else:
                viz_tab1, viz_tab2 = st.tabs(["🌐 3D View", "📐 2D Side View"])
                viz_tab3 = None

            motor_config = st.session_state.motor_config if st.session_state.motor_config else None

            with viz_tab1:
                if TRIMESH_AVAILABLE:
                    # Real 3D with Three.js
                    try:
                        html = get_rocket_preview_html(config, motor_config)
                        st.components.v1.html(html, height=450)

                        # Export buttons
                        export_cols = st.columns(3)
                        with export_cols[0]:
                            from mesh_renderer import export_stl

                            stl_data = export_stl(config, motor_config)
                            st.download_button(
                                "📥 Download STL",
                                data=stl_data,
                                file_name=f"{config.get('name', 'rocket')}.stl",
                                mime="application/octet-stream",
                            )
                        with export_cols[1]:
                            from mesh_renderer import export_gltf

                            gltf_data = export_gltf(config, motor_config)
                            st.download_button(
                                "📥 Download GLTF",
                                data=gltf_data,
                                file_name=f"{config.get('name', 'rocket')}.glb",
                                mime="model/gltf-binary",
                            )
                        with export_cols[2]:
                            from mesh_renderer import export_obj

                            obj_data = export_obj(config, motor_config)
                            st.download_button(
                                "📥 Download OBJ",
                                data=obj_data,
                                file_name=f"{config.get('name', 'rocket')}.obj",
                                mime="text/plain",
                            )
                    except Exception as e:
                        st.warning(f"3D mesh error: {e}")
                        # Fallback to Plotly
                        try:
                            fig_3d = visualize_rocket_3d(config, motor_config)
                            fig_3d.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                scene=dict(bgcolor="rgba(10, 14, 23, 0.8)"),
                                font=dict(family="Outfit", color=COLORS["text"]),
                            )
                            st.plotly_chart(fig_3d, use_container_width=True)
                        except Exception:
                            pass
                else:
                    # Fallback Plotly 3D
                    try:
                        fig_3d = visualize_rocket_3d(config, motor_config)
                        fig_3d.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            scene=dict(bgcolor="rgba(10, 14, 23, 0.8)"),
                            font=dict(family="Outfit", color=COLORS["text"]),
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate 3D visualization: {str(e)}")

            with viz_tab2:
                if TRIMESH_AVAILABLE:
                    # Plotly with mesh-based rendering
                    try:
                        fig_3d = render_to_plotly(config, motor_config)
                        st.plotly_chart(fig_3d, use_container_width=True)
                    except Exception:
                        # Fallback
                        try:
                            fig_3d = visualize_rocket_3d(config, motor_config)
                            fig_3d.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                scene=dict(bgcolor="rgba(10, 14, 23, 0.8)"),
                                font=dict(family="Outfit", color=COLORS["text"]),
                            )
                            st.plotly_chart(fig_3d, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Error: {e}")
                else:
                    try:
                        fig_2d = visualize_rocket_2d_side_view(config, motor_config)
                        fig_2d.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(19, 27, 46, 0.5)",
                            font=dict(family="Outfit", color=COLORS["text"]),
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate 2D visualization: {str(e)}")

            if viz_tab3:
                with viz_tab3:
                    try:
                        fig_2d = visualize_rocket_2d_side_view(config, motor_config)
                        fig_2d.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(19, 27, 46, 0.5)",
                            font=dict(family="Outfit", color=COLORS["text"]),
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate 2D visualization: {str(e)}")

        # Simulation Results
        if st.session_state.simulation_result is not None:
            render_divider()
            render_section_header("📊", "Simulation Results")
            visualize_results(st.session_state.simulation_result)

            # Export section
            render_divider()
            with st.expander("📥 Export Data"):
                history = st.session_state.simulation_result.history
                df = pd.DataFrame(
                    {
                        "time": [s.time for s in history],
                        "x": [s.x for s in history],
                        "y": [s.y for s in history],
                        "z": [s.z for s in history],
                        "velocity": [np.linalg.norm(s.velocity) for s in history],
                        "mach": [s.mach for s in history],
                    }
                )
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name="rocket_simulation.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with col_actions:
        st.markdown(
            """
        <div class="section-card" style="position: sticky; top: 1rem;">
            <div class="section-title">
                <span class="section-title-icon">⚡</span>
                Actions
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Run Simulation Button
        if st.button("🚀 LAUNCH", type="primary", use_container_width=True):
            with st.spinner("Running simulation..."):
                try:
                    # Build rocket
                    if rocket_type == "Calisto (Default)":
                        rocket_raw, motor_raw = build_calisto()
                    elif st.session_state.rocket_config:
                        rocket_raw = build_custom_rocket(st.session_state.rocket_config)
                        if st.session_state.motor_config:
                            motor_raw = build_custom_motor(st.session_state.motor_config)
                        else:
                            _, motor_raw = build_calisto()
                    else:
                        st.error("Configure rocket first")
                        st.stop()

                    rocket = RocketModel(rocket_raw)
                    motor = Motor.from_openrocket(motor_raw)
                    env = Environment(elevation=elevation)

                    solver = FlightSolver(
                        rocket=rocket,
                        motor=motor,
                        environment=env,
                        rail_length=rail_length,
                        inclination_deg=inclination_deg,
                        heading_deg=heading_deg,
                        dt=dt,
                    )

                    result = solver.run(max_time=max_time)

                    if len(result.history) > 0:
                        st.session_state.simulation_result = result
                        st.session_state.solver = solver
                        st.success("✅ Success!")
                        st.rerun()
                    else:
                        st.error("Simulation failed")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Elodin Editor Button
        if st.session_state.simulation_result is not None:
            if st.button("🎮 Elodin Editor", use_container_width=True):
                if "solver" in st.session_state and st.session_state.solver:
                    launch_elodin_editor(
                        st.session_state.simulation_result, st.session_state.solver
                    )
                else:
                    st.error("Run simulation first")

        st.markdown("</div>", unsafe_allow_html=True)

        # Status
        if st.session_state.simulation_result:
            result = st.session_state.simulation_result
            max_alt = max(s.z for s in result.history)
            st.markdown(
                f"""
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(0, 255, 136, 0.1); 
                        border: 1px solid var(--success); border-radius: 8px; text-align: center;">
                <div style="color: var(--success); font-size: 0.8rem; font-weight: 600;">APOGEE</div>
                <div style="font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                    {max_alt:,.0f} m
                </div>
                <div style="color: var(--text-muted); font-size: 0.75rem;">
                    ({max_alt * 3.28084:,.0f} ft)
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
