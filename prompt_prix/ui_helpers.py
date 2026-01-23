"""
UI helper constants and functions for prompt-prix.

Separates CSS, JS, and utility functions from ui.py for cleaner organization.
"""

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Battery grid styling */
#battery-grid table {
    font-family: monospace;
    font-size: 14px;
}
#battery-grid td {
    text-align: center;
    min-width: 80px;
}

/* Status colors for Compare tab */
#model-tabs button.tab-pending {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
    border-left: 4px solid #ef4444 !important;
}
#model-tabs button.tab-streaming {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
    border-left: 4px solid #f59e0b !important;
    animation: pulse 1.5s ease-in-out infinite;
}
#model-tabs button.tab-completed {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
    border-left: 4px solid #10b981 !important;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Hero grid prominence */
#battery-grid {
    min-height: 300px;
}

/* Compact config panels */
.config-row {
    gap: 1rem;
}
"""

# ─────────────────────────────────────────────────────────────────────
# JAVASCRIPT: Tab Status Colors
# ─────────────────────────────────────────────────────────────────────

TAB_STATUS_JS = """
function updateTabColors(tabStates) {
    if (!tabStates) return tabStates;
    const tabContainer = document.getElementById('model-tabs');
    if (!tabContainer) return tabStates;
    const buttons = tabContainer.querySelectorAll('button[role="tab"]');
    tabStates.forEach((status, index) => {
        if (index < buttons.length) {
            const btn = buttons[index];
            if (status === 'pending') {
                btn.style.background = 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)';
                btn.style.borderLeft = '4px solid #ef4444';
                btn.style.animation = '';
            } else if (status === 'streaming') {
                btn.style.background = 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)';
                btn.style.borderLeft = '4px solid #f59e0b';
                btn.style.animation = 'pulse 1.5s ease-in-out infinite';
            } else if (status === 'completed') {
                btn.style.background = 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)';
                btn.style.borderLeft = '4px solid #10b981';
                btn.style.animation = '';
            } else {
                btn.style.background = '';
                btn.style.borderLeft = '';
                btn.style.animation = '';
            }
        }
    });
    return tabStates;
}
"""

# ─────────────────────────────────────────────────────────────────────
# JAVASCRIPT: LocalStorage Persistence
# ─────────────────────────────────────────────────────────────────────

# Load settings from localStorage on app startup.
# Returns undefined for missing keys to preserve Python defaults.
# Guards against race condition where user edits before load completes.
PERSISTENCE_LOAD_JS = """
() => {
    // Skip load if user has already started editing (race condition guard)
    if (window._promptprix_user_edited) {
        return undefined;
    }

    const servers = localStorage.getItem('promptprix_servers');

    // Return undefined to preserve Python defaults when localStorage is empty.
    return servers || undefined;
}
"""

# Save servers to localStorage on change.
# Sets flag to prevent race condition with async load event.
SAVE_SERVERS_JS = """
(servers) => {
    window._promptprix_user_edited = true;  // Guard against load race
    if (servers && servers.trim()) {
        localStorage.setItem('promptprix_servers', servers);
    }
    return servers;
}
"""

# Save temperature to localStorage
SAVE_TEMPERATURE_JS = """
(temp) => {
    if (temp !== null && temp !== undefined) {
        localStorage.setItem('promptprix_temperature', temp.toString());
    }
    return temp;
}
"""

# Snapshot current values to localStorage if not already saved.
# Called on first interaction to capture .env defaults.
SNAPSHOT_IF_EMPTY_JS = """
(servers, temp) => {
    if (!localStorage.getItem('promptprix_servers') && servers && servers.trim()) {
        localStorage.setItem('promptprix_servers', servers);
    }
    if (!localStorage.getItem('promptprix_temperature') && temp !== null && temp !== undefined) {
        localStorage.setItem('promptprix_temperature', temp.toString());
    }
    return [servers, temp];
}
"""

# ─────────────────────────────────────────────────────────────────────
# JAVASCRIPT: Auto-Download for Exports
# ─────────────────────────────────────────────────────────────────────

# Triggers immediate download when a file is ready.
# Uses fileData.url directly instead of DOM scraping (more reliable in Gradio 6.x).
AUTO_DOWNLOAD_JS = """
(fileData) => {
    if (!fileData) return fileData;

    // Gradio 6.x FileData: {path, url, orig_name, size, mime_type, ...}
    // Use the URL directly to trigger download
    const url = fileData.url;
    const filename = fileData.orig_name || fileData.path?.split('/').pop() || 'download';

    if (url) {
        // Create hidden link and trigger download
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    return fileData;
}
"""
