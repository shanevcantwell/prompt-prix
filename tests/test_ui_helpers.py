"""Tests for UI helper constants and functions."""

from prompt_prix.ui_helpers import TAB_STATUS_JS


class TestTabStatusJS:
    """Tests for TAB_STATUS_JS JavaScript constant."""

    def test_does_not_hide_panels(self):
        """
        Regression test for #44: Panel hiding broke Gradio tab rendering.

        The JS should only hide tab buttons, not panels. Gradio manages
        panel visibility internally, and manually hiding panels interferes
        with content rendering during streaming.
        """
        # Should not contain panel hiding logic
        assert "panels[index].style.display" not in TAB_STATUS_JS
        assert 'querySelectorAll(\'[role="tabpanel"]\')' not in TAB_STATUS_JS

    def test_hides_tab_buttons_for_empty_status(self):
        """Tab buttons should be hidden when status is empty."""
        assert "btn.style.display = hasContent" in TAB_STATUS_JS

    def test_updates_tab_labels_with_model_names(self):
        """Tab labels should be updated with truncated model names."""
        assert "btn.textContent = displayName" in TAB_STATUS_JS
        assert "substring(0, 22)" in TAB_STATUS_JS  # Truncation logic

    def test_applies_status_colors(self):
        """Tab buttons should get colored backgrounds based on status."""
        assert "'pending'" in TAB_STATUS_JS
        assert "'streaming'" in TAB_STATUS_JS
        assert "'completed'" in TAB_STATUS_JS

    def test_sets_black_text_color(self):
        """
        Regression test for #42: Tab text should be black for readability.
        """
        # Count occurrences - should be in all status branches
        black_color_count = TAB_STATUS_JS.count("btn.style.color = 'black'")
        assert black_color_count >= 3, "Black text color should be set for all status types"
