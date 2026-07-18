"""Small Python test-source checks for memory scenario regression tests."""


def analyze_python_test_source(source: str, *, path: str = "") -> list[dict[str, str]]:
    del path
    issues: list[dict[str, str]] = []
    if "tk.Button()" in source or "tkinter.Button()" in source:
        issues.append(
            {
                "code": "masterless_gui_widget_in_test",
                "message": "Tk widgets in tests need an explicit root/master.",
            }
        )
    return issues
