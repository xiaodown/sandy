from sandy.paths import web_root


def test_dashboard_page_contains_expected_endpoints() -> None:
    html = (web_root() / "dashboard" / "index.html").read_text(encoding="utf-8")

    assert "Sandy Control Panel" in html
    assert "/favicon.svg" in html
    assert "/dashboard/images/v3.png" in html
    assert "/dashboard/styles.css" in html
    assert "/dashboard/app.js" in html
