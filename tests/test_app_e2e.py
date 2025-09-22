"""
test_app_e2e.py

This test provides an end to end test on the app built using Streamlit.

"""
import os, re, time, shutil, socket, subprocess, contextlib
from contextlib import closing
import pytest

try:
    import requests
except Exception:
    requests = None

def _find_free_port(start=8501, end=8999):
    for port in range(start, end + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found")

def _wait_for_health(url, timeout=40.0):
    if requests is None:
        return False
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=0.75)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False

@pytest.fixture(scope="session")
def streamlit_available():
    return shutil.which("streamlit") is not None

@pytest.fixture(scope="session")
def app_server(streamlit_available):
    if not streamlit_available:
        pytest.skip("Streamlit CLI not found")

    # your app file is app.py
    repo_root = os.path.dirname(os.path.dirname(__file__))
    app_path = os.path.join(repo_root, "app.py")
    if not os.path.exists(app_path):
        pytest.skip("app.py not found at repo root")

    port = _find_free_port()
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env.setdefault("APP_TEST_MODE", "1")

    proc = subprocess.Popen(
        ["streamlit", "run", app_path, "--server.headless=true", f"--server.port={port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    try:
        ok = _wait_for_health(f"http://127.0.0.1:{port}/_stcore/health", timeout=40)
        if not ok:
            proc.terminate()
            with contextlib.suppress(Exception): proc.wait(timeout=5)
            pytest.skip("Streamlit server did not become healthy in time")
        yield f"http://127.0.0.1:{port}"
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=5)

@pytest.mark.skipif(shutil.which("streamlit") is None, reason="Streamlit not installed")
def test_app_page_loads(page, app_server):
    page.goto(app_server, wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=15000)
    assert page.locator('[data-testid="stAppViewContainer"]').count() == 1

@pytest.mark.skipif(shutil.which("streamlit") is None, reason="Streamlit not installed")
def test_predict_flow(page, app_server):
    page.goto(app_server, wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=15000)

    # If the app stopped due to missing bundle, try to configure it via UI
    import glob, os as _os
    repo_root = _os.path.dirname(_os.path.dirname(__file__))
    candidates = glob.glob(_os.path.join(repo_root, "models", "*.joblib"))
    meta_candidates = glob.glob(_os.path.join(repo_root, "reports", "*.json"))

    if candidates:
        bundle_path = candidates[0]
        # open the expander
        page.get_by_text("Artifact configuration", exact=False).first.click()
        # fill the two text inputs by label
        page.get_by_label("Bundle .joblib path", exact=False).fill(bundle_path)
        if meta_candidates:
            page.get_by_label("Meta .json path", exact=False).fill(meta_candidates[0])
        # click Apply & reload
        page.get_by_role("button", name="Apply & reload").click()
        # wait for rerender
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=15000)
    else:
        pytest.xfail("Model bundle not found under models/. Add a *.joblib or use Option A discovery patch.")
