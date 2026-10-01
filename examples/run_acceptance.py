"""Execute the offline notebook with this Python, writing an inspected artifact.

Run from any directory: python /path/to/research-toolkit/examples/run_acceptance.py
The source notebook is never overwritten; output lives in artifacts/acceptance/.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    os.environ.setdefault("IPYTHONDIR", str(output / "ipython"))
    manager = KernelManager()
    manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                     display_name="Acceptance Python", language="python")
    notebook = nbformat.read(root / "examples" / "buy_and_hold_equities.ipynb", as_version=4)
    client = NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    image_count = sum("image/png" in item.get("data", {})
                      for cell in notebook.cells for item in cell.get("outputs", []))
    if image_count < 9:
        raise RuntimeError(f"Expected at least nine inline chart images; found {image_count}")
    nbformat.write(notebook, output / "buy_and_hold_equities.executed.ipynb")
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    (output / "run-provenance.json").write_text(json.dumps({
        "executed_at": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable,
        "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
    }, indent=2))
    print(output / "buy_and_hold_equities.executed.ipynb")


if __name__ == "__main__":
    main()
