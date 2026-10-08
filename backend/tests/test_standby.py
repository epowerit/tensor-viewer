from fastapi.testclient import TestClient

from tensorviewer import runner
from tensorviewer.app import create_app
from tests.test_logit_lens import CODE, recorded

# The model prints where it runs, so a test can see the job's own folder.
WHERE = CODE.replace(
    "    def forward(self, tokens):\n",
    "    def forward(self, tokens):\n        import os\n        print(os.getcwd())\n",
)


def test_a_run_takes_the_waiting_worker_and_runs_in_its_own_folder(tmp_path):
    client = TestClient(create_app(tmp_path))
    runner.warm(runner.WORKER)
    waiting = runner._standby[runner.WORKER]
    assert waiting.poll() is None
    run = recorded(client, WHERE)
    assert run["trace"]["error"] is None
    # The waiting worker took the job, finished, and another now waits.
    assert waiting.poll() is not None
    assert runner._standby[runner.WORKER] is not waiting
    # It ran in the job's folder, as a fresh worker would, not the server's.
    folder = run["trace"]["stdout"].strip()
    assert "tensorviewer-run-" in folder
    assert folder != str(runner.BACKEND_ROOT)


def test_a_worker_that_died_waiting_is_passed_over(tmp_path):
    client = TestClient(create_app(tmp_path))
    runner.warm(runner.WORKER)
    waiting = runner._standby[runner.WORKER]
    waiting.kill()
    waiting.wait()
    run = recorded(client)
    assert run["trace"]["error"] is None
    assert len(run["trace"]["operations"]) > 0


def test_a_watch_takes_its_own_waiting_worker(tmp_path):
    client = TestClient(create_app(tmp_path))
    run = recorded(client)
    runner.warm(runner.EVALUATOR)
    waiting = runner._standby[runner.EVALUATOR]
    answer = client.post(
        f"/api/v1/runs/{run['id']}/evaluate", json={"expression": "1 + 1", "at": None}
    ).json()
    assert answer["kind"] != "error", answer
    assert waiting.poll() is not None
