import logging
import subprocess
import sys


PLAIN_OUTPUT_CODE = (
    "import logging\n"
    "import perturbvi\n"
    "logging.getLogger('perturbvi.test').info('progress line')\n"
)


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_timestamped_output_without_any_logging_setup():
    """Plain scripts and notebooks get CLI-style timestamped progress."""
    result = _run(PLAIN_OUTPUT_CODE)
    assert result.returncode == 0, result.stderr
    stderr = result.stderr
    assert "[20" in stderr and "- INFO] progress line" in stderr


def test_bare_import_attaches_exactly_one_console_handler():
    code = (
        "import logging\n"
        "import perturbvi\n"
        "logger = logging.getLogger('perturbvi')\n"
        "handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]\n"
        "assert len(handlers) == 1, handlers\n"
        "assert logger.propagate is False\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_preconfigured_host_is_not_hijacked():
    code = (
        "import logging\n"
        "logging.basicConfig()\n"
        "import perturbvi\n"
        "logger = logging.getLogger('perturbvi')\n"
        "assert not [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]\n"
        "assert logger.propagate is True\n"
        "assert logger.level == logging.NOTSET\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_set_verbose_routes_chatter():
    code = (
        "import logging\n"
        "from perturbvi.log import set_verbose\n"
        "chatter = logging.getLogger('perturbvi.verbose')\n"
        "set_verbose(True)\n"
        "assert chatter.level == logging.INFO\n"
        "set_verbose(False)\n"
        "assert chatter.level == logging.WARNING\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def _tiny_data():
    import numpy as np

    from perturbvi import PerturbData

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 30)).astype(np.float32)
    G = np.zeros((60, 4), dtype=np.float32)
    G[np.arange(60) % 4, np.arange(60) % 4] = 1
    return PerturbData(
        X=X,
        G=G,
        gene_names=tuple(f"g{i}" for i in range(30)),
        perturbation_names=("p1", "p2", "p3", "p4"),
    )


def test_verbose_false_keeps_milestones_hides_iter(caplog):
    from perturbvi import fit_screen

    with caplog.at_level(logging.NOTSET):
        fit_screen(_tiny_data(), z_dim=2, l_dim=5, tau=1.0, max_iter=3, verbose=False)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting model parameter initialization" in m for m in messages)
    assert any("initialization completed successfully" in m for m in messages)
    assert not any(m.startswith("Iter [") for m in messages)
    assert not any("Base parameters initialized" in m for m in messages)
    milestone_records = [r for r in caplog.records if "initialization completed" in r.getMessage()]
    assert all(r.levelname == "INFO" for r in milestone_records)


def test_verbose_true_shows_iter_and_micro_steps(caplog):
    from perturbvi import fit_screen

    with caplog.at_level(logging.NOTSET):
        fit_screen(_tiny_data(), z_dim=2, l_dim=5, tau=1.0, max_iter=3, verbose=True)

    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("Iter [") for m in messages)
    assert any("Factors initialized (35%)" in m for m in messages)
    iter_records = [r for r in caplog.records if r.getMessage().startswith("Iter [")]
    assert all(r.levelname == "INFO" for r in iter_records)


def test_get_logger_returns_logger():
    from perturbvi.log import get_logger

    name = "perturbvi.test.plain"
    logger = get_logger(name)
    assert isinstance(logger, logging.Logger)
    assert logger.name == name
