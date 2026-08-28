"""CLI parsing and dispatch."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from reality import cli
from reality.core.version import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]
KITTI_CONFIG = REPO_ROOT / "reality" / "configs" / "voxelscape_to_kitti.yaml"

COMMANDS = ["prepare-data", "train", "generate", "evaluate"]
#: All four commands are implemented; none are stubs.
#: Every command is implemented as of ; none are stubs.
STUB_COMMANDS: list = []


def argv_for(command, config):
    argv = [command, "--config", str(config)]
    if command in ("generate", "evaluate"):
        argv += ["--checkpoint", "checkpoints/model.pt"]
    return argv


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_dispatches_to_its_handler(command, monkeypatch):
    called = {}

    def fake(args):
        called["command"] = args.command
        called["config"] = args.config
        return 0

    monkeypatch.setitem(cli.HANDLERS, command, fake)
    assert cli.main(argv_for(command, KITTI_CONFIG)) == 0
    assert called == {"command": command, "config": str(KITTI_CONFIG)}


def test_weather_run_is_described_with_its_degradation_stage(capsys):
    """The run summary names the stages; training itself is not started here."""
    from reality.core.config import Config

    config = Config.load(REPO_ROOT / "reality/configs/weather/voxelscape_to_cadc.yaml")
    cli._describe("train", config)
    out = capsys.readouterr().out
    assert "task=weather" in out
    assert "degradation(degradations:physics)" in out


@pytest.mark.parametrize("command", ["train", "prepare-data", "generate", "evaluate"])
def test_implemented_commands_dispatch_to_their_handler(command, monkeypatch):
    """train and prepare-data are real handlers as of """
    called = {}
    monkeypatch.setitem(cli.HANDLERS, command,
                        lambda args: called.setdefault("config", args.config) and 0 or 0)
    assert cli.main(argv_for(command, KITTI_CONFIG)) == 0
    assert called["config"] == str(KITTI_CONFIG)


def test_train_accepts_its_orchestration_flags():
    parser = cli.build_parser()
    args = parser.parse_args(["train", "--config", "c.yaml", "--epochs", "3",
                              "--no-resume", "--checkpoint-every", "5",
                              "--cache-root", "/scratch/cache", "--limit", "8",
                              "--force-prepare"])
    assert args.epochs == 3 and args.no_resume is True
    assert args.checkpoint_every == 5 and args.cache_root == "/scratch/cache"
    assert args.limit == 8 and args.force_prepare is True


def test_generate_accepts_its_flags():
    args = cli.build_parser().parse_args(
        ["generate", "--config", "c.yaml", "--checkpoint", "gen_r.pt",
         "--split", "test", "--output", "out", "--format", "npy", "--limit", "10"])
    assert args.split == "test" and args.format == "npy" and args.limit == 10
    assert args.checkpoint == "gen_r.pt"


def test_evaluate_accepts_its_flags():
    args = cli.build_parser().parse_args(
        ["evaluate", "--config", "c.yaml", "--checkpoint", "g.pt",
         "--label", "an earlier variant", "--no-clamp"])
    assert args.label == "an earlier variant" and args.clamp is False


def test_every_command_is_implemented():
    """ completes the CLI: train, prepare-data, generate, evaluate."""
    assert set(cli.HANDLERS) == {"train", "prepare-data", "generate", "evaluate"}


def test_prepare_data_accepts_its_flags():
    args = cli.build_parser().parse_args(
        ["prepare-data", "--config", "c.yaml", "--force", "--limit", "4"])
    assert args.force is True and args.limit == 4


def test_config_error_is_reported_cleanly(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"source": {"dataset": "voxelscape"}}))
    assert cli.main(["train", "--config", str(bad)]) == cli.EXIT_CONFIG_ERROR
    err = capsys.readouterr().err
    assert "config error" in err and "target" in err


def test_missing_config_file_is_reported_cleanly(tmp_path, capsys):
    code = cli.main(["train", "--config", str(tmp_path / "nope.yaml")])
    assert code == cli.EXIT_CONFIG_ERROR
    assert "not found" in capsys.readouterr().err


def test_no_command_prints_help(capsys):
    assert cli.main([]) == cli.EXIT_OK
    assert "train" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["frobnicate", "--config", "c.yaml"],
        ["train"],                       # --config is required
        ["generate", "--config", "c.yaml"],  # --checkpoint is required
    ],
)
def test_usage_errors_exit_two(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_help_and_version_exit_zero(capsys):
    for argv in (["--help"], ["--version"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_generate_takes_a_checkpoint(monkeypatch):
    seen = {}
    monkeypatch.setitem(cli.HANDLERS, "generate", lambda args: seen.setdefault("ckpt", args.checkpoint) and 0)
    cli.main(["generate", "--config", str(KITTI_CONFIG), "--checkpoint", "w.pt"])
    assert seen["ckpt"] == "w.pt"


def test_module_entry_point_runs():
    """`python -m reality` is the documented entry point."""
    proc = subprocess.run(
        [sys.executable, "-m", "reality", "--version"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert __version__ in proc.stdout
