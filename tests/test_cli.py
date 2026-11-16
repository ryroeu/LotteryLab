"""Every command runs end to end and says something."""

import pytest

from lotterylab import store
from lotterylab.cli import main

COMMANDS = [
    ["odds"],
    ["wait"],
    ["prove", "eurodreams", "--synth", "--synth-n", "800"],
    ["backtest", "loto", "cold", "--synth", "--synth-n", "400", "--tickets", "2"],
    ["wheel", "eurodreams", "-n", "8"],
    ["wheel", "eurodreams", "-n", "30"],
    ["wheel", "loto", "--numbers", "4", "11", "19", "27", "35", "38"],
    ["freq", "loto", "--synth"],
    ["variance", "eurodreams", "--seasons", "200"],
    ["payouts", "powerball"],
    ["popular", "powerball"],
    ["ev", "powerball"],
]
DATA_COMMANDS = [
    ["payouts", "loto"],
    ["popular", "euromillions"],
    ["ev", "loto", "--numbers", "3", "7", "12", "19", "31", "--jackpot", "13000000"],
    ["ev", "eurodreams"],
    ["sharing", "loto"],
]


@pytest.mark.parametrize("argv", COMMANDS, ids=" ".join)
def test_command_runs(argv, capsys):
    main(argv)
    assert len(capsys.readouterr().out) > 80


@pytest.mark.parametrize("argv", DATA_COMMANDS, ids=" ".join)
def test_data_command_runs(argv, capsys):
    if not store.all_snapshots(argv[1]):
        pytest.skip("no snapshot on disk")
    main(argv)
    assert len(capsys.readouterr().out) > 200


def test_bad_wheel_numbers_exit_cleanly():
    with pytest.raises(SystemExit):
        main(["wheel", "eurodreams", "--numbers", "1", "2", "3"])
    with pytest.raises(SystemExit):
        main(["ev", "loto", "--numbers", "1", "2"])
