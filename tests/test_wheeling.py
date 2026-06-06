"""The covering-design guarantee must actually hold (brute-forced)."""

from itertools import combinations

from lotterylab import games
from lotterylab.wheeling import covering_design, verify_guarantee, wheel_report


def test_small_cover_guarantee_holds():
    numbers = list(range(1, 10))  # wheel 9 numbers
    k, t = 6, 3
    tickets = covering_design(numbers, k, t)
    assert verify_guarantee(tickets, numbers, t)
    # exhaustive: every 3-subset of the 9 is on some ticket
    covered = set()
    for tk in tickets:
        covered.update(combinations(tk, t))
    assert all(ts in covered for ts in combinations(numbers, t))


def test_wheel_report_eurodreams():
    spec = games.get("eurodreams")  # k = 6
    report = wheel_report(spec, list(range(1, 9)))  # wheel 8 numbers
    assert report.n_tickets >= 1
    assert verify_guarantee(report.tickets, list(report.chosen), report.t)
    assert 0 < report.p_condition < 1
    assert report.cost == report.n_tickets * spec.price


def test_full_pool_single_ticket_is_trivial():
    # Wheeling exactly k numbers needs exactly one ticket.
    spec = games.get("powerball")  # k = 5
    report = wheel_report(spec, [3, 11, 19, 27, 35])
    assert report.n_tickets == 1
