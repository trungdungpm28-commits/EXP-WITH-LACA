"""One-container adaptation of FaasCache LambdaScheduler.calc_priority (GD).

FaasCache copyright (C) 2021 Indiana University. GPL-3.0-or-later.
Upstream commit 25110b6f378cdc0384adc712d7885509ecd1450e.
See LICENSE.md for the upstream license notice.
"""


def priority(clock, frequency, cold_time, warm_time, size):
    cost = float(cold_time - warm_time)
    return clock + frequency * (cost / size)
