"""
Capacity accounting models for the sowing crew.

The headline result of the original study depends on how the crew's limited
capacity is counted.  Under the original model the scarce resource is the
FIELD VISIT -- the supervised unit travelling to and setting up at an
isolated contract field -- so sowing the male and the female on the same day
costs one visit instead of two, and the optimiser exploits that to serve more
fields.  Whether that saving is real depends on what actually constrains the
crew, which is an empirical question nobody has answered for this system.

Rather than assume, this module makes the accounting an explicit experimental
factor with four interpretations:

  visits      one trip to a field costs the same whatever is sown there
  operations  each parent sowing costs the same; trips are free
  hours       a trip costs travel + setup time, each sowing costs machine
              time; the realistic middle case, and the one that nests the
              other two as limits (h_op = 0 gives visits, h_travel = 0 gives
              operations)
  hectares    the crew is limited by the area it can drill in a day

Each model answers one question: given that a field needs a female sowing on
day d_f and male sowings on d_f+o and d_f+o+delta, what does it consume on a
given day, and what is the daily budget?

Parameters below are ASSUMPTIONS, not measurements.  They are chosen so that
one crew-day is roughly equivalent across the four models -- about two thirds
of a field per day -- so that gains can be compared at matched resource
pressure rather than at matched nominal capacity.  Where a parameter matters
to a conclusion, the conclusion is reported as a range over the four models.
"""

# Row-area shares of the three sowing operations, for a 2:1 female:male
# planting ratio with the male split over two dates (Chassaigne-Ricciulli et
# al. 2021).  Used only by the hectares model.
SHARE = {"f": 2.0 / 3.0, "m": 1.0 / 6.0, "m2": 1.0 / 6.0}

# Hours model.  Travel and set-up at an isolated contract field, and machine
# time for one parent sowing on a field of mean size.  An eight-hour day.
H_TRAVEL = 1.5
H_OP = 2.0
H_DAY = 8.0


class CapacityModel:
    """How a day's crew capacity is consumed, and how much there is."""

    def __init__(self, name, capacity, note):
        self.name = name
        self.capacity = capacity
        self.note = note

    def serve_cost(self, field, parts):
        """Cost of serving `field` on one day, doing the operations `parts`."""
        raise NotImplementedError

    def field_demand(self, field):
        """
        Reference demand of one field, sown conventionally: three separate
        days, one operation on each.  Used for resource pressure, so that
        pressure does not depend on the schedule the optimiser happens to
        choose.
        """
        return sum(self.serve_cost(field, (p,)) for p in ("f", "m", "m2"))


class VisitModel(CapacityModel):
    def __init__(self, capacity=2.0):
        super().__init__("visits", capacity,
                         "one trip per field per day, whatever is sown")

    def serve_cost(self, field, parts):
        return 1.0


class OperationModel(CapacityModel):
    def __init__(self, capacity=2.0):
        super().__init__("operations", capacity,
                         "each parent sowing costs one slot; trips free")

    def serve_cost(self, field, parts):
        return float(len(parts))


class HourModel(CapacityModel):
    def __init__(self, capacity=H_DAY, h_travel=H_TRAVEL, h_op=H_OP):
        super().__init__("hours", capacity,
                         f"{h_travel} h travel+setup per trip, "
                         f"{h_op} h per sowing, {capacity} h day")
        self.h_travel = h_travel
        self.h_op = h_op

    def serve_cost(self, field, parts):
        return self.h_travel + self.h_op * len(parts)


class HectareModel(CapacityModel):
    def __init__(self, capacity=3.0):
        super().__init__("hectares", capacity,
                         f"{capacity} ha drilled per day")

    def serve_cost(self, field, parts):
        return field.area_ha * sum(SHARE[p] for p in parts)


def make(name):
    return {"visits": VisitModel, "operations": OperationModel,
            "hours": HourModel, "hectares": HectareModel}[name]()


ALL = ["visits", "operations", "hours", "hectares"]


# ---------------------------------------------------------------------------
# Day-by-day cost of a (female day, offset) choice
# ---------------------------------------------------------------------------

def parts_by_day(inst, d_f, o):
    """{day: tuple of operations due that day} for one scheduling choice."""
    d_m = d_f + o
    byday = {}
    byday.setdefault(d_f, []).append("f")
    byday.setdefault(d_m, []).append("m")
    if inst.split_male:
        byday.setdefault(d_m + inst.male_split_gap, []).append("m2")
    return {d: tuple(v) for d, v in byday.items()}


def choice_cost(model, field, inst, d_f, o):
    """{day: capacity consumed} if `field` takes (d_f, o) under `model`."""
    return {d: model.serve_cost(field, parts)
            for d, parts in parts_by_day(inst, d_f, o).items()}


def pressure(model, inst):
    """Resource pressure: demand of all fields / supply over the window."""
    demand = sum(model.field_demand(f) for f in inst.fields)
    return demand / (inst.T * model.capacity)
