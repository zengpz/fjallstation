import math

from fjallstation import score

SPEC = {"model_name": "Room", "msl": "4.1.0", "libraries": ["Buildings 13.0.0"],
        "eval": {"start_time": 0, "stop_time": 10, "interval": 1, "tolerance": 1e-6, "outputs": ["T", "Q"]},
        "hidden_classes": ["Buildings.Fluid.Examples.Room"], "forbidden": ["Fluid.Examples.Room"]}
TIME = [float(i) for i in range(11)]
REF = {"time": TIME, "T": [20.0 + i for i in range(11)], "Q": [100.0 * i for i in range(11)]}
VALUES = {"T": REF["T"], "Q": REF["Q"]}


def test_values_equal_to_the_reference_are_correct():
    assert score.compare(VALUES, REF, SPEC) == {"reward": 1.0, "status": "correct", "outputs": {"T": 0.0, "Q": 0.0}}


def test_an_error_above_one_percent_of_the_range_is_not_correct():
    small = {**VALUES, "T": [v + 0.05 for v in REF["T"]]}       # 0.5% of the range of 10
    large = {**VALUES, "T": [v + 0.2 for v in REF["T"]]}        # 2%
    assert score.compare(small, REF, SPEC)["reward"] == 1.0
    result = score.compare(large, REF, SPEC)
    assert result["reward"] == 0.0 and result["status"] == "outputs differ" and round(result["outputs"]["T"], 3) == 0.02


def test_a_missing_output_a_wrong_count_of_values_and_values_that_are_not_numbers():
    assert score.compare({"T": REF["T"]}, REF, SPEC)["status"] == "output missing"
    assert score.compare(None, REF, SPEC)["status"] == "output missing"
    assert score.compare({**VALUES, "Q": REF["Q"][:6]}, REF, SPEC)["status"] == "result not valid"     # not one value for each time step
    assert score.compare({**VALUES, "Q": "many"}, REF, SPEC)["status"] == "result not valid"
    assert score.compare({**VALUES, "Q": [[1.0]] * 11}, REF, SPEC)["status"] == "result not valid"
    assert score.compare({**VALUES, "Q": [math.nan] * 11}, REF, SPEC) == {"reward": 0.0, "status": "outputs differ", "outputs": {"T": 0.0, "Q": math.inf}}


def test_oversized_integers_are_invalid_results_with_the_simulation_credit():
    for value in (10 ** 400, -(10 ** 400)):
        values = {**VALUES, "T": [value, *REF["T"][1:]]}
        result = score.score({"status": "simulated", "values": values}, SPEC, REF)
        assert result == {"reward": 0.0, "status": "result not valid", "outputs": {}, "omc_reward": 0.3}


def test_the_reference_of_an_older_task_set_is_read_at_the_fixed_time_steps():
    ev = {"start_time": 0, "stop_time": 2, "interval": 1, "outputs": ["y"]}
    # a whole result table has rows at events: two rows with the same time; the value after the event counts
    table = {"time": [0, 1, 1.4, 1.4, 1.5, 1.5, 2], "y": [0, 0, 0, 0, 0, 1, 1]}
    assert score.on_steps(table, "y", score.time_steps(ev)).tolist() == [0.0, 0.0, 1.0]
    at_step = {"time": [0, 1, 1, 2], "y": [0, 0, 1, 1]}
    assert score.on_steps(at_step, "y", score.time_steps(ev)).tolist() == [0.0, 1.0, 1.0]
    gap = {"time": [0, 0.5, 1.5, 2], "y": [0, 0, 1, 1]}           # a time step with no row gets the value between its neighbours
    assert score.on_steps(gap, "y", score.time_steps(ev)).tolist() == [0.0, 0.5, 1.0]


def test_the_reward_has_two_parts():
    result = lambda status, values=None: {"status": status, "omc_reward": 0.3, "values": values or {}}
    assert score.score(result("simulated", VALUES), SPEC, REF) == {"reward": 1.0, "status": "correct", "outputs": {"T": 0.0, "Q": 0.0}, "omc_reward": 0.3}
    wrong = score.score(result("simulated", {**VALUES, "T": [0.0] * 11}), SPEC, REF)
    assert wrong["reward"] == 0.0 and wrong["omc_reward"] == 0.3 and wrong["status"] == "outputs differ"
    # the server takes the OpenModelica part from the status, not from the number in the result file
    assert score.score(result("did not load"), SPEC, REF) == {"reward": 0.0, "omc_reward": 0.0, "status": "did not load", "outputs": {}}
    assert score.score(result("check failed"), SPEC, REF)["omc_reward"] == 0.1
    assert [score.score(result(s), SPEC, REF)["omc_reward"] for s in ("simulation failed", "simulation stopped early", "time limit")] == [0.2, 0.2, 0.2]
    assert score.score(result("no model"), SPEC, REF)["omc_reward"] == 0.0 and score.score(result("uses the source example"), SPEC, REF)["reward"] == 0.0
    assert score.score(result("made up"), SPEC, REF)["status"] == "result not valid" and score.score({}, SPEC, REF)["status"] == "result not valid"
    assert score.score(result("check failed", VALUES), SPEC, REF)["reward"] == 0.0        # values count only for a model that simulates


def test_the_run_part_of_a_task_file_holds_what_the_image_needs_and_no_reference():
    part = score.run_part(SPEC | {"limit": {"T": 0.1}, "files": {"x.txt": ""}})
    assert part == {"msl": "4.1.0", "libraries": ["Buildings 13.0.0"], "eval": SPEC["eval"],
                    "hidden_classes": ["Buildings.Fluid.Examples.Room"], "forbidden": ["Fluid.Examples.Room"]}


def test_a_time_step_with_no_reference_value_is_not_compared():
    spec = {"eval": {"start_time": 0, "stop_time": 4, "interval": 1, "outputs": ["T"]}, "limit": {"T": 0.05}}
    reference = {"T": [math.nan, math.nan, 20.0, 21.0, 22.0]}                # measured data with a gap at the start
    assert score.compare({"T": [0.0, 99.0, 20.0, 21.0, 22.0]}, reference, spec) == {"reward": 1.0, "status": "correct", "outputs": {"T": 0.0}}
    assert score.compare({"T": [20.0, 20.0, 20.0, 21.0, 23.0]}, reference, spec)["status"] == "outputs differ"
    assert score.compare({"T": [20.0, 20.0, math.nan, 21.0, 22.0]}, reference, spec)["outputs"]["T"] == math.inf      # the model must have a value there
    assert score.compare({"T": [1.0] * 5}, {"T": [math.nan] * 5}, spec)["status"] == "outputs differ"                  # a reference with no value scores nothing
